import pytest
import random
import os
import math
import numpy as np
import torch
from einops import rearrange, repeat

from samantha.components.ctiga.ops.flash_attn_2_interface import (
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_kvpacked_func,
)
from samantha.utils.ctiga.reconstruct_attn_probs import (
    convert_flash_attn_S_to_softmax,
    normalize_flash_attn_S_torch,
    normalize_flash_attn_S,
    construct_causal_mask,
    reconstruct_attention_probs,
)
from samantha.utils.ctiga.padding import unpad_input, pad_input

MAX_HEADDIM_SM8x = 192
device = "cuda:0"
dtype = torch.float16
# dtype = torch.bfloat16

is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def random_seqlen(bs, max_seqlen, max_diff_len=20):
    max_seqlen_idx = random.randint(0, bs - 1)
    return [
        max_seqlen
        if i == max_seqlen_idx
        else max_seqlen - random.randint(1, min(max_diff_len, max_seqlen))
        for i in range(bs)
    ]


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    upcast=True,
    reorder_ops=False,
):
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if causal:
        # causal_mask = torch.triu(
        #     torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1
        # )
        causal_mask = construct_causal_mask(
            seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, q.device
        )
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    if (
        causal
    ):  # Some rows are completely masked out so we fill them with zero instead of NaN
        attention = attention.masked_fill(
            torch.all(causal_mask, dim=-1, keepdim=True), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    # attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def get_dropout_fraction(
    dropout_mask, query_padding_mask=None, key_padding_mask=None, causal=False
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if causal:
        # causal_mask = torch.triu(
        #     torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=dropout_mask.device), 1
        # )
        causal_mask = construct_causal_mask(
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(causal_mask, False)
    dropped_total = dropped.sum()
    query_lengths = (
        query_padding_mask.sum(dim=-1)
        if query_padding_mask is not None
        else torch.full((batch_size,), seqlen_q, device=dropout_mask.device)
    )
    key_lengths = (
        key_padding_mask.sum(dim=-1)
        if key_padding_mask is not None
        else torch.full((batch_size,), seqlen_k, device=dropout_mask.device)
    )
    if not causal:
        numel_per_batch = query_lengths * key_lengths
    else:
        numel_per_batch = torch.where(
            key_lengths <= query_lengths,
            key_lengths * (key_lengths + 1) / 2,
            query_lengths * key_lengths - (query_lengths * (query_lengths - 1) / 2),
        )
    return dropped_total / (numel_per_batch.sum() * nheads)


@pytest.mark.skip
@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("seqlen", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("nheads", [1, 4, 16, 24])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dropout_p", [0.1, 0.2])
@pytest.mark.parametrize("return_log", [True, False])
def test_flashattn2_qkvpacked_attn(
    bs, seqlen, nheads, d, causal, dropout_p, return_log
):
    print()
    seed_everything(0)
    qkv = torch.randn(
        (bs, seqlen, 3, nheads, d), device=device, dtype=dtype, requires_grad=True
    )
    # ctiga impl
    out, lse, score_cummax, S_dmask = flash_attn_qkvpacked_func(
        qkv, dropout_p, causal=causal, return_attn_probs=True
    )
    if dropout_p > 0.0:
        with torch.no_grad():
            attn, dropout_mask = reconstruct_attention_probs(
                S_dmask,
                lse,
                score_cummax,
                d,
                seqlen,
                seqlen,
                None,
                None,
                dropout_p > 0,
                causal,
                return_log=return_log,
            )
            if return_log:
                attn = torch.exp(attn)
        dropout_fraction = get_dropout_fraction(
            dropout_mask, None, None, causal=causal
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_ref(
        *qkv.unbind(dim=2), None, None, dropout_p, dropout_mask, causal=causal
    )
    out_pt, attn_pt = attention_ref(
        *qkv.unbind(dim=2),
        None,
        None,
        dropout_p,
        dropout_mask,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    # do_o = (g.float() * out.float()).sum(-1)
    # dv_tmp = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, :64], g[:, :64])
    # dv_tmp1 = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, 64:], g[:, 64:])
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        (dqkv,) = torch.autograd.grad(out, qkv, g)
        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
        (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
        print(f"dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
        print(
            f"dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}"
        )
        print(
            f"dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}"
        )
        print(
            f"dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}"
        )
        print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (
            attn_pt - attn_ref
        ).abs().max().item(), f"ref:\n{attn_ref}val:\n{attn}"
        assert abs(dropout_fraction - dropout_p) <= 0.015 * 2

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (
            dqkv_pt - dqkv_ref
        ).abs().max().item()


@pytest.mark.skip
@pytest.mark.parametrize("bs", [4, 8])
@pytest.mark.parametrize("seqlen", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("nheads", [1, 4, 16, 24])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dropout_p", [0.1, 0.2])
@pytest.mark.parametrize("return_log", [True, False])
def test_flashattn2_varlen_qkvpacked_attn(
    bs, seqlen, nheads, d, causal, dropout_p, return_log
):
    seed_everything(0)

    print()
    seed_everything(0)
    qkv = torch.randn(
        (bs, seqlen, 3, nheads, d), device=device, dtype=dtype, requires_grad=True
    )
    key_padding_mask = torch.stack(
        [
            torch.arange(seqlen, device=device) < tmp_seqlen
            for tmp_seqlen in random_seqlen(bs, seqlen)
        ],
        dim=0,
    ).to(device)
    q, k, v = qkv.unbind(dim=2)
    # unpad qkv
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, key_padding_mask)
    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
    v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v, key_padding_mask)
    qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
    # ctiga impl
    out_unpad, lse, score_cummax, S_dmask = flash_attn_varlen_qkvpacked_func(
        qkv_unpad,
        cu_seqlens_q,
        max_seqlen_q,
        dropout_p,
        causal=causal,
        return_attn_probs=True,
    )
    out = pad_input(out_unpad, indices_q, bs, seqlen)
    if dropout_p > 0.0:
        attn, dropout_mask = reconstruct_attention_probs(
            S_dmask,
            lse,
            score_cummax,
            d,
            seqlen,
            seqlen,
            key_padding_mask,
            key_padding_mask,
            dropout_p > 0,
            causal,
            return_log=return_log,
        )
        if return_log:
            attn = torch.exp(attn)
        dropout_fraction = get_dropout_fraction(
            dropout_mask, key_padding_mask, key_padding_mask, causal=causal
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        key_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        causal=causal,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        key_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        (dqkv_unpad,) = torch.autograd.grad(out, qkv_unpad, g)
        dqkv = pad_input(dqkv_unpad, indices_q, bs, seqlen)
        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
        (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
        print(f"dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
        print(
            f"dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}"
        )
        print(
            f"dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}"
        )
        print(
            f"dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}"
        )
        print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (
            attn_pt - attn_ref
        ).abs().max().item()
        assert abs(dropout_fraction - dropout_p) <= 0.015 * 2

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (
            dqkv_pt - dqkv_ref
        ).abs().max().item()


@pytest.mark.skip
@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("seqlen_q", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("seqlen_k", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("nheads", [1, 4, 16, 24])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dropout_p", [0.1, 0.2])
@pytest.mark.parametrize("return_log", [True, False])
def test_flashattn2_kvpacked_attn(
    bs, seqlen_q, seqlen_k, nheads, d, causal, dropout_p, return_log
):
    print()
    seed_everything(0)

    q = torch.randn(
        bs, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    kv = torch.randn(
        bs, seqlen_k, 2, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    out, lse, score_cummax, S_dmask = flash_attn_kvpacked_func(
        q, kv, dropout_p, causal=causal, return_attn_probs=True
    )
    if dropout_p > 0.0:
        with torch.no_grad():
            attn, dropout_mask = reconstruct_attention_probs(
                S_dmask,
                lse,
                score_cummax,
                d,
                seqlen_q,
                seqlen_k,
                None,
                None,
                dropout_p > 0,
                causal,
                return_log=return_log,
            )
            dropout_fraction = get_dropout_fraction(
                dropout_mask, None, None, causal=causal
            ).item()
        if return_log:
            attn = torch.exp(attn)
            dropout_fraction = get_dropout_fraction(
                dropout_mask, None, None, causal=causal
            ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_ref(
        q, kv[:, :, 0], kv[:, :, 1], None, None, dropout_p, dropout_mask, causal=causal
    )
    out_pt, attn_pt = attention_ref(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        None,
        None,
        dropout_p,
        dropout_mask,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    # do_o = (g.float() * out.float()).sum(-1)
    # dv_tmp = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, :64], g[:, :64])
    # dv_tmp1 = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, 64:], g[:, 64:])
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        (dq, dkv) = torch.autograd.grad(out, (q, kv), g)
        dk, dv = dkv.unbind(2)
        (dq_ref, dkv_ref) = torch.autograd.grad(out_ref, (q, kv), g)
        dk_ref, dv_ref = dkv_ref.unbind(2)
        (dq_pt, dkv_pt) = torch.autograd.grad(out_pt, (q, kv), g)
        dk_pt, dv_pt = dkv_pt.unbind(2)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (
            attn_pt - attn_ref
        ).abs().max().item(), f"ref:\n{attn_ref}val:\n{attn}"
        assert abs(dropout_fraction - dropout_p) <= 0.015 * 2

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 2 * (
            dq_pt - dq_ref
        ).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 2 * (
            dk_pt - dk_ref
        ).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 2 * (
            dv_pt - dv_ref
        ).abs().max().item()


@pytest.mark.skip
@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("seqlen_q", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("seqlen_k", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("nheads", [1, 4, 16, 24])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dropout_p", [0.1, 0.2])
@pytest.mark.parametrize("return_log", [True, False])
def test_flashattn2_varlen_kvpacked_attn(
    bs, seqlen_q, seqlen_k, nheads, d, causal, dropout_p, return_log
):
    print()
    seed_everything(0)

    q = torch.randn(
        bs, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    kv = torch.randn(
        bs, seqlen_k, 2, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    query_padding_mask = torch.stack(
        [
            torch.arange(seqlen_q, device=device) < tmp_seqlen
            for tmp_seqlen in random_seqlen(bs, seqlen_q)
        ],
        dim=0,
    ).to(device)
    key_padding_mask = torch.stack(
        [
            torch.arange(seqlen_k, device=device) < tmp_seqlen
            for tmp_seqlen in random_seqlen(bs, seqlen_k)
        ],
        dim=0,
    ).to(device)

    k, v = kv.unbind(dim=2)
    # unpad qkv
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
    v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v, key_padding_mask)

    kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)

    out_unpad, lse, score_cummax, S_dmask = flash_attn_varlen_kvpacked_func(
        q_unpad,
        kv_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        return_attn_probs=True,
        causal=causal,
    )
    out = pad_input(out_unpad, indices_q, bs, max_seqlen_q)
    if dropout_p > 0.0:
        with torch.no_grad():
            attn, dropout_mask = reconstruct_attention_probs(
                S_dmask,
                lse,
                score_cummax,
                d,
                seqlen_q,
                seqlen_k,
                query_padding_mask,
                key_padding_mask,
                dropout_p > 0,
                causal,
                return_log=return_log,
            )
        if return_log:
            attn = torch.exp(attn)
        dropout_fraction = get_dropout_fraction(
            dropout_mask, query_padding_mask, key_padding_mask, causal=causal
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        causal=causal,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    g = torch.randn_like(out)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        (dq_unpad, dkv_unpad) = torch.autograd.grad(out, (q_unpad, kv_unpad), g)
        dk, dv = pad_input(dkv_unpad, indices_k, bs, max_seqlen_k).unbind(2)
        (dq_ref, dkv_ref) = torch.autograd.grad(out_ref, (q, kv), g)
        dk_ref, dv_ref = dkv_ref.unbind(2)
        (dq_pt, dkv_pt) = torch.autograd.grad(out_pt, (q, kv), g)
        dk_pt, dv_pt = dkv_pt.unbind(2)

        dq = pad_input(dq_unpad, indices_q, bs, max_seqlen_q)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (
            attn_pt - attn_ref
        ).abs().max().item()
        assert abs(dropout_fraction - dropout_p) <= 0.015 * 2

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 2 * (
            dq_pt - dq_ref
        ).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 2 * (
            dk_pt - dk_ref
        ).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 2 * (
            dv_pt - dv_ref
        ).abs().max().item()
