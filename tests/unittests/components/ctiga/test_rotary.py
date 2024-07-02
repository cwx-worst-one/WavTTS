import random

import pytest
import pytorch_lightning as pl
import torch

from samantha.utils.ctiga.padding import pad_input, unpad_input

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = "cuda:0"


@pytest.mark.skip()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 5120])
@pytest.mark.parametrize("nhead", [32, 24, 12, 8, 6])
@pytest.mark.parametrize("bs", [3, 6, 8, 9])
@pytest.mark.parametrize("same_offset", [True, False])
@pytest.mark.parametrize("compat", ["default", "byteformer"])
def test_rotary_qkvpacked(bs, nhead, dim, seqlen, compat, dtype, same_offset):
    print()
    from samantha.components.ctiga.rotary import RotaryEmbedding

    rotary_emb = RotaryEmbedding(
        dim=dim, interleaved=True, device=device, compat=compat, use_triton=False
    )

    pl.seed_everything(0)

    qkv = torch.rand(
        bs, seqlen, 3, nhead, dim, device=device, dtype=dtype, requires_grad=True
    )

    seqlens = torch.randint(
        max(seqlen - 50, 256), seqlen, (bs,), device=device, dtype=torch.long
    )
    padding_mask = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(
        0
    ).repeat(bs, 1) < seqlens.unsqueeze(1)

    if same_offset:
        seqlen_offset = random.randint(0, 64)
    else:
        seqlen_offset = torch.randint(0, 64, (bs,), device=device, dtype=torch.long)
    print(f"{(bs, nhead, dim, seqlen, compat, dtype, seqlen_offset)=}")
    """ fwd """
    # ref
    qkv_ref = qkv.clone()
    out_ref = rotary_emb.forward(qkv_ref, seqlen_offset=seqlen_offset)
    out_ref = torch.masked_fill(out_ref, ~padding_mask[..., None, None, None], 0)

    # unpad
    unpad_qkv, indices, cu_seqlens, max_seqlen = unpad_input(qkv.clone(), padding_mask)
    unpad_out = rotary_emb.forward(
        unpad_qkv,
        seqlen_offset=seqlen_offset,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    out = pad_input(unpad_out, indices, bs, seqlen)

    print(f"{(out-out_ref).abs().mean().item()=}")
    print(f"{(out-out_ref).abs().max().item()=}")
    torch.testing.assert_allclose(out_ref, out)

    """ bwd """
    g = torch.randn_like(out)
    # ref
    (dqkv_ref,) = torch.autograd.grad(out_ref, qkv_ref, g.clone())
    # unpad
    (dqkv_unpad,) = torch.autograd.grad(out, unpad_qkv, g.clone())
    dqkv = pad_input(dqkv_unpad, indices, bs, seqlen)

    print(f"{(dqkv-dqkv_ref).abs().mean().item()=}")
    print(f"{(dqkv-dqkv_ref).abs().max().item()=}")
    torch.testing.assert_allclose(dqkv_ref, dqkv)


@pytest.mark.skip()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(384, 512), (1024, 1024)])
@pytest.mark.parametrize("nhead_q,nhead_k", [(32, 8), (16, 1), (8, 8)])
@pytest.mark.parametrize("bs", [3, 6, 8, 9])
@pytest.mark.parametrize("same_offset", [True, False])
@pytest.mark.parametrize("compat", ["default", "byteformer"])
@pytest.mark.parametrize("crossattn", [False, True])
def test_rotary_kvpacked(
    bs, nhead_q, nhead_k, dim, seqlen_q, seqlen_k, compat, dtype, same_offset, crossattn
):
    print()
    from samantha.components.ctiga.rotary import RotaryEmbedding

    if (not crossattn) and (seqlen_q != seqlen_k):
        pytest.skip()
    rotary_emb = RotaryEmbedding(
        dim=dim, interleaved=True, device=device, compat=compat, use_triton=False
    )

    pl.seed_everything(0)
    q = torch.rand(
        bs, seqlen_q, nhead_q, dim, device=device, dtype=dtype, requires_grad=True
    )
    kv = torch.rand(
        bs, seqlen_k, 2, nhead_k, dim, device=device, dtype=dtype, requires_grad=True
    )

    seqlens_q = torch.randint(
        max(seqlen_q - 50, 256), seqlen_q, (bs,), device=device, dtype=torch.long
    )
    query_padding_mask = torch.arange(
        seqlen_q, device=device, dtype=torch.long
    ).unsqueeze(0).repeat(bs, 1) < seqlens_q.unsqueeze(1)
    if same_offset:
        seqlen_offset_q = random.randint(0, 64)
    else:
        seqlen_offset_q = torch.randint(0, 64, (bs,), device=device, dtype=torch.long)

    if crossattn:
        seqlens_k = torch.randint(
            max(seqlen_k - 50, 256), seqlen_k, (bs,), device=device, dtype=torch.long
        )
        key_padding_mask = torch.arange(
            seqlen_k, device=device, dtype=torch.long
        ).unsqueeze(0).repeat(bs, 1) < seqlens_k.unsqueeze(1)
        if same_offset:
            seqlen_offset_k = random.randint(0, 64)
        else:
            seqlen_offset_k = torch.randint(
                0, 64, (bs,), device=device, dtype=torch.long
            )
    else:
        key_padding_mask = query_padding_mask
        seqlen_offset_k = seqlen_offset_q

    print(
        f"{(bs, nhead_q,nhead_k, dim, seqlen_q,seqlen_k, compat, dtype, seqlen_offset_q, seqlen_offset_k, crossattn)=}"
    )
    """ fwd """
    # ref
    q_ref, kv_ref = q.clone(), kv.clone()
    out_q_ref, out_kv_ref = rotary_emb.forward(
        q_ref, kv_ref, seqlen_offset=seqlen_offset_q, seqlen_offset_k=seqlen_offset_k
    )
    out_q_ref = torch.masked_fill(out_q_ref, ~query_padding_mask[..., None, None], 0)
    out_kv_ref = torch.masked_fill(
        out_kv_ref, ~key_padding_mask[..., None, None, None], 0
    )

    # unpad
    unpad_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        q.clone(), query_padding_mask
    )
    unpad_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
        kv.clone(), key_padding_mask
    )
    unpad_q_out, unpad_kv_out = rotary_emb.forward(
        unpad_q,
        unpad_kv,
        seqlen_offset=seqlen_offset_q,
        seqlen_offset_k=seqlen_offset_k,
        cu_seqlens=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    out_q = pad_input(unpad_q_out, indices_q, bs, seqlen_q)
    out_kv = pad_input(unpad_kv_out, indices_k, bs, seqlen_k)

    print(f"{(out_q-out_q_ref).abs().mean().item()=}")
    print(f"{(out_q-out_q_ref).abs().max().item()=}")
    torch.testing.assert_allclose(out_q, out_q_ref)
    print(f"{(out_kv-out_kv_ref).abs().mean().item()=}")
    print(f"{(out_kv-out_kv_ref).abs().max().item()=}")
    torch.testing.assert_allclose(out_kv, out_kv_ref)

    """ bwd """
    g_q = torch.randn_like(out_q)
    g_kv = torch.randn_like(out_kv)
    # ref
    (dq_ref, dkv_ref) = torch.autograd.grad(
        (out_q_ref, out_kv_ref), (q_ref, kv_ref), (g_q.clone(), g_kv.clone())
    )
    # unpad
    (dq_unpad, dkv_unpad) = torch.autograd.grad(
        (out_q, out_kv), (unpad_q, unpad_kv), (g_q.clone(), g_kv.clone())
    )
    dq = pad_input(dq_unpad, indices_q, bs, seqlen_q)
    dkv = pad_input(dkv_unpad, indices_k, bs, seqlen_k)

    print(f"{(dq-dq_ref).abs().mean().item()=}")
    print(f"{(dq-dq_ref).abs().max().item()=}")
    torch.testing.assert_allclose(dq, dq_ref)

    print(f"{(dkv-dkv_ref).abs().mean().item()=}")
    print(f"{(dkv-dkv_ref).abs().max().item()=}")
    torch.testing.assert_allclose(dkv, dkv_ref)


@pytest.mark.skip()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 5120])
@pytest.mark.parametrize("nhead", [32, 24, 12, 8, 7])
@pytest.mark.parametrize("bs", [3, 6, 8, 9])
@pytest.mark.parametrize("compat", ["default", "byteformer"])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("same_offset", [True, False])
def test_rotary_qkvpacked_triton(
    bs, nhead, dim, seqlen, compat, dtype, varlen, same_offset
):
    print()
    from samantha.components.ctiga.rotary import RotaryEmbedding

    rotary_emb_ref = RotaryEmbedding(
        dim=dim, interleaved=True, device=device, compat=compat, use_triton=False
    )
    rotary_emb_triton = RotaryEmbedding(
        dim=dim, interleaved=True, device=device, compat=compat, use_triton=True
    )

    pl.seed_everything(0)

    qkv = torch.rand(
        bs, seqlen, 3, nhead, dim, device=device, dtype=dtype, requires_grad=True
    )

    if same_offset:
        seqlen_offset = random.randint(0, 64)
    else:
        seqlen_offset = torch.randint(0, 64, (bs,), device=device, dtype=torch.long)
    print(f"{(bs, nhead, dim, seqlen, compat, dtype, varlen, seqlen_offset)=}")
    if varlen:
        seqlens = torch.randint(
            max(seqlen - 50, 256), seqlen, (bs,), device=device, dtype=torch.long
        )
        padding_mask = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(
            0
        ).repeat(bs, 1) < seqlens.unsqueeze(1)

        qkv_ref = qkv.clone()
        qkv_triton = qkv.clone()

        """ cuda fwd"""
        out_ref = rotary_emb_ref.forward(qkv_ref, seqlen_offset=seqlen_offset)
        out_ref = torch.masked_fill(out_ref, ~padding_mask[..., None, None, None], 0)

        """ triton varlen fwd"""
        unpad_qkv_triton, indices, cu_seqlens, max_seqlen = unpad_input(
            qkv_triton, padding_mask
        )
        unpad_out_triton = rotary_emb_triton.forward(
            unpad_qkv_triton,
            seqlen_offset=seqlen_offset,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        out_triton = pad_input(unpad_out_triton, indices, bs, seqlen)

        g_ref = torch.rand_like(out_triton)
        # g_triton = g_ref.clone()

        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv_ref, g_ref)
        (dqkv_unpad_triton,) = torch.autograd.grad(
            out_triton, unpad_qkv_triton, g_ref.clone()
        )
        dqkv_triton = pad_input(dqkv_unpad_triton, indices, bs, seqlen)
    else:
        """cuda fwd"""
        qkv_ref = qkv.clone()
        out_ref = rotary_emb_ref.forward(qkv_ref, seqlen_offset=seqlen_offset)
        """ triton fwd"""
        qkv_triton = qkv.clone()
        out_triton = rotary_emb_triton.forward(qkv_triton, seqlen_offset=seqlen_offset)

        g_ref = torch.rand_like(out_triton)
        g_triton = g_ref.clone()

        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv_ref, g_ref)
        (dqkv_triton,) = torch.autograd.grad(out_triton, qkv_triton, g_triton)
    print(f"{(out_triton-out_ref).abs().mean().item()=}")
    print(f"{(out_triton-out_ref).abs().max().item()=}")
    torch.testing.assert_allclose(out_ref, out_triton)
    print(f"{(dqkv_triton-dqkv_ref).abs().mean().item()=}")
    print(f"{(dqkv_triton-dqkv_ref).abs().max().item()=}")
    torch.testing.assert_allclose(dqkv_ref, dqkv_triton)


@pytest.mark.skip()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(384, 512), (512, 512)])
@pytest.mark.parametrize("nhead_q,nhead_k", [(32, 8), (12, 1), (8, 8)])
@pytest.mark.parametrize("bs", [3, 6, 8, 9])
@pytest.mark.parametrize("compat", ["default", "byteformer"])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("same_offset", [True, False])
@pytest.mark.parametrize("crossattn", [True, False])
def test_rotary_kvpacked_triton(
    bs,
    nhead_q,
    nhead_k,
    dim,
    seqlen_q,
    seqlen_k,
    compat,
    dtype,
    varlen,
    same_offset,
    crossattn,
):
    print()
    from samantha.components.ctiga.rotary import RotaryEmbedding

    if (not crossattn) and (seqlen_q != seqlen_k):
        pytest.skip()
    rotary_emb_ref = RotaryEmbedding(
        dim=dim, interleaved=True, device=device, compat=compat, use_triton=False
    )
    rotary_emb_triton = RotaryEmbedding(
        dim=dim, interleaved=True, device=device, compat=compat, use_triton=True
    )

    pl.seed_everything(0)
    q = torch.rand(
        bs, seqlen_q, nhead_q, dim, device=device, dtype=dtype, requires_grad=True
    )
    kv = torch.rand(
        bs, seqlen_k, 2, nhead_k, dim, device=device, dtype=dtype, requires_grad=True
    )

    if same_offset:
        seqlen_offset_q = random.randint(0, 64)
        seqlen_offset_k = random.randint(0, 64) if crossattn else seqlen_offset_q
    else:
        seqlen_offset_q = torch.randint(0, 64, (bs,), device=device, dtype=torch.long)
        seqlen_offset_k = (
            torch.randint(0, 64, (bs,), device=device, dtype=torch.long)
            if crossattn
            else seqlen_offset_q
        )

    print(
        f"{(bs, nhead_q, nhead_k, dim, seqlen_q, seqlen_k, compat, dtype, varlen, seqlen_offset_q, seqlen_offset_k, crossattn)=}"
    )
    if varlen:
        seqlens_q = torch.randint(
            max(seqlen_q - 50, 256), seqlen_q, (bs,), device=device, dtype=torch.long
        )
        query_padding_mask = torch.arange(
            seqlen_q, device=device, dtype=torch.long
        ).unsqueeze(0).repeat(bs, 1) < seqlens_q.unsqueeze(1)
        q_ref = q.clone()
        q_triton = q.clone()

        if crossattn:
            seqlens_k = torch.randint(
                max(seqlen_k - 50, 256),
                seqlen_k,
                (bs,),
                device=device,
                dtype=torch.long,
            )
            key_padding_mask = torch.arange(
                seqlen_k, device=device, dtype=torch.long
            ).unsqueeze(0).repeat(bs, 1) < seqlens_k.unsqueeze(1)
        else:
            seqlens_k = seqlen_q
            key_padding_mask = query_padding_mask
        kv_ref = kv.clone()
        kv_triton = kv.clone()

        """ cuda fwd"""
        out_q_ref, out_kv_ref = rotary_emb_ref.forward(
            q_ref,
            kv_ref,
            seqlen_offset=seqlen_offset_q,
            seqlen_offset_k=seqlen_offset_k,
        )
        out_q_ref = torch.masked_fill(
            out_q_ref, ~query_padding_mask[..., None, None], 0
        )
        out_kv_ref = torch.masked_fill(
            out_kv_ref, ~key_padding_mask[..., None, None, None], 0
        )

        """ triton varlen fwd"""
        unpad_q_triton, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            q_triton, query_padding_mask
        )
        unpad_kv_triton, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
            kv_triton, key_padding_mask
        )
        unpad_out_q_triton, unpad_out_kv_triton = rotary_emb_triton.forward(
            unpad_q_triton,
            unpad_kv_triton,
            seqlen_offset=seqlen_offset_q,
            seqlen_offset_k=seqlen_offset_k,
            cu_seqlens=cu_seqlens_q,
            max_seqlen=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )
        out_q_triton = pad_input(unpad_out_q_triton, indices_q, bs, seqlen_q)
        out_kv_triton = pad_input(unpad_out_kv_triton, indices_k, bs, seqlen_k)

        g_q_ref = torch.rand_like(out_q_triton)
        g_kv_ref = torch.rand_like(out_kv_triton)

        (dq_ref, dkv_ref) = torch.autograd.grad(
            (out_q_ref, out_kv_ref), (q_ref, kv_ref), (g_q_ref, g_kv_ref)
        )
        (dq_unpad_triton, dkv_unpad_triton) = torch.autograd.grad(
            (out_q_triton, out_kv_triton),
            (unpad_q_triton, unpad_kv_triton),
            (g_q_ref.clone(), g_kv_ref.clone()),
        )
        dq_triton = pad_input(dq_unpad_triton, indices_q, bs, seqlen_q)
        dkv_triton = pad_input(dkv_unpad_triton, indices_k, bs, seqlen_k)

    else:
        """cuda fwd"""
        q_ref = q.clone()
        kv_ref = kv.clone()
        out_q_ref, out_kv_ref = rotary_emb_ref.forward(
            q_ref,
            kv_ref,
            seqlen_offset=seqlen_offset_q,
            seqlen_offset_k=seqlen_offset_k,
        )
        """ triton fwd"""
        q_triton = q.clone()
        kv_triton = kv.clone()
        out_q_triton, out_kv_triton = rotary_emb_triton.forward(
            q_triton,
            kv_triton,
            seqlen_offset=seqlen_offset_q,
            seqlen_offset_k=seqlen_offset_k,
        )

        g_q_ref = torch.rand_like(out_q_triton)
        g_kv_ref = torch.rand_like(out_kv_triton)

        (dq_ref, dkv_ref) = torch.autograd.grad(
            (out_q_ref, out_kv_ref),
            (q_ref, kv_ref),
            (g_q_ref.clone(), g_kv_ref.clone()),
        )
        (dq_triton, dkv_triton) = torch.autograd.grad(
            (out_q_triton, out_kv_triton),
            (q_triton, kv_triton),
            (g_q_ref.clone(), g_kv_ref.clone()),
        )

    print(f"{(out_q_triton-out_q_ref).abs().mean().item()=}")
    print(f"{(out_q_triton-out_q_ref).abs().max().item()=}")
    torch.testing.assert_allclose(out_q_ref, out_q_triton)
    print(f"{(out_kv_triton-out_kv_ref).abs().mean().item()=}")
    print(f"{(out_kv_triton-out_kv_ref).abs().max().item()=}")
    torch.testing.assert_allclose(out_kv_ref, out_kv_triton)

    print(f"{(dq_triton-dq_ref).abs().mean().item()=}")
    print(f"{(dq_triton-dq_ref).abs().max().item()=}")
    torch.testing.assert_allclose(dq_ref, dq_triton)
    print(f"{(dkv_triton-dkv_ref).abs().mean().item()=}")
    print(f"{(dkv_triton-dkv_ref).abs().max().item()=}")
    torch.testing.assert_allclose(dkv_ref, dkv_triton)


@pytest.mark.skip()
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("dim", [32, 64, 128, 80, 224])
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 986])
def test_llama_mha_varlen_training(mha_type, dtype, use_triton, dim, seqlen):
    pl.seed_everything(0)
    bs = 8
    # seqlen = 512
    nhead_q = 32
    nhead_k = nhead_q if mha_type == "mha" else (4 if mha_type == "gqa" else 1)
    seqlens = torch.randint(seqlen // 2, seqlen, (bs,), device=device, dtype=torch.long)
    attention_mask = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(
        0
    ).repeat(bs, 1) < seqlens.unsqueeze(1)
    print(f"\n{(mha_type, dtype, use_triton, seqlens)=}")
    from samantha.components.ctiga.mha import MHA

    if mha_type == "mha":
        mha = MHA(
            nhead_q * dim,
            nhead_q,
            None,
            qkv_proj_bias=False,
            out_proj_bias=False,
            rotary_emb_dim=dim,
            use_rotary_triton=use_triton,
            use_flash_attn=True,
            device=device,
            dtype=dtype,
        )
    else:
        mha = MHA(
            nhead_q * dim,
            nhead_q,
            nhead_k,
            qkv_proj_bias=False,
            out_proj_bias=False,
            rotary_emb_dim=dim,
            use_rotary_triton=use_triton,
            use_flash_attn=True,
            device=device,
            dtype=dtype,
        )
    for name, module in mha.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)

    h = torch.rand(
        bs, seqlen, nhead_q * dim, device=device, dtype=dtype, requires_grad=True
    )
    unpad_h, indices, cu_seqlens, max_seqlen = unpad_input(h, attention_mask)

    # ref
    print("Process ref")
    unpad_h_ref = unpad_h.clone()
    unpad_out_ref = mha(
        unpad_h_ref,
        cu_seqlens=cu_seqlens,
        max_seqlen=seqlen,
        key_padding_mask=attention_mask,
        indices=indices,
    )[0]
    # varlen
    print("Process varlen")
    unpad_h_act = unpad_h.clone()
    unpad_out_act = mha(unpad_h_act, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)[0]

    print(f"{(unpad_out_ref-unpad_out_act).abs().diff().mean().item()=}")
    print(f"{(unpad_out_ref-unpad_out_act).abs().diff().max().item()=}")
    if dtype == torch.bfloat16:
        torch.testing.assert_allclose(
            unpad_out_act, unpad_out_ref, atol=0.016 * 2, rtol=1e-5 * 2
        )
    else:
        torch.testing.assert_allclose(unpad_out_act, unpad_out_ref)

    g = torch.rand_like(unpad_out_act)

    g_ref = g.clone()
    (d_h_ref,) = torch.autograd.grad((unpad_out_ref,), (unpad_h_ref), (g_ref,))
    (d_h_act,) = torch.autograd.grad((unpad_out_act,), (unpad_h_act), (g,))

    print(f"{(d_h_ref-d_h_act).abs().diff().mean().item()=}")
    print(f"{(d_h_ref-d_h_act).abs().diff().max().item()=}")
    if dtype == torch.bfloat16:
        torch.testing.assert_allclose(d_h_act, d_h_ref, atol=0.016 * 2, rtol=1e-5 * 2)
    else:
        torch.testing.assert_allclose(d_h_act, d_h_ref)
