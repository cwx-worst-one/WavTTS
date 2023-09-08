import math

import torch
import torch.nn.functional as F
from einops import rearrange

from samantha.components.ctiga.ops.flash_attn_2_interface import _get_block_size


def construct_causal_mask(
    seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, device=None
):
    # causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=device), 1) # noqa
    # return causal_mask

    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )

    return col_idx > row_idx + sk - sq


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """

    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(S.device, head_dim, is_dropout, causal)
    # nblocks_n = (seqlen_k_rounded + blocksize_n - 1) // blocksize_n
    # nblocks_m = (seqlen_q_rounded + blocksize_m - 1) // blocksize_m
    # print(blocksize_m, nblocks_n)
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(
        S,
        "b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)",  # noqa
        blocksize_m=blocksize_m,
        blocksize_n=blocksize_n,
        # nblocks_n=nblocks_n,
        # nblocks_m=nblocks_m,
    )
    S_converted = rearrange(
        S_flat,
        "b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)",  # noqa
        mmas_n=mmas_n,
        warps_n=warps_n,
        eight=8,
        c0=2,
        c1=2,
        c2=2,
        four=4,
        # nblocks_n=nblocks_n,
        # nblocks_m=nblocks_m,
    )

    if causal:
        causal_mask = construct_causal_mask(
            seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, S.device
        )
        causal_mask = F.pad(
            causal_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted.masked_fill_(causal_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with
    # random values and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1]
        if query_padding_mask is not None
        else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(
            query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og)
        )
        S_converted = S_converted.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    seqlen_k_og = (
        key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    )
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0
        )
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S_torch(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    is_dropout=False,
    causal=False,
):
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    _, block_size_n = _get_block_size(scores.device, head_dim, is_dropout, causal)
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
    _, block_size_n = _get_block_size(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those
    # to inf so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack(
        [torch.amax(s, dim=-1) for s in scores_block], dim=-1
    )
    cummax_block = (
        torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    )
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )

    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.type_as(attn_unnorm)


def normalize_flash_attn_S(
    attn_unnorm,  # b,n,tq,tk
    lse,  # b,tq
    score_cummax,  # n_blocks,b,n,tq
    head_dim,
    query_padding_mask=None,  # b,tq
    key_padding_mask=None,  # b,tk
    is_dropout=False,
    causal=False,
    return_log=False,
):
    attn_unnorm = attn_unnorm.abs()
    if query_padding_mask is not None and key_padding_mask is not None:
        if query_padding_mask.shape[-1] != key_padding_mask.shape[-1]:
            assert not causal, "no support causal=True when seqlen_q!=seqlen_k now"
    _, block_size_n = _get_block_size(attn_unnorm.device, head_dim, is_dropout, causal)

    cummax_block = score_cummax.split(1, dim=0)

    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    if not return_log:
        attn_norm = torch.cat(
            [
                a * rearrange(torch.exp(m[0] - lse), "b h s -> b h s 1")
                for a, m in zip(attn_unnorm_block, cummax_block)
            ],
            dim=-1,
        )
    else:
        attn_norm = torch.cat(
            [
                torch.log(a) + rearrange(m[0] - lse, "b h s -> b h s 1")
                for a, m in zip(attn_unnorm_block, cummax_block)
            ],
            dim=-1,
        )

    if query_padding_mask is not None:
        attn_norm.masked_fill_(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"),
            float("-inf") if return_log else 0.0,
        )

    return attn_norm.type_as(attn_unnorm)


@torch.no_grad()
def reconstruct_attention_probs(
    dmask,
    lse,
    score_cummax,
    head_dim,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    dropout_p,
    causal,
    return_log=False,
):
    S_dmask_converted = convert_flash_attn_S_to_softmax(
        dmask,
        seqlen_q,
        seqlen_k,
        query_padding_mask,
        key_padding_mask,
        head_dim,
        dropout_p > 0.0,
        causal,
    )

    dropout_mask = S_dmask_converted >= 0
    attn_unnorm = S_dmask_converted.abs()

    attn_norm = normalize_flash_attn_S(
        attn_unnorm,
        lse,
        score_cummax,
        head_dim,
        query_padding_mask,
        key_padding_mask,
        dropout_p > 0.0,
        causal,
        return_log=return_log,
    )

    return attn_norm, dropout_mask
