from typing import Any, Tuple

import torch
import torch.nn.functional as F
from einops import einsum, rearrange


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    mask: Any = None,
    attn_bias: Any = None,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score = einsum(
        q, k, "b n_heads seq_i d_k, b n_heads seq_j d_k -> b n_heads seq_i seq_j"
    )
    score *= scale

    if attn_bias is not None:
        score = score + attn_bias

    if mask is not None:
        mask = rearrange(mask, "b j -> b 1 1 j")
        score = score.masked_fill(~mask, -torch.finfo(score.dtype).max)

    if is_causal:
        i, j = score.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=score.device).triu(
            j - i + 1
        )
        score = score.masked_fill(causal_mask, -torch.finfo(score.dtype).max)

    attention = score.float().softmax(dim=-1).type_as(q)

    attention = F.dropout(attention, p=dropout_p)

    score = einsum(
        attention,
        v,
        "b n_heads seq_i seq_j, b n_heads seq_j d_k -> b n_heads seq_i d_k",
    )
    return score, attention
