import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from ... import _is_triton_available
from ...triton.softmax import softmax as triton_softmax
from ..positional_embedding.rotary import RotaryEmbedding, SeerEmbedding


def _softmax(x: torch.Tensor, causal: bool = False) -> torch.Tensor:
    if _is_triton_available():
        return triton_softmax(x, mask=None, causal=causal)
    else:
        return x.softmax(dim=-1)


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

    attention = _softmax(score)
    attention = F.dropout(attention, p=dropout_p)

    score = einsum(
        attention,
        v,
        "b n_heads seq_i seq_j, b n_heads seq_j d_k -> b n_heads seq_i d_k",
    )
    return score, attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        scale: Optional[float] = None,
        causal: bool = False,
        use_rotary_embeddings: bool = False,
        d_k: Optional[int] = None,
        enable_flash: bool = True,
        enable_mem_efficient: bool = True,
    ) -> None:
        super().__init__()
        self._use_cache = False

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k

        if not self.d_k:
            self.d_k = d_model // n_heads

        self.dropout_p = dropout
        self.causal = causal
        self.enable_flash = enable_flash
        self.enable_mem_efficient = enable_mem_efficient

        if (enable_flash or enable_mem_efficient) and not hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            raise ImportError("Flash Attention requires PyTorch >= 2.0")

        self.rotary_embeddings = (
            RotaryEmbedding(self.d_k) if use_rotary_embeddings else None
        )

        self.qkv_dim_with_heads = self.d_k * self.n_heads

        self.to_q = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
        self.to_k = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
        self.to_v = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)

        self.scale = scale
        if scale is None:
            self.scale = self.d_k**-0.5

        self.WO = nn.Linear(self.qkv_dim_with_heads, d_model, bias=bias)

    def rearrange_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x, "b s (n_heads d_k) -> b n_heads s d_k", n_heads=self.n_heads
        )

    def qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """Projects the q, k and v matrices and rearranges them
        according to the following convention:

        x and (optionally) the context: (batch_size, seq_len d_model)
        And are rearranged as:          (batch_size, n_heads, seq_len, d_k)

        The k/v cache mechanism uses a forward hook to append outputs to the
        kv_cache dictionary. The lookup is the hash of the `self.to_k` and
        `self.to_v` variables (i.e., the reference to the `nn.Linear` class
        respectively).

        Args:
            x (torch.Tensor): _description_
            context (Optional[torch.Tensor], optional): Defaults to None.
            kv_cache (Optional[dict], optional): Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        if context is not None:
            k, v = context, context
        else:
            k, v = x, x

        q = self.to_q(x)
        k = self.to_k(k)
        v = self.to_v(v)

        q = self.rearrange_heads(q)
        k = self.rearrange_heads(k)
        v = self.rearrange_heads(v)

        if kv_cache is not None:
            if not self._use_cache:
                raise RuntimeError("`_use_cache` needs to be enabled first")
            kv_cache[self.to_k] = (
                torch.cat((kv_cache[self.to_k], k), dim=2)
                if self.to_k in kv_cache
                else k
            )
            kv_cache[self.to_v] = (
                torch.cat((kv_cache[self.to_v], v), dim=2)
                if self.to_v in kv_cache
                else v
            )

            k = kv_cache[self.to_k]
            v = kv_cache[self.to_v]

        if self.rotary_embeddings:
            cache_len = k.shape[2] if self._use_cache else None
            q, k = self.rotary_embeddings(q, k, q_len=cache_len)
        return q, k, v

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_attention: bool
    ):
        """Computes the scaled dot-product of the q, k and v matrices.
        Here, we assume that each have the following shape:

        (batch_size, n_heads, seq_len, d_k)

        When performing cached inference (`self._use_cache = True`),
        we need to set an explicity attention mask (`attn_mask`), so that
        the scaled dot-product behaves consistently.

        Args:
            q (torch.Tensor): _description_
            k (torch.Tensor): _description_
            v (torch.Tensor): _description_
            return_attention (bool): _description_

        Returns:
            _type_: _description_
        """
        dropout_p = self.dropout_p if self.training else 0
        attention = None
        if self.enable_flash or self.enable_mem_efficient:
            if self.causal and self._use_cache:
                i = q.shape[2]
                j = k.shape[2]
                attn_mask = torch.ones(i, j, dtype=torch.bool, device=q.device).tril(
                    j - i
                )
                is_causal = False
            else:
                is_causal = self.causal
                attn_mask = None

            with torch.backends.cuda.sdp_kernel(
                enable_flash=self.enable_flash,
                enable_math=True,
                enable_mem_efficient=self.enable_mem_efficient,
            ):
                score = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        else:
            score, attention = scaled_dot_product(
                q, k, v, scale=self.scale, dropout_p=dropout_p, is_causal=self.causal
            )
        score = rearrange(score, "b n_heads s d_k -> b s (n_heads d_k)")
        if return_attention:
            return self.WO(score), attention
        return self.WO(score)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self.qkv(x, context, kv_cache=kv_cache)
        return self.attention(q, k, v, return_attention=return_attention)


class SeerAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_priors: int,
        n_seers: int,
        seq_len: int,
        dropout: float = 0.0,
        bias: bool = True,
        scale: Optional[float] = None,
        d_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._use_cache = False

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_priors = n_priors
        self.n_seers = n_seers
        self.seq_len = seq_len
        self.d_k = d_k

        if not self.d_k:
            self.d_k = d_model // n_heads

        self.dropout_p = dropout

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("Flash Attention requires PyTorch >= 2.0")

        self.rotary_embeddings = SeerEmbedding(
            self.d_k, self.n_priors, self.n_seers, self.seq_len
        )

        self.qkv_dim_with_heads = self.d_k * self.n_heads

        self.to_q = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
        self.to_k = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
        self.to_v = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)

        self.scale = scale
        if scale is None:
            self.scale = self.d_k**-0.5

        self.out = nn.Linear(self.qkv_dim_with_heads, d_model, bias=bias)

    def rearrange_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x, "b s (n_heads d_k) -> b n_heads s d_k", n_heads=self.n_heads
        )

    def qkv(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.rearrange_heads(q)
        k = self.rearrange_heads(k)
        v = self.rearrange_heads(v)

        if kv_cache is not None:
            if not self._use_cache:
                raise RuntimeError("`_use_cache` needs to be enabled first")
            kv_cache[self.to_k] = (
                torch.cat((kv_cache[self.to_k], k), dim=2)
                if self.to_k in kv_cache
                else k
            )
            kv_cache[self.to_v] = (
                torch.cat((kv_cache[self.to_v], v), dim=2)
                if self.to_v in kv_cache
                else v
            )
            k = kv_cache[self.to_k]
            v = kv_cache[self.to_v]

        cache_len = k.shape[2] if self._use_cache else None
        q, k = self.rotary_embeddings(q, k, q_len=cache_len)
        return q, k, v

    def seer_mask(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Assuming:
        # q_len <= k_len
        # (q_len - self.n_priors) % self.n_seers == 0
        # (k_len - self.n_priors) % self.n_seers == 0
        q_len = q.size(2)
        k_len = k.size(2)
        attn_mask = torch.zeros(q_len, k_len, dtype=torch.bool, device=q.device)
        attn_mask[:, : self.n_priors] = True
        prefix_q_len = self.n_priors if q_len == k_len else 0
        sub_q_len = math.ceil((q_len - prefix_q_len) / self.n_seers)
        sub_k_len = math.ceil((k_len - self.n_priors) / self.n_seers)
        sub_mask = torch.ones(
            sub_q_len, sub_k_len, dtype=torch.bool, device=q.device
        ).tril(sub_k_len - sub_q_len)
        sub_mask = sub_mask.repeat_interleave(self.n_seers, dim=0)
        sub_mask = sub_mask.repeat_interleave(self.n_seers, dim=1)
        attn_mask[prefix_q_len:, self.n_priors :] = sub_mask
        return attn_mask

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
    ):
        dropout_p = self.dropout_p if self.training else 0
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )

        attn_out = rearrange(attn_out, "b n_heads s d_k -> b s (n_heads d_k)")
        return self.out(attn_out)

    def forward(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
        q, k, v = self.qkv(x, kv_cache=kv_cache)
        attn_mask = self.seer_mask(q, k)
        return self.attention(q, k, v, attn_mask)
