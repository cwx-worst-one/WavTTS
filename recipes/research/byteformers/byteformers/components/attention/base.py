from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from byteformers.components.attention.sdp import (
    flash_scaled_dot_product,
    scaled_dot_product,
    setup_kernels,
)
from byteformers.components.positional_embedding.rotary import RotaryEmbedding

logger = getLogger(__name__)
from uuid import uuid4

try:
    from byteformers.components.attention.sdp import flash_2_scaled_dot_product

    _flash_2_attn_installed = True
except ImportError as e:
    _flash_2_attn_installed = False
    print(e)

TensorPointerDict = Dict[int, torch.Tensor]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        scale: Optional[float] = None,
        is_causal: bool = False,
        use_rotary_embeddings: bool = False,
        d_k: Optional[int] = None,
        fused_qkv: bool = True,
        fused_kv: bool = False,
        enable_flash: bool = False,
        enable_mem_efficient: bool = False,
        enable_math: bool = True,
        enable_flash_2: bool = False,
    ) -> None:
        super().__init__()
        if enable_flash_2:
            self.seq_before_head_dim = True
        else:
            self.seq_before_head_dim = False
        
        self.default_seq_dim = -3 if self.seq_before_head_dim else -2
        self._use_cache = False
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_p = dropout
        self.is_causal = is_causal

        assert not (fused_qkv and fused_kv), "Either qkv or kv can be fused"

        self.fused_qkv = fused_qkv
        self.fused_kv = fused_kv
        self.d_k = d_k
        if not self.d_k:
            self.d_k = d_model // n_heads

        self.scale = scale
        if scale is None:
            self.scale = self.d_k**-0.5

        if self.seq_before_head_dim:
            self.rearrange_heads = Rearrange(
                "b s (n_heads d_k) -> b s n_heads d_k", n_heads=self.n_heads
            )
            self.rearrange_score = Rearrange(
                "b s n_heads d_k -> b s (n_heads d_k)"
            )
        else:
            self.rearrange_heads = Rearrange(
                "b s (n_heads d_k) -> b n_heads s d_k", n_heads=self.n_heads
            )
            self.rearrange_score = Rearrange(
                "b n_heads s d_k -> b s (n_heads d_k)"
            )

        self.qkv_dim_with_heads = self.d_k * self.n_heads
        if self.fused_qkv:
            self.to_qkv = nn.Linear(self.d_model, self.qkv_dim_with_heads * 3)
        elif self.fused_kv:
            self.to_q = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
            self.to_kv = nn.Linear(self.d_model, self.qkv_dim_with_heads * 2)
        else:
            self.to_q = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
            self.to_k = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
            self.to_v = nn.Linear(self.d_model, self.qkv_dim_with_heads, bias=bias)
        self.WO = nn.Linear(self.qkv_dim_with_heads, d_model, bias=bias)

        if use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(self.d_k, seq_before_head_dim=self.seq_before_head_dim)

        self.setup_kernels(enable_flash_2, enable_flash, enable_mem_efficient, enable_math)

    @property
    def identifier(self) -> str:
        return str(self.WO.weight.data_ptr())
    
    def setup_kernels(self, enable_flash_2: bool, enable_flash: bool, enable_mem_efficient: bool, enable_math: bool) -> None:
        if (enable_flash or enable_mem_efficient) and not hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            raise ImportError("Flash Attention requires PyTorch >= 2.0")

        if enable_flash_2 and not _flash_2_attn_installed:
            raise ImportError("Flash Attention 2 requires flash_attn >= 2.0")

        if enable_flash_2 and (enable_flash or enable_mem_efficient):
            logger.warn(
                "Flash Attention 2 is enabled, overriding `enable_flash`, `enable_mem_efficient` and `enable_math` SDP kernels."
            )

        self.enable_flash_2 = enable_flash_2
        self.enable_flash = enable_flash
        self.enable_mem_efficient = enable_mem_efficient
        self.enable_math = enable_math
        logger.warn(f"Initializing kernels: Flash2: {self.enable_flash_2}, Flash: {self.enable_flash}, Mem efficient: {self.enable_mem_efficient}, Math: {self.enable_math}")
        setup_kernels(self.enable_flash, self.enable_mem_efficient, self.enable_math)

    def qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[TensorPointerDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if self.fused_qkv:
            assert context is None, "fused_qkv=True is not compatible with `context`"
            qkv = self.to_qkv(x)
            qkv = self.rearrange_heads(qkv)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            if context is not None:
                kv = context
            else:
                kv = x

            q = self.to_q(x)
            q = self.rearrange_heads(q)
            if self.fused_kv:
                kv = self.to_kv(kv)
                kv = self.rearrange_heads(kv)
                k, v = kv.chunk(2, dim=-1)
            else:
                k = self.to_k(kv)
                v = self.to_v(kv)
                k = self.rearrange_heads(k)
                v = self.rearrange_heads(v)

        if kv_cache is not None:
            self._use_cache = True
            k_key = f"{self.identifier}_k"
            v_key = f"{self.identifier}_v"
            kv_cache[k_key] = (
                torch.cat((kv_cache[k_key], k), dim=2)
                if k_key in kv_cache
                else k
            )
            kv_cache[v_key] = (
                torch.cat((kv_cache[v_key], v), dim=2)
                if v_key in kv_cache
                else v
            )

            k = kv_cache[k_key]
            v = kv_cache[v_key]
        else:
            self._use_cache = False

        if hasattr(self, "rotary_emb"):
            cache_len = k.shape[self.default_seq_dim] if self._use_cache else None
            q, k = self.rotary_emb(q, k, q_len=cache_len)
        return q, k, v

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_attn_probs: bool = False,
    ) -> torch.Tensor:
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

        Returns:
            _type_: _description_
        """
        dropout_p = self.dropout_p if self.training else 0.0
        if self.enable_flash_2:
            score = flash_2_scaled_dot_product(
                q,
                k,
                v,
                scale=self.scale,
                is_causal=self.is_causal,
                dropout_p=dropout_p,
                use_cache=self._use_cache,
            )
        elif self.enable_flash or self.enable_mem_efficient or self.enable_math:
            score = flash_scaled_dot_product(
                q,
                k,
                v,
                is_causal=self.is_causal,
                dropout_p=dropout_p,
                use_cache=self._use_cache,
                enable_flash=self.enable_flash,
                enable_mem_efficient=self.enable_mem_efficient,
                enable_math=self.enable_math,
            )
        else:
            score, attention = scaled_dot_product(
                q, k, v, scale=self.scale, dropout_p=dropout_p, is_causal=self.is_causal
            )
        score = self.rearrange_score(score)
        if return_attn_probs:
            return self.WO(score), score
        return self.WO(score)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[TensorPointerDict] = None,
        return_attn_probs: bool = False,
    ) -> torch.Tensor:
        q, k, v = self.qkv(x, context=context, kv_cache=kv_cache)
        return self.attention(q, k, v, return_attn_probs=return_attn_probs)
