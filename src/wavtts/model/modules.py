"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb
from transformers import AutoModel, AutoFeatureExtractor

from wavtts.model.utils import is_package_available, lens_to_mask


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x


# rotary positional embedding related


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if self.context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x
        c: float["b n d"] = None,  # context c
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor

if is_package_available("flash_attn"):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input


class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int | None = None,  # number of attention head to apply rope, None for all
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):
        if attn_backend == "flash_attn":
            assert is_package_available("flash_attn"), "Please install flash-attn first."

        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, q_max_seqlen_in_batch)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        c: float["b nt d"] = None,  # context c, here text
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c


# DiT Block


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(
        self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_dim=None, context_pre_only=False, qk_norm=None
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNorm_Final(context_dim) if context_pre_only else AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=context_dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        if self.context_pre_only:
            c = None
        else:  # if not last layer
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time

# auxiliary loss for mel spectrogram
class MelSpectrogramLoss(nn.Module):
    """
    General PyTorch implementation of the Multi-Scale Mel Spectrogram Loss from DAC.
    Reference: https://github.com/descriptinc/lyrebird-audiotools
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[Optional[float]] = [None, None, None, None, None, None, None],
        pow: float = 1.0,
        clamp_eps: float = 1e-5,
        weight: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.weight = weight
        self.pow = pow
        self.clamp_eps = clamp_eps

        
        assert len(n_mels) == len(window_lengths), "n_mels and window_lengths must have the same length"
        
        self.mel_transforms = nn.ModuleList()
        
        for i, (nm, wl) in enumerate(zip(n_mels, window_lengths)):
            fmin = mel_fmin[i] if i < len(mel_fmin) else 0.0
            fmax = mel_fmax[i] if i < len(mel_fmax) else None
            
            # DAC default logic: hop_length = window_length // 4
            hop_length = wl // 4
            
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=wl,        # using window_length as n_fft usually
                win_length=wl,
                hop_length=hop_length,
                n_mels=nm,
                f_min=fmin,
                f_max=fmax,
                power=1.0,       # computing Magnitude Mel (power=1.0) initially
                normalized=False,
                center=True,
                pad_mode="reflect"
            )
            self.mel_transforms.append(transform)

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        frame_lengths: torch.Tensor | None = None,
    ):
        """
        Args:
            x_pred: [B, T] or [B, 1, T] Estimated Waveform
            x_true: [B, T] or [B, 1, T] Ground Truth Waveform
        Returns:
            Weighted scalar loss
        """
        # Ensure correct shape [B, T] for torchaudio, or [B, 1, T] is also fine but usually squeeze
        if x_pred.ndim == 3 and x_pred.shape[1] == 1:
            x_pred = x_pred.squeeze(1)
        if x_true.ndim == 3 and x_true.shape[1] == 1:
            x_true = x_true.squeeze(1)
            
        total_loss = x_pred.new_tensor(0.0)

        use_mask = frame_mask is not None
        if use_mask:
            span_starts, span_ends = self._get_span_bounds_from_mask(frame_mask, frame_lengths)
            span_starts = span_starts.to(device=x_pred.device)
            span_ends = span_ends.to(device=x_pred.device)
        
        for mel_transform in self.mel_transforms:
            x_mels = mel_transform(x_pred)
            y_mels = mel_transform(x_true)

            mel_mask = None
            if use_mask:
                hop = int(mel_transform.hop_length)
                mel_t = x_mels.shape[-1]
                mel_mask = torch.zeros(
                    (x_mels.shape[0], mel_t),
                    device=x_mels.device,
                    dtype=torch.bool,
                )
                for i in range(x_mels.shape[0]):
                    s = int(span_starts[i].item())
                    e = int(span_ends[i].item())
                    if e <= s:
                        continue
                    ms = s // hop
                    me = (e + hop - 1) // hop
                    ms = max(0, min(ms, mel_t))
                    me = max(0, min(me, mel_t))
                    if me > ms:
                        mel_mask[i, ms:me] = True

                if not mel_mask.any():
                    continue

                mel_mask = mel_mask.unsqueeze(1)
            
            # Log Magnitude Loss
            # Formula: L1( log10(x^pow + eps), log10(y^pow + eps) )
            x_log = x_mels.clamp(min=self.clamp_eps).pow(self.pow).log10()
            y_log = y_mels.clamp(min=self.clamp_eps).pow(self.pow).log10()

            diff_full = (x_log - y_log).abs()

            if mel_mask is None:
                diff = diff_full.mean(dim=(1, 2))
            else:
                diff = []
                for i in range(diff_full.shape[0]):
                    m = mel_mask[i]
                    if m.any():
                        diff.append(diff_full[i].masked_select(m).mean())
                    else:
                        diff.append(diff_full.new_tensor(0.0))
                diff = torch.stack(diff, dim=0)

            total_loss += diff.mean()

        return total_loss * self.weight

    @staticmethod
    def _get_span_bounds_from_mask(mask: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert boolean frame mask [B, T] into per-sample [start, end) bounds.

        This helper assumes one contiguous masked span per sample (matching current
        random-span masking strategy). If a sample has no masked frame, start=end=0.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()

        bsz, t = mask.shape
        device = mask.device

        if lengths is None:
            lengths = torch.full((bsz,), t, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        starts = torch.zeros((bsz,), device=device, dtype=torch.long)
        ends = torch.zeros((bsz,), device=device, dtype=torch.long)

        for i in range(bsz):
            li = int(lengths[i].item())
            row = mask[i, :li]
            idx = row.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            starts[i] = idx[0]
            ends[i] = idx[-1] + 1

        return starts, ends

    @staticmethod
    def _aligned_random_span_mask(
        lengths: torch.Tensor,
        frac_lengths: torch.Tensor,
        align_to: int,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Sample one contiguous mask span per sample with boundary alignment.

        Args:
            lengths: [B] valid sequence lengths in model frames.
            frac_lengths: [B] desired masked length fraction.
            align_to: span start/end alignment in model frames.
            max_length: optional mask width. If None, use lengths.max().
        Returns:
            bool mask with shape [B, max_length].
        """
        assert align_to >= 1
        device = lengths.device
        lengths = lengths.to(dtype=torch.long)
        frac_lengths = frac_lengths.to(device=device, dtype=torch.float32)

        max_len = int(lengths.max().item()) if max_length is None else int(max_length)
        mask = torch.zeros((lengths.shape[0], max_len), device=device, dtype=torch.bool)

        for i in range(lengths.shape[0]):
            li = int(lengths[i].item())
            if li <= 0:
                continue

            target = int((frac_lengths[i].item() * li))
            target = max(1, min(target, li))

            span_len = ((target + align_to - 1) // align_to) * align_to
            span_len = min(span_len, li)

            max_start = li - span_len
            if max_start <= 0:
                start = 0
            else:
                candidate = torch.randint(0, max_start + 1, (1,), device=device).item()
                start = (candidate // align_to) * align_to
                start = min(start, max_start)

            end = start + span_len
            mask[i, start:end] = True

        return mask
