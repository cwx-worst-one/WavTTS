# Copyright (c) 2023, Tri Dao.

from typing import Optional, Tuple, Union

import rotary_emb
import torch
from einops import rearrange, repeat

from samantha.components.ctiga.triton.rotary import apply_rotary as apply_rotary_triton


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class ApplyRotaryEmbTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        dtype = x.dtype
        if x.dtype != cos.dtype:
            x = x.to(cos.dtype)
        out = apply_rotary_triton(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(
                cos, sin, cu_seqlens
            )  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return (out if not inplace else x).to(dtype)

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dtype = do.dtype
        if do.dtype != cos.dtype:
            do = do.to(cos.dtype)
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary_triton(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        dx = dx.to(dtype)
        return dx, None, None, None, None, None, None, None


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        dtype = x.dtype
        if x.dtype != cos.dtype:
            x = x.to(cos.dtype)
        seqlen, nheads, headdim = x.shape[-3:]
        rotary_seqlen, rotary_dim = cos.shape[-2:]
        rotary_dim *= 2
        assert seqlen <= rotary_seqlen
        assert rotary_dim <= headdim
        assert sin.shape[-2:] == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = (
            x_ro.chunk(2, dim=-1)
            if not interleaved
            else (x_ro[..., ::2], x_ro[..., 1::2])
        )
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]
        if inplace:
            o1, o2 = x1, x2
        else:
            o1, o2 = (
                out_ro.chunk(2, dim=-1)
                if not interleaved
                else (out_ro[..., ::2], out_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            x1,
            x2,
            rearrange(cos[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin[..., :seqlen, :], "... s d -> ... s 1 d"),
            o1,
            o2,
            False,
        )
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return (out if not inplace else x).to(dtype)

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        dtype = do.dtype
        if do.dtype != cos.dtype:
            do = do.to(cos.dtype)

        seqlen, _, headdim = do.shape[-3:]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        do1, do2 = (
            do_ro.chunk(2, dim=-1)
            if not ctx.interleaved
            else (do_ro[..., ::2], do_ro[..., 1::2])
        )
        dx = torch.empty_like(do) if not inplace else do
        if inplace:
            dx1, dx2 = do1, do2
        else:
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (
                dx_ro.chunk(2, dim=-1)
                if not ctx.interleaved
                else (dx_ro[..., ::2], dx_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            do1,
            do2,
            rearrange(cos[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin[..., :seqlen, :], "... s d -> ... s 1 d"),
            dx1,
            dx2,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        dx = dx.to(dtype)
        return dx, None, None, None, None


apply_rotary_emb_func = ApplyRotaryEmb.apply
apply_rotary_emb_triton_func = ApplyRotaryEmbTriton.apply


class ApplyRotaryEmbQKVTriton_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        dtype = qkv.dtype
        if qkv.dtype != cos.dtype:
            qkv = qkv.to(cos.dtype)
        seqlen, three, nheads, headdim = qkv.shape[-4:]
        assert three == 3
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            qk = rearrange(qkv[..., :2, :, :], "... t h d -> ... (t h) d")
            # qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            apply_rotary_triton(
                qk,
                cos,
                sin,
                seqlen_offsets,
                interleaved=interleaved,
                inplace=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            q, k = qkv[..., 0], qkv[..., 1]
            apply_rotary_triton(
                q,
                cos,
                sin,
                seqlen_offsets,
                interleaved=interleaved,
                inplace=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            apply_rotary_triton(
                k,
                cos_k,
                sin_k,
                seqlen_offsets,
                interleaved=interleaved,
                inplace=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None

        ctx.interleaved = interleaved
        ctx.max_seqlen = max_seqlen
        return qkv.to(dtype)

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k, cu_seqlens = ctx.saved_tensors
        dtype = dqkv.dtype
        if dqkv.dtype != cos.dtype:
            dqkv = dqkv.to(cos.dtype)
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            dqk = rearrange(dqkv[..., :2, :, :], "... t h d -> ... (t h) d")
            apply_rotary_triton(
                dqk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            dq, dk = dqkv[..., 0, :, :], dqkv[..., 1, :, :]
            apply_rotary_triton(
                dq,
                cos,
                sin,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
            )
            apply_rotary_triton(
                dk,
                cos_k,
                sin_k,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
            )
        dqkv = dqkv.to(dtype)
        return dqkv, None, None, None, None, None, None, None, None


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        """
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """

        dtype = qkv.dtype
        if qkv.dtype != cos.dtype:
            qkv = qkv.to(cos.dtype)
        seqlen, three, nheads, headdim = qkv.shape[-4:]
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape[-2:]
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert (
            sin.shape[-2:]
            == cos_k.shape[-2:]
            == sin_k.shape[-2:]
            == (rotary_seqlen, rotary_dim // 2)
        )
        q_ro = qkv[..., :, 0, :, :rotary_dim]
        q1, q2 = (
            q_ro.chunk(2, dim=-1)
            if not interleaved
            else (q_ro[..., ::2], q_ro[..., 1::2])
        )
        rotary_emb.apply_rotary(
            q1,
            q2,
            rearrange(cos[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin[..., :seqlen, :], "... s d -> ... s 1 d"),
            q1,
            q2,
            False,
        )
        k_ro = qkv[..., :, 1, :, :rotary_dim]
        k1, k2 = (
            k_ro.chunk(2, dim=-1)
            if not interleaved
            else (k_ro[..., ::2], k_ro[..., 1::2])
        )
        rotary_emb.apply_rotary(
            k1,
            k2,
            rearrange(cos_k[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin_k[..., :seqlen, :], "... s d -> ... s 1 d"),
            k1,
            k2,
            False,
        )
        qkv = qkv.to(dtype)
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        dtype = dqkv.dtype
        if dqkv.dtype != cos.dtype:
            dqkv = dqkv.to(cos.dtype)
        seqlen, _, _, headdim = dqkv.shape[-4:]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq_ro = dqkv[..., :, 0, :, :rotary_dim]
        dq1, dq2 = (
            dq_ro.chunk(2, dim=-1)
            if not ctx.interleaved
            else (dq_ro[..., ::2], dq_ro[..., 1::2])
        )
        rotary_emb.apply_rotary(
            dq1,
            dq2,
            rearrange(cos[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin[..., :seqlen, :], "... s d -> ... s 1 d"),
            dq1,
            dq2,
            True,
        )
        dk_ro = dqkv[..., :, 1, :, :rotary_dim]
        dk1, dk2 = (
            dk_ro.chunk(2, dim=-1)
            if not ctx.interleaved
            else (dk_ro[..., ::2], dk_ro[..., 1::2])
        )
        rotary_emb.apply_rotary(
            dk1,
            dk2,
            rearrange(cos_k[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin_k[..., :seqlen, :], "... s d -> ... s 1 d"),
            dk1,
            dk2,
            True,
        )
        dqkv = dqkv.to(dtype)
        return dqkv, None, None, None, None, None


apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply
apply_rotary_emb_qkv_triton_ = ApplyRotaryEmbQKVTriton_.apply


class ApplyRotaryEmbKVTriton_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        dtype = kv.dtype
        if kv.dtype != cos.dtype:
            kv = kv.to(cos.dtype)
        seqlen, two, nheads, headdim = kv.shape[-4:]
        assert two == 2
        k = kv[..., 0, :, :]
        apply_rotary_triton(
            k,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=interleaved,
            inplace=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(
                cos, sin, cu_seqlens
            )  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets, cu_seqlens)
            ctx.seqlen_offsets = None
        ctx.max_seqlen = max_seqlen
        ctx.interleaved = interleaved
        return kv.to(dtype)

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets, cu_seqlens = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dtype = dkv.dtype
        if dkv.dtype != cos.dtype:
            dkv = dkv.to(cos.dtype)
        apply_rotary_triton(
            dkv[..., 0, :, :],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
        )
        dkv = dkv.to(dtype)
        return dkv, None, None, None, None, None, None


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cos, sin, interleaved=False):
        """
            kv: (batch_size, seqlen, 2, nheads, headdim)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        dtype = kv.dtype
        if kv.dtype != cos.dtype:
            kv = kv.to(cos.dtype)
        seqlen, two, nheads, headdim = kv.shape[-4:]
        assert two == 2
        rotary_seqlen, rotary_dim = cos.shape[-2:]
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen, f"{seqlen=} {rotary_seqlen=}"
        assert sin.shape[-2:] == cos.shape[-2:] == (rotary_seqlen, rotary_dim // 2)
        k_ro = kv[..., :, 0, :, :rotary_dim]
        k1, k2 = (
            k_ro.chunk(2, dim=-1)
            if not interleaved
            else (k_ro[..., ::2], k_ro[..., 1::2])
        )
        rotary_emb.apply_rotary(
            k1,
            k2,
            rearrange(cos[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin[..., :seqlen, :], "... s d -> ... s 1 d"),
            k1,
            k2,
            False,
        )
        kv = kv.to(dtype)
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        cos, sin = ctx.saved_tensors
        dtype = dkv.dtype
        if dkv.dtype != cos.dtype:
            dkv = dkv.to(cos.dtype)
        seqlen, _, _, headdim = dkv.shape[-4:]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dk_ro = dkv[..., :, 0, :, :rotary_dim]
        dk1, dk2 = (
            dk_ro.chunk(2, dim=-1)
            if not ctx.interleaved
            else (dk_ro[..., ::2], dk_ro[..., 1::2])
        )
        rotary_emb.apply_rotary(
            dk1,
            dk2,
            rearrange(cos[..., :seqlen, :], "... s d -> ... s 1 d"),
            rearrange(sin[..., :seqlen, :], "... s d -> ... s 1 d"),
            dk1,
            dk2,
            True,
        )
        dkv = dkv.to(dtype)
        return dkv, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply
apply_rotary_emb_kv_triton_ = ApplyRotaryEmbKVTriton_.apply


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000,
        interleaved=False,
        scale_base=None,
        device=None,
        compat="byteformer",
        use_triton=False,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        """
        super().__init__()
        assert compat in ["default", "byteformer"]

        self.base = base
        self.compat = compat
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float().to(device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2).float().to(device=device) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

        self.use_triton = use_triton

    def _update_cos_sin_cache(
        self,
        x,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim) or (cum_seqlen, nheads, headdim) or (cum_seqlen, 3, nheads, headdim)"""
        assert not ((cu_seqlens is None) ^ (max_seqlen is None))
        if isinstance(seqlen_offset, torch.Tensor):
            seqlen_offset_i = seqlen_offset.max().item()
        else:
            seqlen_offset_i = seqlen_offset
        if max_seqlen is None:
            # fixlen
            seqlen = x.shape[1] + seqlen_offset_i
        else:
            # varlen
            seqlen = max_seqlen + seqlen_offset_i
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seqlen
            # t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype) # align byteformer
            if self.compat == "byteformer":
                t = torch.arange(seqlen, device=x.device, dtype=torch.float32)
                freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype)).float()
            else:
                t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                freqs = torch.outer(t, self.inv_freq.to(device=t.device))

            if self.scale is None:
                self._cos_cached = torch.cos(freqs)
                self._sin_cached = torch.sin(freqs)
                if self.compat != "byteformer":
                    self._cos_cached = self._cos_cached.to(x.dtype)
                    self._sin_cached = self._sin_cached.to(x.dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(
                    power, "s -> s 1"
                )
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)
                if self.compat != "byteformer":
                    self._cos_cached = self._cos_cached.to(x.dtype)
                    self._sin_cached = self._sin_cached.to(x.dtype)
                    self._cos_k_cached = self._cos_k_cached.to(x.dtype)
                    self._sin_k_cached = self._sin_k_cached.to(x.dtype)

    @staticmethod
    def get_varlen_cos_sin_cache(
        x, cos, sin, seqlen_offset, cu_seqlens, max_seqlen, use_triton=False
    ):
        if not use_triton:
            if isinstance(seqlen_offset, int):
                indices = torch.cat(
                    [
                        torch.arange(
                            seqlen_offset,
                            seqlen + seqlen_offset,
                            device=x.device,
                            dtype=torch.long,
                        )
                        for seqlen in cu_seqlens[1:] - cu_seqlens[:-1]
                    ],
                    dim=0,
                )
            else:
                indices = torch.cat(
                    [
                        torch.arange(
                            offset, seqlen + offset, device=x.device, dtype=torch.long
                        )
                        for offset, seqlen in zip(
                            seqlen_offset, cu_seqlens[1:] - cu_seqlens[:-1]
                        )
                    ],
                    dim=0,
                )
            cos = torch.index_select(cos, 0, indices)
            sin = torch.index_select(sin, 0, indices)
            assert (
                len(cos) == len(sin) == x.shape[0]
            ), f"{(len(cos), len(sin), x.shape[0])=}"
        return cos, sin

    @staticmethod
    def get_fixlen_cos_sin_cache(x, cos, sin, seqlen_offset, use_triton=False):
        seqlen = x.shape[1]
        if not use_triton:
            if isinstance(seqlen_offset, int):
                cos = cos[seqlen_offset : seqlen_offset + seqlen]
                sin = sin[seqlen_offset : seqlen_offset + seqlen]
            else:
                cos = torch.stack(
                    [cos[offset : offset + seqlen] for offset in seqlen_offset], dim=0
                )  # b,t,d//2
                sin = torch.stack(
                    [sin[offset : offset + seqlen] for offset in seqlen_offset], dim=0
                )  # b,t,d//2
        return cos, sin

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor],
        seqlen_offset: Union[int, torch.Tensor],
        seqlen_offset_k: Optional[Union[int, torch.Tensor]],
        cu_seqlens: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        max_seqlen_k: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qkv/q: (batch, seqlen, 3/1, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        assert not ((cu_seqlens is None) ^ (max_seqlen is None))
        is_varlen = cu_seqlens is not None and max_seqlen is not None

        if kv is None:
            # qkvpacked
            assert (
                seqlen_offset_k is None
                and max_seqlen_k is None
                and cu_seqlens_k is None
            )
            self._update_cos_sin_cache(qkv, seqlen_offset, cu_seqlens, max_seqlen)
            if is_varlen:
                cos_cached, sin_cached = self.get_varlen_cos_sin_cache(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    seqlen_offset,
                    cu_seqlens,
                    max_seqlen,
                    self.use_triton,
                )
            else:
                cos_cached, sin_cached = self.get_fixlen_cos_sin_cache(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    seqlen_offset,
                    self.use_triton,
                )
            if self.scale is None:
                cos_k_cached = None
                sin_k_cached = None
            else:
                if is_varlen:
                    cos_k_cached, sin_k_cached = self.get_varlen_cos_sin_cache(
                        qkv,
                        self._cos_k_cached,
                        self._sin_k_cached,
                        seqlen_offset,
                        cu_seqlens,
                        max_seqlen,
                        self.use_triton,
                    )
                else:
                    cos_k_cached, sin_k_cached = self.get_fixlen_cos_sin_cache(
                        qkv,
                        self._cos_k_cached,
                        self._sin_k_cached,
                        seqlen_offset,
                        self.use_triton,
                    )
            if self.use_triton:
                return apply_rotary_emb_qkv_triton_(
                    qkv,
                    cos_cached,
                    sin_cached,
                    cos_k_cached,
                    sin_k_cached,
                    self.interleaved,
                    seqlen_offset,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    cos_cached,
                    sin_cached,
                    cos_k_cached,
                    sin_k_cached,
                    self.interleaved,
                )
        else:
            # TODO: support kv seqlen_offset
            q = qkv
            self._update_cos_sin_cache(q, seqlen_offset, cu_seqlens, max_seqlen)
            if is_varlen:
                cos_cached, sin_cached = self.get_varlen_cos_sin_cache(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    seqlen_offset,
                    cu_seqlens,
                    max_seqlen,
                    self.use_triton,
                )
            else:
                cos_cached, sin_cached = self.get_fixlen_cos_sin_cache(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    seqlen_offset,
                    self.use_triton,
                )

            if self.use_triton:
                q = apply_rotary_emb_triton_func(
                    q,
                    cos_cached,
                    sin_cached,
                    self.interleaved,
                    True,
                    seqlen_offset,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                q = apply_rotary_emb_func(
                    q, cos_cached, sin_cached, self.interleaved, True
                )
            if seqlen_offset_k is None:
                seqlen_offset_k = 0
            self._update_cos_sin_cache(kv, seqlen_offset_k, cu_seqlens_k, max_seqlen_k)
            if self.scale is None:
                if is_varlen:
                    cos_k_cached, sin_k_cached = self.get_varlen_cos_sin_cache(
                        kv,
                        self._cos_cached,
                        self._sin_cached,
                        seqlen_offset_k,
                        cu_seqlens_k,
                        max_seqlen_k,
                        self.use_triton,
                    )
                else:
                    cos_k_cached, sin_k_cached = self.get_fixlen_cos_sin_cache(
                        kv,
                        self._cos_cached,
                        self._sin_cached,
                        seqlen_offset_k,
                        self.use_triton,
                    )
            else:
                if is_varlen:
                    cos_k_cached, sin_k_cached = self.get_varlen_cos_sin_cache(
                        kv,
                        self._cos_k_cached,
                        self._sin_k_cached,
                        seqlen_offset_k,
                        cu_seqlens_k,
                        max_seqlen_k,
                        self.use_triton,
                    )
                else:
                    cos_k_cached, sin_k_cached = self.get_fixlen_cos_sin_cache(
                        kv,
                        self._cos_k_cached,
                        self._sin_k_cached,
                        seqlen_offset_k,
                        self.use_triton,
                    )

            if self.use_triton:
                kv = apply_rotary_emb_kv_triton_(
                    kv,
                    cos_k_cached,
                    sin_k_cached,
                    self.interleaved,
                    seqlen_offset_k,
                    cu_seqlens_k,
                    max_seqlen_k,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv, cos_k_cached, sin_k_cached, self.interleaved
                )
            return q, kv
