# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/fmha.py
import flash_attn_cuda
import torch

from samantha.utils.ctiga.blockmask import convert_blockmask


def _flash_blocksparse_attn_forward(
    qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal, return_softmax
):
    q, k, v = qkv.unbind(dim=1)
    context, softmax_lse, *rest = flash_attn_cuda.fwd_block(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        blockmask,
        max_s,
        max_s,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
        None,
    )
    # if context.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return context, softmax_lse, S_dmask


def _flash_blocksparse_attn_backward(
    dout,
    qkv,
    dqkv,
    out,
    S_dmask,
    softmax_lse,
    cu_seqlens,
    blockmask,
    dropout_p,
    max_s,
    softmax_scale,
    causal,
):
    q, k, v = qkv.unbind(dim=1)
    dq, dk, dv = dqkv.unbind(dim=1)
    dq, dk, dv, softmax_d = flash_attn_cuda.bwd_block(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens,
        cu_seqlens,
        blockmask,
        max_s,
        max_s,
        dropout_p,
        softmax_scale,
        causal,
        None,
    )
    dqkv = torch.stack([dq, dk, dv], dim=1)

    # if dqkv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dqkv


class FlashBlocksparseAttnFun(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal
    ):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_blocksparse_attn_forward(
            qkv,
            cu_seqlens,
            blockmask,
            dropout_p,
            max_s,
            softmax_scale,
            causal=causal,
            return_softmax=False,
        )
        ctx.save_for_backward(
            qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context

    @staticmethod
    def backward(ctx, dout):
        (
            qkv,
            context,
            S_dmask,
            softmax_lse,
            cu_seqlens,
            blockmask,
            rng_state,
        ) = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        # S_dmask is None, temporarily use another tensor just to get it running
        dqkv = torch.empty_like(qkv)
        dqkv = _flash_blocksparse_attn_backward(
            dout,
            qkv,
            dqkv,
            context,
            context,
            softmax_lse,
            cu_seqlens,
            blockmask,
            ctx.dropout_p,
            ctx.max_s,
            ctx.softmax_scale,
            ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None, None


# We duplicate code to return both the output and the softmax for testing
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class FlashBlocksparseAttnFunWithS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal
    ):
        # Save rng_state because the backward pass is gonna regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_blocksparse_attn_forward(
            qkv,
            cu_seqlens,
            blockmask,
            dropout_p,
            max_s,
            softmax_scale,
            causal=causal,
            return_softmax=True,
        )
        ctx.save_for_backward(
            qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context, S_dmask, softmax_lse

    @staticmethod
    def backward(ctx, dout, _dS_dmask_ignored, _dsoftmax_sum_ignored):
        (
            qkv,
            context,
            S_dmask,
            softmax_lse,
            cu_seqlens,
            blockmask,
            rng_state,
        ) = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dqkv = torch.empty_like(qkv)
        dqkv = _flash_blocksparse_attn_backward(
            dout,
            qkv,
            dqkv,
            context,
            S_dmask,
            softmax_lse,
            cu_seqlens,
            blockmask,
            ctx.dropout_p,
            ctx.max_s,
            ctx.softmax_scale,
            ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None


def flash_blocksparse_attn_func(
    qkv,
    cu_seqlens,
    blockmask,
    dropout_p,
    max_s,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    convert_mask=True,
):
    """dropout_p should be set to 0.0 during evaluation"""
    func = (
        FlashBlocksparseAttnFun
        if not return_attn_probs
        else FlashBlocksparseAttnFunWithS
    )
    if convert_mask:
        blockmask = convert_blockmask(blockmask)
    return func.apply(
        qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal
    )
