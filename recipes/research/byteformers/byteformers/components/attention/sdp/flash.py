import torch
from torch.backends.cuda import (
    enable_flash_sdp,
    enable_math_sdp,
    enable_mem_efficient_sdp,
    flash_sdp_enabled,
)


def setup_kernels(enable_flash: bool, enable_mem_efficient: bool, enable_math: bool):
    enable_flash_sdp(enable_flash)
    enable_mem_efficient_sdp(enable_mem_efficient)
    enable_math_sdp(enable_math)


def flash_scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    dropout_p: float,
    use_cache: bool,
    enable_flash: bool,
    enable_mem_efficient: bool,
    enable_math: bool,
):
    _reinit_kernels = False
    if is_causal and use_cache:
        i = q.shape[2]
        j = k.shape[2]
        attn_mask = torch.ones(i, j, dtype=torch.bool, device=q.device).tril(j - i)
        is_causal = False

        if enable_flash or enable_mem_efficient:
            print(
                "The flash/memefficient kernels do not work with causal and use_cache (attn_mask required). Setting `enable_math=True`"
            )
            # raise Exception("The flash/memefficient kernels do not work with causal and use_cache")
            setup_kernels(enable_flash, enable_mem_efficient, enable_math=True)
            _reinit_kernels = True
    else:
        attn_mask = None

    # TODO: add proper logging mechanism to warn user of slower kernel
    # if not flash_sdp_enabled():
    #     logger.warning("The flash attention kernel is not enabled")

    score = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )

    if _reinit_kernels:
        setup_kernels(enable_flash, enable_mem_efficient, enable_math)
    return score
