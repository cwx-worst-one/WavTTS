# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
from torch import nn, sin, pow
from torch.nn import Parameter
from recipes.waveformvae.utils.flops_utils import *
from typing import List

class Snake(nn.Module):
    '''
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        '''
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

    def flops(self, in_shape):
        return 7 * calculate_product(in_shape), in_shape


class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        '''
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x, fused=True):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        '''
        if self.training and fused:
            return SnakebetaTriton.apply(x, self.alpha, self.beta, self.alpha_logscale)
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

    @torch.jit.export
    def flops(self, in_shape: List[int]):
        return 7 * calculate_product(in_shape), in_shape


# triton fused kernel
import triton
import triton.language as tl
from typing import Any

@triton.jit
def snakebeta_fwd_kernel(
    y_ptr,
    x_ptr, a_ptr, b_ptr,
    x_stride0, x_stride1, x_stride2,
    y_stride0, y_stride1, y_stride2,
    B, C, T,
    alpha_logscale: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr):
    """
        x: [B, C, T]
        a: [C]
        b: [C]
        y: [B, C, T]
    """
    # block [B, C, T]
    pid0 = tl.program_id(axis=0)
    gid_i_n = tl.cdiv(B, BLOCK_SIZE_B)
    gid_j_n = tl.cdiv(C, BLOCK_SIZE_C)
    gid_k_n = tl.cdiv(T, BLOCK_SIZE_T)

    gid_i = pid0 // (gid_k_n * gid_j_n)
    gid_j = (pid0 // gid_k_n) % gid_j_n
    gid_k = pid0 % gid_k_n
    i = gid_i*BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B*BLOCK_SIZE_T) // BLOCK_SIZE_T
    j = gid_j*BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    k = gid_k*BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_B*BLOCK_SIZE_T) % BLOCK_SIZE_T
    # load as B*T, C
    offsets = (i*x_stride0+k*x_stride2)[:, None] + (j*x_stride1)[None, :]
    offsets_y = (i*y_stride0+k*y_stride2)[:, None] + (j*y_stride1)[None, :]
    mask_ijk = ((i[:, None]<B) & (k[:, None]<T)) & (j[None, :]<C)
    mask_j = (j<C)

    x = tl.load(x_ptr + offsets, mask=mask_ijk, other=0.)
    a = tl.load(a_ptr + j, mask=mask_j, other=0.)
    b = tl.load(b_ptr + j, mask=mask_j, other=0.)

    if alpha_logscale:
        a = tl.exp(a)
        b = tl.exp(b)
    sin_xa = tl.sin(x * a)
    y = x +  sin_xa * sin_xa / (b + 1e-9)
    tl.store(y_ptr + offsets_y, y, mask=mask_ijk)


@triton.jit
def snakebeta_bwd_kernel(
    dx_ptr, da_ptr, db_ptr,
    dy_ptr, x_ptr, a_ptr, b_ptr,
    x_stride0, x_stride1, x_stride2,
    y_stride0, y_stride1, y_stride2,
    B, C, T,
    alpha_logscale: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr):
    """
        x: [B, C, T]
        a: [C]
        b: [C]
        dy: [B, C, T]

    """
    # block [B, C, T]
    pid0 = tl.program_id(axis=0)
    gid_i_n = tl.cdiv(B, BLOCK_SIZE_B)
    gid_j_n = tl.cdiv(C, BLOCK_SIZE_C)
    gid_k_n = tl.cdiv(T, BLOCK_SIZE_T)

    gid_i = pid0 // (gid_k_n * gid_j_n)
    gid_j = (pid0 // gid_k_n) % gid_j_n
    gid_k = pid0 % gid_k_n
    i = gid_i*BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B*BLOCK_SIZE_T) // BLOCK_SIZE_T
    j = gid_j*BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    k = gid_k*BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_B*BLOCK_SIZE_T) % BLOCK_SIZE_T
    # load as B*T, C
    offsets_x = (i*x_stride0+k*x_stride2)[:, None] + (j*x_stride1)[None, :]
    offsets_y = (i*y_stride0+k*y_stride2)[:, None] + (j*y_stride1)[None, :]
    mask_ijk = ((i[:, None]<B) & (k[:, None]<T)) & (j[None, :]<C)

    mask_j = (j<C)
    x = tl.load(x_ptr + offsets_x, mask=mask_ijk, other=0.)
    dy = tl.load(dy_ptr + offsets_y, mask=mask_ijk, other=0.)
    a = tl.load(a_ptr + j, mask=mask_j, other=0.)
    b = tl.load(b_ptr + j, mask=mask_j, other=0.)

    if alpha_logscale:
        a = tl.exp(a)
        b = tl.exp(b)

    save_a = a
    save_b = b
    # dx = dy + dy * a * 2.0 * tl.sin(x*a) * tl.cos(x*a) * (1.0 / (b+1e-9))
    # da = dy * (x * 2.0 * tl.sin(x*a) * tl.cos(x*a))
    # db = -dy * (tl.sin(x*a) * tl.sin(x*a)) / ((1.0/(b+1e-9)*(1.0/(b+1e-9))))
    xa = x * a
    sin_xa = tl.sin(xa)
    cos_xa = tl.cos(xa)
    rb = 1.0 / (b + 1e-9)
    dyrb = dy * rb
    # combine (sin_xa * cos_xa) will cause error
    dx = dy + dyrb * 2.0 * sin_xa * cos_xa * a
    da = dyrb * 2.0 * sin_xa * cos_xa * x
    rb_sin_xa = sin_xa * rb
    db = -dy * rb_sin_xa * rb_sin_xa
    # reduce add axis 0, [B*T, C] -> [C]
    # TODO: tl.sum rtol must lager 1e-5
    da = tl.sum(da, axis=0)
    db = tl.sum(db, axis=0)
    if alpha_logscale:
        da = da * save_a
        db = db * save_b

    tl.store(dx_ptr + offsets_x, dx, mask=mask_ijk)
    tl.atomic_add(da_ptr + j, da, mask=mask_j)
    tl.atomic_add(db_ptr + j, db, mask=mask_j)


class SnakebetaTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x, a, b, alpha_logscale) -> Any:
        ctx.save_for_backward(x, a, b)
        ctx.alpha_logscale = alpha_logscale
        return snakebeta_triton_forward(x, a, b, alpha_logscale)
    @staticmethod
    def backward(ctx: Any, dy) -> Any:
        x, a, b = ctx.saved_tensors
        alpha_logscale = ctx.alpha_logscale
        return snakebeta_triton_backward(dy, x, a, b, alpha_logscale)


def snakebeta_triton_forward(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha_logscale: bool):
    assert x.is_cuda and a.is_cuda and b.is_cuda
    y = torch.empty_like(x, layout=torch.strided)

    B, C, T = x.shape
    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_SIZE_B"]) *
        triton.cdiv(C, meta["BLOCK_SIZE_C"]) *
        triton.cdiv(T, meta["BLOCK_SIZE_T"]),
    )
    snakebeta_fwd_kernel[grid](
        y, x, a, b,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        B, C, T,
        alpha_logscale,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_C=1,
        BLOCK_SIZE_T=128
    )
    return y

def snakebeta_triton_backward(dy: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha_logscale: bool):
    assert x.is_cuda and a.is_cuda and b.is_cuda
    x = x.contiguous()
    dx = torch.empty_like(x)
    da = torch.zeros_like(a)
    db = torch.zeros_like(b)

    B, C, T = x.shape

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_SIZE_B"]) *
        triton.cdiv(C, meta["BLOCK_SIZE_C"]) *
        triton.cdiv(T, meta["BLOCK_SIZE_T"]),
    )
    block_size_b = min(4, triton.next_power_of_2(B))
    block_size_c = 1 # higher accuracy
    block_size_t = min(256, triton.next_power_of_2(T))
    snakebeta_bwd_kernel[grid](
        dx, da, db,
        dy, x, a, b,
        x.stride(0), x.stride(1), x.stride(2),
        dy.stride(0), dy.stride(1), dy.stride(2),
        B, C, T,
        alpha_logscale,
        BLOCK_SIZE_B=block_size_b,
        BLOCK_SIZE_C=block_size_c,
        BLOCK_SIZE_T=block_size_t
    )
    return dx, da, db, None
