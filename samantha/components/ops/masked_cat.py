import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def masked_cat2d_fwd_kernel(
    X,
    Y,
    O,
    X_valid,
    Y_valid,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_om,
    stride_on,
    BLOCK_SIZE: tl.constexpr,
):

    row = tl.program_id(axis=0)
    Nx = tl.load(X_valid + row)
    Ny = tl.load(Y_valid + row)
    n = tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_xm + n * stride_xn
    x = tl.load(X, mask=n < Nx, other=-1)
    Y = Y + row * stride_ym + n * stride_yn
    y = tl.load(Y, mask=n < Ny, other=-1)

    O1 = O + row * stride_om + n * stride_on
    tl.store(O1, x, mask=n < Nx)

    O2 = O1 + Nx * stride_on
    tl.store(O2, y, mask=n < Ny)


@triton.jit
def masked_cat2d_bwd_kernel(
    dX,
    dY,
    dO,
    X_valid,
    Y_valid,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_om,
    stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    Nx = tl.load(X_valid + row)
    Ny = tl.load(Y_valid + row)

    n = tl.arange(0, BLOCK_SIZE)
    x = dX + row * stride_xm + n * stride_xn
    y = dY + row * stride_ym + n * stride_yn

    O1 = dO + row * stride_om + n * stride_on
    O2 = O1 + Nx * stride_on

    O1 = tl.load(O1, mask=n < Nx, other=-1)
    tl.store(x, O1, mask=n < Nx)

    O2 = tl.load(O2, mask=n < Ny)
    tl.store(y, O2, mask=n < Ny)


@triton.jit
def masked_cat3d_fwd_kernel(
    X,
    Y,
    Z,
    O,
    X_valid,
    Y_valid,
    Z_valid,
    stride_xm,
    stride_xn,
    stride_xk,
    stride_ym,
    stride_yn,
    stride_yk,
    stride_zm,
    stride_zn,
    stride_zk,
    stride_om,
    stride_on,
    stride_ok,
    BLOCK_SIZE: tl.constexpr,
):

    px = tl.program_id(axis=0)
    pz = tl.program_id(axis=1)

    Nx = tl.load(X_valid + px)
    Ny = tl.load(Y_valid + px)
    Nz = tl.load(Z_valid + px)
    n = tl.arange(0, BLOCK_SIZE)

    X = X + px * stride_xm + n * stride_xn + pz * stride_xk
    x = tl.load(X, mask=n < Nx, other=-1)
    Y = Y + px * stride_ym + n * stride_yn + pz * stride_yk
    y = tl.load(Y, mask=n < Ny, other=-1)
    Z = Z + px * stride_zm + n * stride_zn + pz * stride_zk
    z = tl.load(Z, mask=n < Nz, other=-1)

    O1 = O + px * stride_om + n * stride_on + pz * stride_ok
    O2 = O1 + Nx * stride_on
    O3 = O2 + Ny * stride_on

    tl.store(O1, x, mask=n < Nx)
    tl.store(O2, y, mask=n < Ny)
    tl.store(O3, z, mask=n < Nz)


@triton.jit
def masked_cat3d_bwd_kernel(
    dX,
    dY,
    dZ,
    dO,
    X_valid,
    Y_valid,
    Z_valid,
    stride_xm,
    stride_xn,
    stride_xk,
    stride_ym,
    stride_yn,
    stride_yk,
    stride_zm,
    stride_zn,
    stride_zk,
    stride_om,
    stride_on,
    stride_ok,
    BLOCK_SIZE: tl.constexpr,
):
    px = tl.program_id(axis=0)
    pz = tl.program_id(axis=1)

    Nx = tl.load(X_valid + px)
    Ny = tl.load(Y_valid + px)
    Nz = tl.load(Z_valid + px)

    n = tl.arange(0, BLOCK_SIZE)
    x = dX + px * stride_xm + n * stride_xn + pz * stride_xk
    y = dY + px * stride_ym + n * stride_yn + pz * stride_yk
    z = dZ + px * stride_zm + n * stride_zn + pz * stride_zk

    O1 = dO + px * stride_om + n * stride_on + pz * stride_ok
    O2 = O1 + Nx * stride_on
    O3 = O2 + Ny * stride_on

    O1 = tl.load(O1, mask=n < Nx, other=-1)
    tl.store(x, O1, mask=n < Nx)

    O2 = tl.load(O2, mask=n < Ny)
    tl.store(y, O2, mask=n < Ny)

    O3 = tl.load(O3, mask=n < Nz)
    tl.store(z, O3, mask=n < Nz)


def masked_cat2d_fwd(x, y, x_valid, y_valid):
    assert x.is_cuda and y.is_cuda
    assert x.shape[0] == y.shape[0] == x_valid.shape[0] == y_valid.shape[0]
    M = x.shape[0]
    N = torch.max(x_valid + y_valid).item()
    BLOCK_SIZE = triton.next_power_of_2(N)

    o = torch.zeros([M, N], dtype=x.dtype, device=x.device)
    grid = (M,)
    masked_cat2d_fwd_kernel[grid](
        x,
        y,
        o,
        x_valid,
        y_valid,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        o.stride(0),
        o.stride(1),
        BLOCK_SIZE,
    )
    return o


def masked_cat2d_bwd(do, x, y, x_valid, y_valid):
    assert x.is_cuda and y.is_cuda
    assert x.shape[0] == y.shape[0] == x_valid.shape[0] == y_valid.shape[0]
    M = x.shape[0]
    N = torch.max(x_valid + y_valid).item()
    assert do.shape == (M, N)
    BLOCK_SIZE = triton.next_power_of_2(N)
    dx = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    dy = torch.zeros_like(y, dtype=y.dtype, device=y.device)
    grid = (M,)
    masked_cat2d_bwd_kernel[grid](
        dx,
        dy,
        do,
        x_valid,
        y_valid,
        dx.stride(0),
        dx.stride(1),
        dy.stride(0),
        dy.stride(1),
        do.stride(0),
        do.stride(1),
        BLOCK_SIZE,
    )
    return dx, dy, None, None


def masked_cat3d_fwd(x, y, z, x_valid, y_valid, z_valid):
    assert x.is_cuda and y.is_cuda and z.is_cuda
    assert (
        x.shape[0]
        == y.shape[0]
        == z.shape[0]
        == x_valid.shape[0]
        == y_valid.shape[0]
        == z_valid.shape[0]
    )
    assert x.shape[2] == y.shape[2] == z.shape[2]

    M = x.shape[0]
    N = torch.max(x_valid + y_valid + z_valid).item()
    K = x.shape[2]
    BLOCK_SIZE = triton.next_power_of_2(N)

    o = torch.zeros([M, N, K], dtype=x.dtype, device=x.device)
    grid = (M, K)
    masked_cat3d_fwd_kernel[grid](
        x,
        y,
        z,
        o,
        x_valid,
        y_valid,
        z_valid,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        BLOCK_SIZE,
    )
    return o


def masked_cat3d_bwd(do, x, y, z, x_valid, y_valid, z_valid):
    assert x.is_cuda and y.is_cuda and z.is_cuda
    assert (
        x.shape[0]
        == y.shape[0]
        == z.shape[0]
        == x_valid.shape[0]
        == y_valid.shape[0]
        == z_valid.shape[0]
    )
    assert x.shape[2] == y.shape[2] == z.shape[2]
    M = x.shape[0]
    N = torch.max(x_valid + y_valid + z_valid).item()
    K = x.shape[2]
    assert do.shape == (M, N, K)
    BLOCK_SIZE = triton.next_power_of_2(N)
    dx = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    dy = torch.zeros_like(y, dtype=y.dtype, device=y.device)
    dz = torch.zeros_like(z, dtype=z.dtype, device=z.device)

    grid = (M, K)
    masked_cat3d_bwd_kernel[grid](
        dx,
        dy,
        dz,
        do,
        x_valid,
        y_valid,
        z_valid,
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dz.stride(0),
        dz.stride(1),
        dz.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        BLOCK_SIZE,
    )
    return dx, dy, dz, None, None, None


class MaskedCat2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, x_valid, y_valid):
        ctx.save_for_backward(x, y, x_valid, y_valid)
        return masked_cat2d_fwd(x, y, x_valid, y_valid)

    @staticmethod
    def backward(ctx: torch.Any, dz) -> torch.Any:
        x, y, x_valid, y_valid = ctx.saved_tensors
        return masked_cat2d_bwd(dz, x, y, x_valid, y_valid)


class MaskedCat3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, z, x_valid, y_valid, z_valid):
        ctx.save_for_backward(x, y, z, x_valid, y_valid, z_valid)
        return masked_cat3d_fwd(x, y, z, x_valid, y_valid, z_valid)

    @staticmethod
    def backward(ctx: torch.Any, do) -> torch.Any:
        x, y, z, x_valid, y_valid, z_valid = ctx.saved_tensors
        return masked_cat3d_bwd(do, x, y, z, x_valid, y_valid, z_valid)


masked_cat2d = MaskedCat2d.apply
masked_cat3d = MaskedCat3d.apply
