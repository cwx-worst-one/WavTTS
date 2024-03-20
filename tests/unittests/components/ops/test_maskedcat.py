import logging

import pytest
import torch

from samantha.components.ops import masked_cat2d, masked_cat3d

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        import triton  # noqa F401

        from samantha.utils.cuda import get_compute_capability

    except Exception:
        logging.warning(
            "Triton is not available, some optimizations will not be tested."
        )
        _triton_available = False


def naive_masked_cat2d(x, y, x_valid, y_valid):
    M = x.shape[0]
    N = torch.max(x_valid + y_valid).item()
    o = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    for i in range(M):
        o[i, : x_valid[i]] = x[i, : x_valid[i]]
        o[i, x_valid[i] : x_valid[i] + y_valid[i]] = y[i, : y_valid[i]]
    return o


def naive_masked_cat3d(x, y, z, x_valid, y_valid, z_valid):
    M = x.shape[0]
    N = torch.max(x_valid + y_valid + z_valid).item()
    K = x.shape[2]
    o = torch.zeros(M, N, K, dtype=x.dtype, device=x.device)
    for i in range(M):
        o[i, : x_valid[i]] = x[i, : x_valid[i]]
        o[i, x_valid[i] : x_valid[i] + y_valid[i]] = y[i, : y_valid[i]]
        o[i, x_valid[i] + y_valid[i] : x_valid[i] + y_valid[i] + z_valid[i]] = z[
            i, : z_valid[i]
        ]
    return o


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or get_compute_capability() < 8.0,
    reason="bfloat16 requires a SM80+ GPU",
)
def test_masked_cat2d():
    x_valid = torch.as_tensor([1, 2]).cuda()
    y_valid = torch.as_tensor([4, 1]).cuda()
    x = torch.rand(2, 2, dtype=torch.bfloat16, requires_grad=True).cuda()
    y = torch.rand(2, 4, dtype=torch.bfloat16, requires_grad=True).cuda()
    x_ref = torch.rand(2, 2, dtype=torch.bfloat16, requires_grad=True).cuda()
    y_ref = torch.rand(2, 4, dtype=torch.bfloat16, requires_grad=True).cuda()
    x_ref.data, y_ref.data = x.data, y.data
    x.retain_grad()
    y.retain_grad()
    x_ref.retain_grad()
    y_ref.retain_grad()

    o = masked_cat2d(x, y, x_valid, y_valid)
    loss = o.sum()
    loss.backward()

    o_ref = naive_masked_cat2d(x_ref, y_ref, x_valid, y_valid)
    loss_ref = o_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(o, o_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(y.grad, y_ref.grad)


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or get_compute_capability() < 8.0,
    reason="bfloat16 requires a SM80+ GPU",
)
def test_masked_cat3d():
    x_valid = torch.as_tensor([1, 2]).cuda()
    y_valid = torch.as_tensor([4, 1]).cuda()
    z_valid = torch.as_tensor([5, 3]).cuda()

    x = torch.rand(2, 2, 10, dtype=torch.bfloat16, requires_grad=True).cuda()
    y = torch.rand(2, 4, 10, dtype=torch.bfloat16, requires_grad=True).cuda()
    z = torch.rand(2, 5, 10, dtype=torch.bfloat16, requires_grad=True).cuda()

    x_ref = torch.rand(2, 2, 10, dtype=torch.bfloat16, requires_grad=True).cuda()
    y_ref = torch.rand(2, 4, 10, dtype=torch.bfloat16, requires_grad=True).cuda()
    z_ref = torch.rand(2, 5, 10, dtype=torch.bfloat16, requires_grad=True).cuda()
    x_ref.data, y_ref.data, z_ref.data = x.data, y.data, z.data
    x.retain_grad()
    y.retain_grad()
    z.retain_grad()
    x_ref.retain_grad()
    y_ref.retain_grad()
    z_ref.retain_grad()

    o = masked_cat3d(x, y, z, x_valid, y_valid, z_valid)
    loss = o.sum()
    loss.backward()

    o_ref = naive_masked_cat3d(x_ref, y_ref, z_ref, x_valid, y_valid, z_valid)
    loss_ref = o_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(o, o_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(y.grad, y_ref.grad)
    torch.testing.assert_close(z.grad, z_ref.grad)
