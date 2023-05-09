import pytest
import torch

from tests.helpers.runif import RunIf


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("batch_size, seq_len, dtype", [(1151, 8192, torch.float32)])
def test_fused_layer_norm(batch_size, seq_len, dtype, eps=1e-5):
    import triton

    from samantha.byteformers.triton.layer_norm import FusedLayerNorm

    torch.manual_seed(0)

    x_shape = (batch_size, seq_len)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)

    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)

    y_tri = FusedLayerNorm.apply(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)

    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]

    triton.testing.assert_almost_equal(y_tri, y_ref)
    triton.testing.assert_almost_equal(dx_tri, dx_ref)
    triton.testing.assert_almost_equal(db_tri, db_ref, decimal=1)
    triton.testing.assert_almost_equal(dw_tri, dw_ref, decimal=1)
