import pytest
import torch

from tests.helpers.runif import RunIf


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("batch_size, seq_len, dtype", [(1151, 8192, torch.float32)])
def test_fused_softmax(batch_size: int, seq_len: int, dtype):
    from samantha.byteformers.triton.softmax import softmax

    x = torch.randn(batch_size, seq_len, dtype=dtype, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
