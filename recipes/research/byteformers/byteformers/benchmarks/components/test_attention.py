import pytest
import torch
import torch.utils.benchmark as benchmark
from byteformers.components.attention import MultiHeadAttention
from torch.nn import MultiheadAttention as MultiheadAttentionPT

from tests.components.test_attention import (
    convert_pytorch_to_byteformers_attention_weights,
)
from tests.runif import RunIf
from tests.utils import torch_device


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [100])
@pytest.mark.parametrize("d_model", [1024])
@pytest.mark.parametrize("n_heads", [128])
@RunIf(min_cuda_gpus=1)
def test_attention_benchmark(batch_size, seq_len, d_model, n_heads):
    test_input = torch.randn(batch_size, seq_len, d_model)

    pt_mha = MultiheadAttentionPT(d_model, n_heads, dropout=0, batch_first=True)

    bf_mha = MultiHeadAttention(
        d_model, n_heads, dropout=0, scale=None, causal=False, enable_flash=False
    )

    bf_mha = convert_pytorch_to_byteformers_attention_weights(pt_mha, bf_mha)

    t_pt_mha = benchmark.Timer(
        stmt="pt_mha(test_input, test_input, test_input)",
        globals={"pt_mha": pt_mha, "test_input": test_input},
    )
    t_pt_mha.timeit(5)  # warmup
    print(t_pt_mha.timeit(50))

    t_bf_mha = benchmark.Timer(
        stmt="bf_mha(test_input)", globals={"bf_mha": bf_mha, "test_input": test_input}
    )
    t_bf_mha.timeit(5)  # warmup
    print(t_bf_mha.timeit(50))
