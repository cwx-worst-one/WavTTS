import matplotlib.pyplot as plt
import pytest
import torch
import triton
from byteformers.triton.layer_norm import FusedLayerNorm

from tests.runif import RunIf


@RunIf(min_cuda_gpus=1)
@pytest.mark.benchmark
@pytest.mark.disable
def test_bench_layer_norm():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[256 * i for i in range(2, 10)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="GB/s",
            plot_name="layer-norm",
            args={"M": 512, "dtype": torch.float32, "mode": "forward"},
        )
    )
    def bench_layer_norm(M, N, dtype, provider, mode, eps=1e-5, device="cuda"):
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1],)

        # N x 1
        weight = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)

        # M x N
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
        dy = 0.1 * torch.randn_like(x)
        x.requires_grad_(True)
        # utility functions
        if provider == "triton":
            layer_norm = FusedLayerNorm(w_shape, bias=True, affine=True, eps=eps)
            y_fwd = lambda: layer_norm.forward(x)  # noqa
        if provider == "torch":
            y_fwd = lambda: torch.nn.functional.layer_norm(  # noqa
                x, w_shape, weight, bias, eps
            )  # noqa
        # forward pass
        if mode == "forward":
            gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6  # noqa
            ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, rep=500)
        # backward pass
        if mode == "backward":
            gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa
            y = y_fwd()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: y.backward(dy, retain_graph=True), grad_to_none=[x], rep=500
            )
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    bench_layer_norm.run(save_path=".", print_data=True)
    plt.savefig("test_bench_layer_norm.png")
    plt.close()
