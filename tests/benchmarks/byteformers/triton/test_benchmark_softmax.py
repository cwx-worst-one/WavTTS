import matplotlib.pyplot as plt
import pytest
import torch
import triton

from tests.helpers.runif import RunIf
from samantha.byteformers.triton.softmax import softmax


@torch.jit.script
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@pytest.mark.benchmark
@pytest.mark.disable
@RunIf(min_cuda_gpus=1)
def test_fused_softmax_benchmark() -> None:
    """Triton should achieve 4x faster performance than the Torch JIT implementation
    of the softmax operation. This confirms that the Torch JIT does not do any
    fusion at all!
    """

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[128 * i for i in range(2, 128)],
            line_arg="provider",
            line_vals=["triton", "torch-native", "torch-jit"],
            line_names=["Triton", "Torch (native)", "Torch (jit)"],
            styles=[("blue", "-"), ("green", "-"), ("green", "--")],
            ylabel="GB/s",
            plot_name="softmax-performance",
            args={"M": 4096},
        )
    )
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        if provider == "torch-native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.softmax(x, axis=-1)
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x))
        if provider == "torch-jit":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x))
        gbps = (
            lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        )  # noqa
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(show_plots=True, print_data=True)
    plt.title("Benchmark - Fused Softmax Operation")
    plt.savefig("test_fused_softmax_benchmark.png")
    plt.close()
