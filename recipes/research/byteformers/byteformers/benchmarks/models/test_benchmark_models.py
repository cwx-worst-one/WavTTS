import pytest
from benchmarks.models.common import benchmark_model_plots

from tests.models.test_gpt2 import model_tester as gpt2_model_tester  # noqa
from tests.models.test_llama import model_tester as llama_model_tester  # noqa
from tests.runif import RunIf


@RunIf(min_cuda_gpus=1)
@pytest.mark.benchmark
@pytest.mark.disable
def test_benchmark_llama(llama_model_tester):  # noqa
    benchmark_model_plots(llama_model_tester, num_steps=50)


@RunIf(min_cuda_gpus=1)
@pytest.mark.benchmark
@pytest.mark.disable
def test_benchmark_gpt2(gpt2_model_tester):  # noqa
    benchmark_model_plots(gpt2_model_tester, num_steps=50)
