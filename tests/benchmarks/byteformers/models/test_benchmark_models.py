import pytest

from tests.benchmarks.byteformers.models.common import benchmark_model_plots
from tests.helpers.runif import RunIf
from tests.unittests.byteformers.models.test_gpt2 import (  # noqa
    model_tester as gpt2_model_tester,
)
from tests.unittests.byteformers.models.test_llama import (  # noqa
    model_tester as llama_model_tester,
)


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
