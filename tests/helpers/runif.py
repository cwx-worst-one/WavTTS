import os
import sys
from typing import Optional

import pytest
import torch
from packaging.version import Version
from pkg_resources import get_distribution


class RunIf:
    """RunIf wrapper for simple marking specific cases, fully compatible with pytest.mark::

    @RunIf(min_torch="0.0")
    @pytest.mark.parametrize("arg1", [1, 2.0])
    def test_wrapper(arg1):
        assert arg1 > 0.0
    """

    def __new__(
        self,
        *args,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        min_cuda_gpus: Optional[int] = 0,
        has_hdfs: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            *args: Any :class:`pytest.mark.skipif` arguments.
            min_cuda_gpus: Require this number of gpus .
            **kwargs: Any :class:`pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_cuda_gpus:
            conditions.append(torch.cuda.device_count() < min_cuda_gpus)
            reasons.append(f"GPUs>={min_cuda_gpus}")

        if has_hdfs:
            conditions.append("HADOOP_HOME" not in os.environ)
            reasons.append("HADOOP_HOME does not exist in environment variables.")

        if min_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"  # noqa
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args,
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )


@RunIf(min_cuda_gpus=99)
def test_always_skip():
    exit(1)


@pytest.mark.parametrize("arg1", [1, 2, 4])
@RunIf(min_cuda_gpus=0)
def test_wrapper(arg1: float):
    assert arg1 > 0.0
