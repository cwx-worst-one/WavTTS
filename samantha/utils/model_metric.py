import time
import warnings

import torch
from pytorch_lightning.trainer.states import RunningStage
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn

from .flops_calculator import retrieve_calculator


class ModelMetric(Metric):
    def __init__(self, precision, model_obj):
        super().__init__()
        self.flops_per_token = retrieve_calculator(model_obj)
        self.metric_available = self.flops_per_token is not None
        if not self.metric_available:
            rank_zero_warn(
                "`flops_per_token()` is not available, "
                "disable model metric calculation."
            )
            return
        self.add_state("num_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_time", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("delta_flops", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.precision = precision
        self.last_time = 0
        self.last_step = -1

    def update(self, batch_size, seq_length, stage, exclude_time=0.0):
        if not self.metric_available:
            return
        if self.last_time == 0:
            self.last_time = time.perf_counter()
            return

        cur_time = time.perf_counter()
        # only log training metric
        if stage != RunningStage.TRAINING:
            self.last_time = cur_time
            return

        num_tokens = batch_size * seq_length
        self.num_tokens += num_tokens
        self.delta_time += cur_time - self.last_time - exclude_time
        self.delta_flops += num_tokens * self.flops_per_token(seq_length)
        self.last_time = cur_time

    def compute(self, step):
        if not self.metric_available or step == self.last_step or self.delta_time == 0:
            return {}
        self.last_step = step
        flops = self.delta_flops / self.delta_time
        throughput = self.num_tokens / self.delta_time / 1e6
        metric = {
            "train/TFlops": flops / (2**40),
            "train/mfu": flops / self.flops_theoretical(),
            "train/tokens_per_second(M)": throughput,
        }
        self.reset()
        return metric

    def flops_theoretical(self):
        r"""A100 theoretical throughput based on which precision training on.

        See Also:
             https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf  # noqa

        Returns:
            float: theoretical peak throughput

        """

        if self.precision in (32, "32", "32-true"):
            return 19.5e12
        if self.precision in (16, "16", "16-mixed", "bf16", "bf16-mixed"):
            return 312e12
        if self.precision in (64, "64", "64-true"):
            return 9.7e12
        warnings.warn(
            f"Unsupported precision {self.precision=}, flops_theoretical set to 1."
        )
        return 1.0
