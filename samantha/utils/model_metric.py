import time
from typing import Dict, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn

from .flops_calculator import retrieve_calculator


class ModelMetric(Metric):
    r"""A PyTorch Lightning metric to calculate MFU(model flops utilization)

    Args:
        precision (Union[int, str]):
            precision of the model, can be 32, 16, 16-true, etc.
        model_obj_or_objs (Union[torch.nn.Module, Dict[str, torch.nn.Module]]):
            model object or a dict of model objects, if a dict is provided,
            the key should be the name of the model, and the value should
            be the model object.
    """

    def __init__(
        self,
        precision: Union[int, str],
        model_obj_or_objs: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    ):
        super().__init__()
        self.has_multi_model = isinstance(model_obj_or_objs, dict)
        if isinstance(model_obj_or_objs, torch.nn.Module):
            model_obj_or_objs = {"model": model_obj_or_objs}
        self.flops_fn = {
            k: retrieve_calculator(v) for k, v in model_obj_or_objs.items()
        }
        self.metric_available = True
        for model_name, flops_fn in self.flops_fn.items():
            if flops_fn is None:
                self.metric_available = False
                cls_name = model_obj_or_objs[model_name].__class__.__name__
                rank_zero_warn(
                    f"Model `{model_name}`[{cls_name}] is not supported for "
                    "flops calculation, disable model metric calculation."
                )
        if not self.metric_available:
            return
        self.add_state("num_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_time", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("delta_flops", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.precision = precision
        self.last_time = 0
        self.last_step = -1

    def update(self, num_tokens, stage, exclude_time=0.0, model_kwargs=None):
        r"""Update the metric with the number of tokens and the time spent.

        Args:
            num_tokens (int): number of tokens processed in the current step
            stage (RunningStage): current running stage
            exclude_time (float): time spent on other operations like embedder
            model_kwargs (Dict[str, Any]): model kwargs, if has_multi_model is True,
                this should be a dict of dict, the key should be the name of the model,
                and the value should be the kwargs of the model flops_fn.
        """
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

        self.num_tokens += num_tokens
        self.delta_time += cur_time - self.last_time - exclude_time
        if self.has_multi_model:
            for model_name, flops_fn in self.flops_fn.items():
                kwargs = model_kwargs[model_name]
                self.delta_flops += flops_fn(**kwargs)
        else:
            self.delta_flops += self.flops_fn["model"](**model_kwargs)
        self.last_time = cur_time

    def compute(self, step):
        if not self.metric_available or step == self.last_step or self.delta_time == 0:
            return {}
        self.last_step = step
        flops = self.delta_flops / self.delta_time
        throughput = self.num_tokens / self.delta_time / 1e6
        metric = {
            "training/TFlops": flops / (2**40),
            "training/mfu": flops / self.flops_theoretical(),
            "training/tokens_per_sec(M)": throughput,
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
        rank_zero_warn(
            f"Unsupported precision `{self.precision}`, flops_theoretical set to 1."
        )
        return 1.0
