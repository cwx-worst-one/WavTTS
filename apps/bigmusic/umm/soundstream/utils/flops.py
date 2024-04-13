from copy import deepcopy
from functools import cached_property, wraps
import logging
import os
import signal
from typing import Tuple
import traceback

import torch
from torch import nn
from thop.profile import register_hooks, count_parameters
from thop.vision.basic_hooks import calculate_conv2d_flops
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from apps.bigmusic.umm.soundstream.models.modules.basic_blocks import Snake1d
from samantha.utils.distributed import get_global_rank, get_node_rank, get_world_size

log = logging.getLogger(__name__)


def with_timeout(timeout=5, tag=""):
    tag = tag.strip()
    tag = "" if not tag else tag + " "
    tag = f"{tag}rank=[{get_global_rank()}/{get_world_size()}/{get_node_rank()}], "

    def _decorator(func):
        def timeout_handler(signum, frame):
            stack = "".join(traceback.format_stack(frame))
            log.warning(
                f"{tag}[{func.__qualname__}] Timeout occurred. Printing stack trace:\n{stack}"
            )

        @wraps(func)
        def _wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            try:
                func(*args, **kwargs)  # invoke the original func
            finally:
                signal.alarm(0)  # cancel alarm

        return _wrapper

    return _decorator


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)


def get_device_memory(device=None):
    torch.cuda.synchronize(device=device)
    mem_gb = torch.cuda.max_memory_allocated(device=device) / 2**30
    malloc_reties = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    return mem_gb, malloc_reties


def get_device_flops(precision="16"):
    precision = str(precision)
    device_name = torch.cuda.get_device_name()
    if "A100" in device_name or "A800" in device_name:
        return 312e12 if precision == "16" else 156e12
    elif "H100" in device_name or "H800" in device_name:
        return 1979e12 if precision == "16" else 989e12
    elif "V100" in device_name:
        return 125e12

    raise Exception(f"unkown device: {device_name}")


def count_conv(m: nn.Module, x, y: torch.Tensor):
    x = x[0]

    m.total_ops += 2 * calculate_conv2d_flops(
        input_size=list(x.shape),
        output_size=list(y.shape),
        kernel_size=list(m.weight.shape),
        groups=m.groups,
        bias=m.bias,
    )


def count_snake(m, x, y):
    x = x[0]
    m.total_ops += m.alpha.numel() * 2 + x.numel() * 5


def get_default_register_hooks():
    default_hooks = deepcopy(register_hooks)
    default_hooks.update(
        {
            Snake1d: count_snake,
            nn.Conv1d: count_conv,
            nn.Conv2d: count_conv,
            nn.Conv3d: count_conv,
        }
    )
    return default_hooks


class FlopsProfiler:
    def __init__(
        self,
        model: nn.Module,
        name="module",
        custom_ops=None,
        verbose=False,
        training=True,
    ):
        self.model: nn.Module = model
        self.name = name
        self.custom_ops: dict = custom_ops or {}
        self.verbose: bool = verbose
        self.training: bool = training

        self.handler_collection = dict()
        self.types_collection = set()
        self.registered = False

        self.flops_ = 0
        self.params_ = 0

    def start(self, reset=True):
        self._register_hooks(reset=reset)
        self.flops_, self.params_ = 0, 0

    def stop(self):
        if not self.registered:
            raise Exception("should call .start() first.")
        self.flops_, self.params_ = self._get_flops_and_params()
        self._unregister_hooks()

    def report(self) -> Tuple[float, int]:
        return self.flops, self.params

    @cached_property
    def flops(self):
        return self.flops_

    @cached_property
    def params(self):
        return self.params_

    def _register_hooks(self, reset=True):
        default_register_hooks = get_default_register_hooks()

        def add_hooks(m: nn.Module):
            if self.registered:
                if not reset:
                    return
                m.total_ops = 0.0
                m.total_params = 0.0
                return
            m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
            m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

            m_type = type(m)
            fn = None
            if m_type in self.custom_ops:
                # if defined both op maps, use custom_ops to overwrite.
                fn = self.custom_ops[m_type]
                if m_type not in self.types_collection and self.verbose:
                    rank_zero_print(
                        "[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type)
                    )
            elif m_type in default_register_hooks:
                fn = default_register_hooks[m_type]
                if m_type not in self.types_collection and self.verbose:
                    rank_zero_print(
                        "[INFO] Register %s() for %s." % (fn.__qualname__, m_type)
                    )
            else:
                if m_type not in self.types_collection and self.verbose:
                    rank_zero_print(
                        "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                        % m_type
                    )

            if fn is not None:
                self.handler_collection[m] = (
                    m.register_forward_hook(fn),
                    m.register_forward_hook(count_parameters),
                )
            self.types_collection.add(m_type)

        self.model.apply(add_hooks)
        self.registered = True

    def _unregister_hooks(self):
        if not self.registered:
            return
        # reset model to original status
        for m, (op_handler, params_handler) in self.handler_collection.items():
            op_handler.remove()
            params_handler.remove()
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")
        self.registered = False

    def _get_flops_and_params(self):
        def dfs_count(module: nn.Module, name="", prefix="") -> Tuple[int, int, dict]:
            if not (hasattr(module, "total_ops") and hasattr(module, "total_params")):
                return 0, 0, {}

            total_ops, total_params = module.total_ops.item(), 0
            ret_dict = {}
            for n, m in module.named_children():
                next_dict = {}
                if m in self.handler_collection and not isinstance(
                    m, (nn.Sequential, nn.ModuleList)
                ):
                    m_ops, m_params = m.total_ops.item(), m.total_params.item()
                else:
                    m_ops, m_params, next_dict = dfs_count(
                        m, name=".".join([name, n]), prefix=prefix + "\t"
                    )
                ret_dict[n] = (m_ops, m_params, next_dict)
                total_ops += m_ops
                total_params += m_params
                if self.verbose:
                    n1 = ".".join([name, n])
                    rank_zero_print(
                        f"name={n1}, ops={m_ops}, params={m_params}, m={type(m)}"
                    )

            return total_ops, total_params, ret_dict

        total_ops, total_params, _ = dfs_count(self.model, name=self.name)
        rate_ops = self.training and 3.0 or 1.0

        if self.verbose:
            rank_zero_print(
                f"name={self.name}, ops={total_ops}, params={total_params}, m={type(self.model)}"
            )

        return total_ops * rate_ops, total_params
