# Copyright (c) 2023, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31 # noqa
from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_sequence_len: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    fused_ft_kernel: bool = False
    lengths_per_sample: Optional[Tensor] = None

    n_look_past: Optional[int] = None
    n_look_feature: Optional[int] = None
    last: bool = False
