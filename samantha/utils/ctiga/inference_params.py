# Copyright (c) 2023, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31 # noqa
from dataclasses import dataclass, field
from typing import Optional

import torch
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
    n_look_future: Optional[int] = None
    last: bool = False
    mem_efficient: bool = True


def get_varlen_kv_cache_idx_v1(
    inference_params: InferenceParams,
    cache_seqlens: torch.Tensor,
    cache_batch_indices: Optional[torch.Tensor] = None,
):
    device = cache_seqlens.device
    bs = len(cache_seqlens)

    if cache_batch_indices is None:
        cache_batch_indices = torch.arange(
            inference_params.batch_size_offset,
            bs + inference_params.batch_size_offset,
            device=device,
            dtype=torch.int,
        )
    assert cache_batch_indices.max() < inference_params.max_batch_size

    if inference_params.lengths_per_sample is None:
        inference_params.lengths_per_sample = torch.zeros(
            (inference_params.max_batch_size,),
            device=cache_seqlens.device,
            dtype=torch.int,
        )

    sequence_start = inference_params.lengths_per_sample[cache_batch_indices]
    sequence_end = sequence_start + cache_seqlens
    assert sequence_end.max() <= inference_params.max_sequence_len

    to_cache_indices = torch.cat(
        [
            (
                torch.arange(
                    sequence_start[bidx],
                    sequence_end[bidx],
                    dtype=torch.int,
                    device=device,
                )
                + cache_bidx * inference_params.max_sequence_len
            )
            for bidx, cache_bidx in enumerate(cache_batch_indices)
        ],
        dim=0,
    )
    cached_indices = torch.cat(
        [
            (
                torch.arange(0, sequence_end[bidx], dtype=torch.int, device=device)
                + cache_bidx * inference_params.max_sequence_len
            )
            for bidx, cache_bidx in enumerate(cache_batch_indices)
        ],
        dim=0,
    )
    return to_cache_indices, cached_indices


def get_varlen_kv_cache_idx_v2(
    inference_params: InferenceParams,
    cache_seqlens: torch.Tensor,
    cache_batch_indices: torch.Tensor,
):
    device = cache_seqlens.device
    # bs = len(cache_seqlens)

    assert cache_batch_indices.max() < inference_params.max_batch_size

    if inference_params.lengths_per_sample is None:
        inference_params.lengths_per_sample = torch.zeros(
            (inference_params.max_batch_size,),
            device=cache_seqlens.device,
            dtype=torch.int,
        )

    sequence_start = inference_params.lengths_per_sample[cache_batch_indices]
    sequence_end = sequence_start + cache_seqlens
    assert sequence_end.max() <= inference_params.max_sequence_len

    to_cache_indices = torch.cat(
        [
            torch.stack(
                [
                    torch.full(
                        (sequence_end[bidx] - sequence_start[bidx],),
                        cache_bidx,
                        device=device,
                        dtype=torch.int,
                    ),
                    torch.arange(
                        sequence_start[bidx],
                        sequence_end[bidx],
                        device=device,
                        dtype=torch.int,
                    ),
                ],
                dim=0,
            )
            for bidx, cache_bidx in enumerate(cache_batch_indices)
        ],
        dim=1,
    )
    cached_indices = torch.cat(
        [
            torch.stack(
                [
                    torch.full(
                        (sequence_end[bidx],),
                        cache_bidx,
                        device=device,
                        dtype=torch.int,
                    ),
                    torch.arange(sequence_end[bidx], device=device, dtype=torch.int),
                ],
                dim=0,
            )
            for bidx, cache_bidx in enumerate(cache_batch_indices)
        ],
        dim=1,
    )
    return to_cache_indices, cached_indices
