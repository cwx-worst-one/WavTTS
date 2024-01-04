r"""distributed relevant utilities."""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import Tensor


def get_global_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_node_rank() -> int:
    group_rank = os.environ.get("GROUP_RANK", 0)
    return int(os.environ.get("NODE_RANK", group_rank))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 0))


def is_local_zero():
    local_rank = os.getenv("LOCAL_RANK", None)
    return local_rank is None or local_rank == "0"


def is_global_zero():
    rank = os.getenv("RANK", None)
    return rank is None or rank == "0"


@contextmanager
def rank_zero_first(is_global: bool = False):
    r"""A helper function for doing something first on rank_zero and then
    other ranks, like data downloading, model requirements, etc.

    .. note::

        This method will query environment variable ``LOCAL_RANK`` or ``RANK``
        to determine whether local/global rank zero or not.

    If user want download data once and all ranks load data, codes may be

    .. code-block:: python

        if rank == 0:
            download_data()
        else:
            barrier()

        load_data()

        if rank == 0:
            barrier()

    With ``rank_zero_first`` context, user could implement exactly same thing
    as above but more elegant

    .. code-block:: python

        with rank_zeros_first():
            if not os.path.exists(data_path):
                download_data()
            load_data()

    Args:
        is_global (bool): rank zero within global scope or local scope.

    """

    rank_zero_function = is_global_zero if is_global else is_local_zero

    if not dist.is_initialized():
        yield
    else:
        if not rank_zero_function():
            dist.barrier()
        yield
        if rank_zero_function():
            dist.barrier()


# Raw operation, does not support autograd, but does support async
def all_gather_raw(
    input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False
):
    world_size = torch.distributed.get_world_size(process_group)
    output = torch.empty(
        world_size * input_.shape[0],
        *input_.shape[1:],
        dtype=input_.dtype,
        device=input_.device,
    )
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def reduce_scatter_raw(
    input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False
):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(
        input_.shape[0] // world_size,
        *input_.shape[1:],
        dtype=input_.dtype,
        device=input_.device,
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def all_reduce_raw(
    input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False
):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(
        input_, group=process_group, async_op=async_op
    )
    return input_, handle


class AllGatherFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: dist.ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
all_gather = AllGatherFunc.apply


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: dist.ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = reduce_scatter_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
reduce_scatter = ReduceScatterFunc.apply


class AllReduceFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: dist.ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_reduce_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


# Supports autograd, but does not support async
all_reduce = AllReduceFunc.apply


def sync_shared_params(model: torch.nn.Module, process_group: dist.ProcessGroup):
    # We want to iterate over parameters with _shared_params=True in the same order, as
    # different ranks might have different number of parameters
    # e.g., only rank 0 has bias
    pamams_shared = {
        name: p
        for name, p in model.named_parameters()
        if getattr(p, "_shared_params", False)
    }
    for _, p in sorted(pamams_shared.items()):
        with torch.no_grad():
            # Broadcast needs src to be global rank, not group rank
            torch.distributed.broadcast(
                p,
                src=torch.distributed.get_global_rank(process_group, 0),
                group=process_group,
            )


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/52e636888cccc41e931251c417a7181fc36de926/megatron/optimizer/optimizer.py#L256 # noqa
def allreduce_sequence_parallel_grad(
    model: torch.nn.Module, process_group: dist.ProcessGroup
):
    # We want to iterate over parameters with _sequence_parallel=True in the same order
    # as different ranks might have different number of parameters
    # (e.g., only rank 0 has bias).
    params_seqparallel = {
        name: p
        for name, p in model.named_parameters()
        if getattr(p, "_sequence_parallel", False)
    }
    grads = [p.grad for _, p in sorted(params_seqparallel.items())]
    if grads:
        with torch.no_grad():
            coalesced = torch._utils._flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=process_group)
            for buf, synced in zip(
                grads, torch._utils._unflatten_dense_tensors(coalesced, grads)
            ):
                buf.copy_(synced)
