r"""distributed relevant utilities."""

import os
from contextlib import contextmanager

import torch.distributed as dist


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
