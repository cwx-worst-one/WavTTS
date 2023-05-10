'''
distributed utils.
'''

import functools

try:
    from panther.torch.distributed import (
        distributed_init,
        get_dist_info,
        get_rank,
        get_local_rank,
        get_local_size,
        get_world_size,
        dist_allreduce,
        dist_barrier,
        dist_parallel,
        dist_broadcast_model,
        dist_broadcast,
        ReduceOp,
        get_communicator,
    )
except Exception as e:
    raise Exception("Panther not be installed correctly!") from e


def master_only(func):
    '''only master(rank == 0) do this function.'''

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_rank()
        if rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapper


def local_master_only(func):
    '''only local master(loca_rank == 0) do this function.'''

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_rank = get_local_rank()
        if local_rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapper


__all__ = [
    'distributed_init',
    'get_dist_info',
    'get_rank',
    'get_local_rank',
    'get_world_size',
    'get_local_size',
    'dist_allreduce',
    'dist_barrier',
    'dist_parallel',
    'master_only',
    'local_master_only',
    'dist_broadcast_model',
    'dist_broadcast',
    'ReduceOp',
    'get_communicator',
]
