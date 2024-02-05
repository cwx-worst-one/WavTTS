
import torch.distributed as dist
from torch.utils.data import IterableDataset

def distributed_subset(dataset, num_replicas=None, rank=None):
    if num_replicas is None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
    if rank is None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank()

    if isinstance(dataset, IterableDataset):
        print('Dataset provided is iterable dataset. Iterating through items first.')
        dataset = [batch for batch in iter(dataset)]
    total_size = len(dataset)         # true value without extra samples
    subset = dataset[rank:total_size:num_replicas]
    return subset