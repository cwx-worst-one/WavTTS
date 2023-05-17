r"""Dataset examples for loading individual data points"""
import logging
from itertools import islice
from typing import List

import torch
from torch.utils.data import IterableDataset
from webdataset.utils import make_seed, pytorch_worker_info

logger = logging.getLogger(__name__)


class MultiIterableDataset(IterableDataset):
    r"""
    Combines multiple `IterableDataset`s by suppring sampling from multiple different
    datasets. Gets a batch of data (all samples from one dataset) based on a sampling
    probability per dataset.

    Args:
        datasets (List[IterableDataset]):
            A list of `IterableDataset`s to be sampled
        num_samples (int):
            The total number of samples to drawn.
        weights (List[float]):
            A list of sample probabilities. If None, all datasets will be sampled
            with equal probabilities. default=None
        seed (int):
            Random seed. default=0
        sample_chunk_size (int):
            The number of samples to be drawn at once. default=1024
    """

    def __init__(
        self,
        datasets: List[IterableDataset],
        num_samples: int,
        weights: List[float] = None,
        seed: int = 0,
        sample_chunk_size: int = 1024,
    ):
        super().__init__()
        self._datasets = datasets
        self._len = num_samples
        self._chunk_size = sample_chunk_size
        _weights = weights if weights else [1.0] * len(datasets)
        self._weights = torch.tensor(_weights, dtype=torch.float)
        assert len(self._weights) == len(self._datasets)
        self._seed = seed

    def _generate_idx_by_chunk(self, rng):
        """Generate indices by chunk"""
        for i in range(0, self._len, self._chunk_size):
            size = min(self._chunk_size, self._len - i)
            yield from torch.multinomial(
                self._weights, size, replacement=True, generator=rng
            ).tolist()

    def __iter__(self):
        sources = [iter(ds) for ds in self._datasets]
        # Get worker info
        rank, _, worker_id, num_workers = pytorch_worker_info()
        unique_id = rank * num_workers + worker_id
        # Create generator here to avoid pickle issues
        rng = torch.Generator().manual_seed(make_seed(self._seed, unique_id))

        # Use islice and worker id to split the work
        for c in islice(self._generate_idx_by_chunk(rng), worker_id, None, num_workers):
            try:
                yield next(sources[c])
            except StopIteration:
                sources[c] = iter(self._datasets[c])
                yield next(sources[c])

    def __len__(self):
        return self._len
