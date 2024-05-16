import logging
import os
import random
import sys
import time

from torch.utils.data import IterableDataset
from webdataset import utils

logger = logging.getLogger(__name__)


class SimpleShardList(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, seed=None):
        """Iterate through the list of shards.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        self.urls = urls
        self.seed = seed

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = self.urls.copy()
        if self.seed is not None:
            random.Random(self.seed).shuffle(urls)
        for url in urls:
            yield url


class ResampledShards(IterableDataset):
    r"""Sample shards from the shard list.

    Args:
        urls(List): a list of dataset URLs
        nshards(int): how many urls will be sampled
        worker_seed(Callable): seed generation function for each worker
        deterministic(bool): deterministic or not
        replacement(bool): sample url with replacement or not
    """

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        replacement=False,
    ):
        super().__init__()
        self.urls = urls
        self.nshards = nshards
        self.worker_seed = (
            utils.pytorch_worker_seed if worker_seed is None else worker_seed
        )
        self.deterministic = deterministic
        self.epoch = -1
        self.replacement = replacement
        self._tik = time.perf_counter()

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if self.deterministic:
            seed = utils.make_seed(self.worker_seed(), self.epoch)
        else:
            seed = utils.make_seed(
                self.worker_seed(),
                self.epoch,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")
        self.rng = random.Random(seed)
        logger.info(f"resample data urls with mode {self.replacement=}")
        if self.replacement:
            for _ in range(self.nshards):
                index = self.rng.randint(0, len(self.urls) - 1)
                yield self.urls[index]
        else:
            rank, world_size, worker, num_workers = utils.pytorch_worker_info()
            url_length = len(self.urls)
            self._tik = time.perf_counter()
            loop = 0
            for cursor in range(self.nshards):
                index = cursor % url_length
                if index == 0:
                    loop = cursor // url_length
                    logger.info(f"{rank=} {worker=} #{loop} shuffle")
                    self.rng.shuffle(self.urls)
                if self._should_stamp():
                    progress = index / url_length
                    logger.info(f"{rank=} {worker=} {loop=} {progress=:.2%}")
                yield self.urls[index]

    def _should_stamp(self):
        if time.perf_counter() - self._tik >= 1800:
            self._tik = time.perf_counter()
            return True
        return False
