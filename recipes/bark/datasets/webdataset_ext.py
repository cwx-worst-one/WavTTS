import random
import sys

from webdataset import tariterators
from webdataset.compat import FluidInterface
from webdataset.filters import reraise_exception
from webdataset.pipeline import DataPipeline
from webdataset.pytorch import IterableDataset
from webdataset.shardlists import expand_urls


class MultiResampledShards(IterableDataset):

    def __init__(self, index_file, nshards=sys.maxsize):
        super().__init__()
        with open(index_file, "r") as fp:
            self.urls = [f"pipe: hdfs dfs -cat {line.split(',')[1]}" for line in fp.readlines()]

        self.nshards = nshards
        self.epoch = -1
        self.rng = random.Random()

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        for _ in range(self.nshards):
            shard_index = self.rng.randint(0, len(self.urls) - 1)
            yield dict(url=self.urls[shard_index])


class MultiWebDataset(DataPipeline, FluidInterface):

    def __init__(self, index_file, handler=reraise_exception):
        super().__init__()
        self.append(MultiResampledShards(index_file))
        self.append(tariterators.tarfile_to_samples(handler=handler))
