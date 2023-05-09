import random
import sys

from webdataset import tariterators
from webdataset.compat import FluidInterface
from webdataset.filters import reraise_exception
from webdataset.pipeline import DataPipeline
from webdataset.pytorch import IterableDataset
from webdataset.shardlists import expand_urls


class MultiResampledShards(IterableDataset):
    """
    Sample sub shards according to specified probability.
    """

    def __init__(self, urls, probs, nshards=sys.maxsize):
        assert len(urls) == len(
            probs
        ), f"len(urls): {len(urls)} != len(probs): {len(probs)}"
        print(f"urls: {[len(u) for u in urls]}, probs: {probs}")
        super().__init__()
        self.urls = [expand_urls(x) for x in urls]
        for x in self.urls:
            assert isinstance(x[0], str)
        self.probs = []
        for prob in probs:
            if len(self.probs) == 0:
                self.probs.append(prob)
            else:
                self.probs.append(self.probs[-1] + prob)
        if self.probs[-1] != 1:
            print(f"WARNING: last prob is not 1 ({self.probs[-1]}), setting to 1")
            self.probs[-1] = 1
        self.nshards = nshards
        self.epoch = -1
        self.rng = random.Random()

    def get_set_index(self, num):
        idx = len(self.probs) - 1
        for i in range(len(self.probs)):
            if num <= self.probs[i]:
                idx = i
                break
        return idx

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        for _ in range(self.nshards):
            num = self.rng.random()
            set_index = self.get_set_index(num)
            # print(f"num: {num}, set_index: {set_index}")
            shard_index = self.rng.randint(0, len(self.urls[set_index]) - 1)
            yield dict(url=self.urls[set_index][shard_index])


class MultiWebDataset(DataPipeline, FluidInterface):
    """
    Wrapper to support multiple shards with sampling probs. Only support resampled mode.
    """

    def __init__(self, urls, probs, handler=reraise_exception):
        super().__init__()
        self.append(MultiResampledShards(urls, probs))
        self.append(tariterators.tarfile_to_samples(handler=handler))
