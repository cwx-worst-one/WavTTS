import os
from abc import abstractmethod
from typing import Any, Iterable

import webdataset as wds

from samantha.dataio.data_bucket import data_bucket
from samantha.dataio.utils import expand_urls
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class MIRDataModuleBase:
    _splits = []
    _data_sample_rate = None

    def __init__(
        self,
        split: str,
        resampled: bool,
        shardshuffle: bool,
        use_pipe: bool = True,
        nodesplitter=wds.shardlists.single_node_only,
    ):
        super().__init__()

        if split not in self._splits:
            raise NotImplementedError(f"{split} does not exist for this dataset")

        self.split = split
        self.resampled = resampled
        self.shardshuffle = shardshuffle
        self.logger = logger

        urls = self.shard_urls
        if use_pipe:
            urls = self.pipe_shard_urls(self.shard_urls)

        self.dataset = (
            wds.WebDataset(
                urls,
                resampled=self.resampled,
                shardshuffle=self.shardshuffle,
                nodesplitter=nodesplitter,
            )
            .decode()
            .compose(self.transform)
        )
        self.dataset.urls = expand_urls(urls)

        self.logger.info(
            f"Collected {len(urls)} shard urls for data split: {self.split}"
        )

    @abstractmethod
    def transform(self, items: Iterable[Any]) -> Iterable[Any]:
        pass

    @property
    def _root(self) -> str:
        return data_bucket("data/music/mir_benchmark")

    @property
    def is_train(self):
        return self.split == "train"

    @property
    def data_sample_rate(self) -> int:
        return self._data_sample_rate

    @property
    def shard_urls(self):
        urls = []
        for d in self._directories[self.split]:
            urls.extend(hdfs_ls(os.path.join(self._root, d)))
        return urls

    @staticmethod
    def pipe_shard_urls(shard_urls):
        return [f"pipe: hdfs dfs -cat {url}" for url in shard_urls]
