import logging
from typing import Any, Callable, Dict, List, Optional

from webdataset import filters, shardlists, warn_and_continue
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline

from samantha.dataio.parquet.extension import _ParquetSample
from samantha.dataio.parquet.shardlists import ResampledShards, SimpleShardList
from samantha.dataio.utils import resolve_data_urls

logger = logging.getLogger(__name__)


class ParquetDataset(DataPipeline, FluidInterface):
    def __init__(
        self,
        data_id: int = None,
        data_urls: List[Dict[str, str]] = None,
        handler: Callable[[Exception], bool] = warn_and_continue,
        resampled: bool = False,
        shardshuffle: Optional[Any] = None,
        detshuffle: bool = False,
        nodesplitter=shardlists.single_node_only,
        sample_limit_per_file: int = None,
        **kwargs,
    ):
        super().__init__()
        self.data_id = data_id
        self.data_urls = data_urls
        # To fix conflicts between filesystem and multiprocessing, use empty url.
        # Real url list will be globed and assigned in __iter__()
        urls = []

        if resampled:
            self.append(ResampledShards(urls))
        else:
            self.append(SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))

        # To fix conflicts between filesystem and multiprocessing, use empty filesystem.
        # Real filesystem will be assigned in __iter__()
        self._pq_sample = _ParquetSample(None, handler, sample_limit_per_file)
        self.append(self._pq_sample)
        self._url_fetched = False

    def __iter__(self):
        if not self._url_fetched:
            fs, urls = resolve_data_urls(data_id=self.data_id, data_urls=self.data_urls)
            self.stage(0).urls = urls
            self._pq_sample.filesystem = fs
            self._url_fetched = True
        return super().__iter__()
