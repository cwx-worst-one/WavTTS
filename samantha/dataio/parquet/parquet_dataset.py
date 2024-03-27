import logging
from typing import Any, Callable, Dict, List, Optional

from webdataset import filters, shardlists, warn_and_continue
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline

from samantha.dataio.parquet.extension import setup_sampler
from samantha.dataio.parquet.shardlists import ResampledShards, SimpleShardList
from samantha.dataio.utils import resolve_data_urls

logger = logging.getLogger(__name__)


class YieldState:
    def __init__(self):
        pass

    def __call__(self, item):
        # now yield None to adapt cruise lite dataloader
        if item is not None:
            return item, (None, None)


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
        extra_fields_in_data: Optional[List[str]] = None,
        sample_config: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_id = data_id
        self.data_urls = data_urls
        urls = resolve_data_urls(data_id=self.data_id, data_urls=self.data_urls)

        if resampled:
            self.append(
                ResampledShards(urls, replacement=kwargs.get("replacement", False))
            )
        else:
            self.append(SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle:
                shardshuffle = 100
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))

        self.append(
            setup_sampler(
                handler, sample_limit_per_file, extra_fields_in_data, sample_config
            )
        )

        sample_shuffle_buffer_size = kwargs.get("sample_shuffle_buffer_size", 0)
        if sample_shuffle_buffer_size > 1:
            self.append(filters.shuffle(sample_shuffle_buffer_size))

        item_transform = kwargs.get("item_transform")
        if item_transform is not None:
            self.map(item_transform)

        batch_size = kwargs.get("batch_size_in_worker")
        if batch_size is not None:
            self.append(filters.batched(batchsize=batch_size, collation_fn=None))

        yield_state = kwargs.get("yield_state", False)
        if yield_state:
            yield_state_func = YieldState()
            self.map(yield_state_func)

    def load_state_dict(self, dataset_state):
        pass
