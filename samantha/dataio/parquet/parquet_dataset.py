import copy
import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from webdataset import filters, shardlists, warn_and_continue
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline

from samantha.dataio.parquet.shardlists import ResampledShards, SimpleShardList
from samantha.dataio.utils import parquet_reader, parse_data_urls_2

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
        self._pq_sample = _ParquetSample(None, handler)
        self.append(self._pq_sample)
        self._url_fetched = False

    def __iter__(self):
        if not self._url_fetched:
            fs, urls = parse_data_urls_2(data_id=self.data_id, data_urls=self.data_urls)
            self.stage(0).urls = urls
            self._pq_sample.filesystem = fs
        return super().__iter__()


class _ParquetSample:
    def __init__(
        self, filesystem=None, handler: Callable[[Exception], bool] = warn_and_continue
    ):
        self.filesystem = filesystem
        self.handler = handler

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        handler = self.handler
        filesystem = self.filesystem
        for src in sources:
            src_url = {f"__{k}_url__": v for k, v in src.items()}
            try:
                common_utt = set(
                    e["uttid"]
                    for v in src.values()
                    for _, e in parquet_reader(v, columns=["uttid"], fs=filesystem)
                )

                ordered_utt = []
                for _, e in parquet_reader(
                    src["index"], columns=["uttid"], fs=filesystem
                ):
                    if e["uttid"] in common_utt:
                        ordered_utt.append(e["uttid"])

                readers = {
                    name: parquet_reader(url, fs=filesystem)
                    for name, url in src.items()
                }
                for utt in ordered_utt:
                    try:
                        sample = copy.deepcopy(src_url)
                        sample.update({"__key__": utt, "uttid": utt})
                        for name, reader in readers.items():
                            _, cur_sample = next(reader)
                            while cur_sample["uttid"] != utt:
                                cur_sample = next(reader)
                            if name == "data":
                                cur_sample = {"wav": cur_sample["audio"]}
                            elif name == "index":
                                cur_sample.pop("row_group_no", None)
                                cur_sample.pop("data_file", None)
                            sample.update(cur_sample)
                        yield sample
                    except Exception as exn:
                        if hasattr(exn, "args") and len(exn.args) > 0:
                            exn.args = (
                                exn.args[0] + " @ " + json.dumps(src_url) + "&" + utt,
                            ) + exn.args[1:]
                        if handler(exn):
                            continue
                        else:
                            break
            except Exception as exn:  # pragma: no cover
                exn.args = exn.args + (json.dumps(src_url),)
                if handler(exn):
                    continue
                else:
                    break
