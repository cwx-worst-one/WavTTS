import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from webdataset import filters, shardlists, warn_and_continue
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline

from samantha.dataio.utils import parquet_reader
from samantha.utils.hdfs_helper import hopen


def indexed_parquet_samples(
    sources: Iterable[Dict[str, Any]],
    data2idx: Dict[str, str],
    handler: Callable[[Exception], bool] = warn_and_continue,
) -> Iterable[Dict[str, Any]]:
    for src in sources:
        url = src["url"]
        idx = data2idx[url]
        try:
            data_reader = parquet_reader(url)
            idx_reader = parquet_reader(idx)
            for data, idx in zip(data_reader, idx_reader):
                try:
                    _, data = data  # drop the group no
                    _, idx = idx  # drop the group no
                    assert idx["uttid"] == data["uttid"]
                    if "meta" in idx:
                        idx["meta"] = json.loads(idx["meta"])
                    sample = data
                    sample["__url__"] = url
                    sample["__key__"] = idx["uttid"]
                    sample["__index_data__"] = idx
                    yield sample
                except Exception as exn:  # pragma: no cover
                    if hasattr(exn, "args") and len(exn.args) > 0:
                        exn.args = (exn.args[0] + " @ " + url + "&" + idx,) + exn.args[
                            1:
                        ]
                    if handler(exn):
                        continue
                    else:
                        break
        except Exception as exn:  # pragma: no cover
            exn.args = exn.args + (src.get("url"))
            if handler(exn):
                continue
            else:
                break


def resolve_data2idx(data2idx: Union[Dict[str, str], List[Tuple[str, str]], str]):
    if isinstance(data2idx, list):
        urls = [x[0] for x in data2idx]
        data2idx = dict(data2idx)
    elif isinstance(data2idx, dict):
        urls = list(data2idx.keys())
    elif isinstance(data2idx, str):
        data2idx = {}
        urls = []
        with hopen(data2idx, "r") as f:
            for line in f:
                if type(line) == bytes:
                    line = line.decode("utf-8")
                ary = line.strip().split("\t")
                data2idx[ary[0]] = ary[1]
                urls.append(ary[0])
    else:
        raise ValueError("data2idx must be a dict or a list of tuples or a string")
    return data2idx, urls


class IndexedParquetDataset(DataPipeline, FluidInterface):
    def __init__(
        self,
        data2idx: Union[Dict[str, str], List[Tuple[str, str]], str],
        handler: Callable[[Exception], bool] = warn_and_continue,
        resampled: bool = False,
        shardshuffle: Optional[Any] = None,
        detshuffle: bool = False,
        nodesplitter=shardlists.single_node_only,
        **kwargs,
    ):
        super().__init__()
        data2idx, urls = resolve_data2idx(data2idx)
        self.data2idx = data2idx
        if resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))
        self.append(
            filters.pipelinefilter(indexed_parquet_samples)(
                data2idx=data2idx, handler=handler
            )
        )
