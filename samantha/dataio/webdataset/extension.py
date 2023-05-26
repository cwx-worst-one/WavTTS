import json
import re
import tarfile
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Set,
    Tuple,
    Union,
)

from pyarrow.fs import FileSystem
from webdataset import filters, shardlists
from webdataset.compat import FluidInterface
from webdataset.filters import reraise_exception
from webdataset.pipeline import DataPipeline
from webdataset.tariterators import (
    base_plus_ext,
    meta_prefix,
    meta_suffix,
    valid_sample,
)

from samantha.utils.hdfs_helper import hopen


def group_by_keys(
    data: Iterable[Dict[str, Any]],
    keys: Callable[[str], Tuple[str, str]] = base_plus_ext,
    lcase: bool = True,
    suffixes: Optional[Set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """
    Similar to original group_by_keys, but ignoring duplicate key errors.
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if current_sample is None or prefix != current_sample["__key__"]:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffix in current_sample:
                # We don't always ensure that there's no duplicate key when creating the
                # dataset, so we'll just ignore duplicate key errors here.
                print(
                    f"WARN {fname}: duplicate file name in tar file {suffix} {current_sample.keys()}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
    if valid_sample(current_sample):
        yield current_sample


def resolve_url2index(url2index: Union[str, Dict[str, str]]) -> Dict[str, str]:
    if type(url2index) == str:  # Load mapping from file
        url2index_map = {}
        with hopen(url2index, "r") as f:
            for line in f:
                if type(line) == bytes:
                    line = line.decode("utf-8")
                ary = line.strip().split("\t")
                url2index_map[ary[0]] = ary[1]
        return url2index_map
    else:
        assert type(url2index) == dict
        return url2index


def parse_index(line: Union[str, bytes]) -> Dict[str, Any]:
    if type(line) == bytes:
        line = line.decode("utf-8")
    ary = line.strip().split("\t")
    index = {"prefix": ary[0], "found": False}
    # We'll ignore everything beyond the 2nd column
    if len(ary) > 1:
        index["data"] = json.loads(ary[1])
    return index


def indexed_tarfile_iterator(
    fileobj,
    index: str,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.
    Args:
        fileobj: the tar file stream.
        index: index file path.
        skip_meta: regexp for keys that are skipped entirely.
                   Defaults to r"__[^/]*__($|/)".
        handler: exception handler. Defaults to reraise_exception.
    Yields:
        a stream of samples.
    """
    with tarfile.open(fileobj=fileobj, mode="r:") as stream:
        with hopen(index, "r") as index_stream:
            index_iter = iter(index_stream)
            curr_index = None
            for tarinfo in stream:
                fname = tarinfo.name
                try:
                    if not tarinfo.isreg():
                        continue
                    if fname is None:
                        continue
                    if (
                        "/" not in fname
                        and fname.startswith(meta_prefix)
                        and fname.endswith(meta_suffix)
                    ):
                        # skipping metadata for now
                        continue
                    if skip_meta is not None and re.match(skip_meta, fname):
                        continue
                    prefix, _ = base_plus_ext(fname)
                    if prefix is None:
                        continue
                    if curr_index is None or (
                        curr_index["found"] and curr_index["prefix"] != prefix
                    ):
                        curr_index = parse_index(next(index_iter))
                    if curr_index["prefix"] != prefix:
                        continue
                    if not curr_index["found"]:  # flush index data if there's any
                        if "data" in curr_index:
                            # WebDataset will not decode if the filename begins with "_"
                            yield dict(
                                fname=f"{prefix}.__index_data__",
                                data=curr_index["data"],
                            )
                        curr_index["found"] = True
                    data = stream.extractfile(tarinfo).read()
                    result = dict(fname=fname, data=data)
                    yield result
                    stream.members = []
                except StopIteration:  # this means the index stream is finished
                    break
                except Exception as exn:  # pragma: no cover
                    if hasattr(exn, "args") and len(exn.args) > 0:
                        exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
                    if handler(exn):
                        continue
                    else:
                        break
    # We need to close fileobj after close the stream,
    # otherwise, hdfs client will throw an error
    fileobj.close()


def indexed_tarfile_expander(
    data: Iterable[Dict[str, Any]],
    url2index: Dict[str, str],
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """Expand indexed tar files.
    Args:
        data: iterator over opened tar file streams.
        url2index: mapping from tar file path to index file path.
        handler: exception handler.
    Yields:
        a stream of samples.
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in indexed_tarfile_iterator(
                source["stream"],
                index=url2index[url],
                handler=handler,
            ):
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:  # pragma: no cover
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def url_opener_ra(data, handler=reraise_exception, **kw):
    """Open url as a random accessible stream."""
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            fs, path = FileSystem.from_uri(url)
            stream = fs.open_input_file(path)
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def indexed_tarfile_samples(
    src: Iterable[Dict[str, Any]],
    url2index: Dict[str, str],
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterable[Dict[str, Any]]:
    """Given a stream of indexed tar files, yield samples.
    Args:
        src: stream of tar files
        url2index: mapping from tar file path to index file path
        handler: exception handler
    Returns:
        stream of samples
    """
    streams = url_opener_ra(src, handler=handler)
    files = indexed_tarfile_expander(
        streams,
        url2index=url2index,
        handler=handler
    )
    samples = group_by_keys(files, handler=handler)
    return samples


class IndexedWebDataset(DataPipeline, FluidInterface):
    r"""
    Extension of WebDataset that supports sequential indexing. We assume that every
    tar has an associated tab-separated index file in the following format:
        <key1>  <metadata1>
        <key2>  <metadata2>
        ...
        <keyN>  <metadataN>
    <metadata> is an optional json struct that will be added to the parsed sample.

    When reading this indexed tar, we'll only read the keys contained in its index
    file; other keys in the tar file will be skipped. Requirements of index file:
        (1) The ordering of keys is the same as in the tar file
        (2) All keys must actually be present in the tar file
    If either (1) or (2) is not true, reading will not work!

    Concrete example: suppose the content of the tar file is:
        {"__key__": key1, "mp3": x1}
        {"__key__": key2, "mp3": x2}
        {"__key__": key3, "mp3": x3}
    and the content of the index file is:
        key1    {"m1": n1, "a1": b1}
        key3    {"m3": n3, "a3": b3}
    then the iterator on this dataset will yield:
        {"__key__": key1, "mp3": x1, "__index_data__": {"m1": n1, "a1": b1}}
        {"__key__": key3, "mp3": x3, "__index_data__": {"m3": n3, "a3": b3}}
    """

    def __init__(
        self,
        url2index: Union[str, Dict[str, str]],
        handler: Callable[[Exception], bool] = reraise_exception,
        resampled: bool = False,
        shardshuffle: Optional[Any] = None,
        detshuffle: bool = False,
        nodesplitter=shardlists.single_node_only,
    ):
        super().__init__()
        url2index = resolve_url2index(url2index)
        def maybe_remove_hdfs_cat(url):
            # Backward compatiblity, in old style we use hdfs -cat to
            # read webdataset from hdfs
            return url.replace("pipe:hdfs dfs -cat ", "")
        url2index = {maybe_remove_hdfs_cat(k): v for k, v in url2index.items()}
        urls = list(url2index.keys())
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
            filters.pipelinefilter(indexed_tarfile_samples)(
                url2index=url2index,
                handler=handler,
            )
        )
