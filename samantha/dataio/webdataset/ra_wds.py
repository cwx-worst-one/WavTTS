import re
import tarfile

from webdataset import filters, shardlists, warn_and_continue
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline
from webdataset.tariterators import meta_prefix, meta_suffix

from samantha.dataio.utils import expand_urls
from samantha.dataio.webdataset.extension import group_by_keys, url_opener_ra


def tar_file_iterator(fileobj, skip_meta=r"__[^/]*__($|/)", handler=warn_and_continue):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    Args:
        fileobj: byte stream suitable for tarfile
        skip_meta: regexp for keys that are skipped entirely
            Default value = r"__[^/]*__($|/)"

    """
    stream = tarfile.open(fileobj=fileobj, mode="r:")
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
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream
    fileobj.close()


def tar_file_expander(data, handler=warn_and_continue):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator(source["stream"], handler=handler):
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def tarfile_samples(src, handler=warn_and_continue, skip_instance_cache=False):
    streams = url_opener_ra(
        src, handler=handler, skip_instance_cache=skip_instance_cache
    )
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys(files, handler=handler)
    return samples


class WebDataset(DataPipeline, FluidInterface):
    r"""WebDataset accelerated by random accessible stream

    Args:
        urls (Union[str, List[str]]): data path urls, could be patten or expandable
            expression like hdfs://dataset/*.tar or like hdfs://dataset/{000..010}.tar.
        handler (Callable): exception handler.
        resampled (bool): whether placement repetitively sample an url from urls.
        shardshuffle (bool): shuffle on shard (url/file) level or not.
        detshuffle (bool): determined shuffle or not, only works when shardshuffle on.
        nodesplitter (Callable): split urls into each node.
        skip_instance_cache (bool): whether disable fsspec instance cache, the instance
            cache may cause excessive memory usage in some situations, see
            https://filesystem-spec.readthedocs.io/en/latest/features.html#instance-caching # noqa
    """

    def __init__(
        self,
        urls,
        handler=warn_and_continue,
        resampled=False,
        shardshuffle=None,
        detshuffle=False,
        nodesplitter=shardlists.single_node_only,
        skip_instance_cache=False,
    ):
        super().__init__()

        def maybe_remove_hdfs_cat(url):
            # Backward compatiblity, in old style we use hdfs -cat to
            # read webdataset from hdfs
            return url.replace("pipe:", "").replace("hdfs dfs -cat ", "").strip()

        urls = expand_urls(urls)
        urls = [maybe_remove_hdfs_cat(url) for url in urls]
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
            filters.pipelinefilter(tarfile_samples)(
                handler=handler, skip_instance_cache=skip_instance_cache
            )
        )
