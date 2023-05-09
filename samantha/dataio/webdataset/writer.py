import logging
import time
from typing import Any, Callable, Optional  # noqa

import webdataset as wds
from pyarrow.fs import FileSystem

from samantha.utils.hdfs_helper import ishdfs

logger = logging.getLogger(__name__)


class ShardWriter:
    r"""Like webdataset.ShardWriter but support HDFS too.

    Create a Shard Writer.

    Args:
        pattern (str): output file pattern, either be local pattern or
            remote hdfs pattern. eg.: ``hdfs://haruna/<path>/%05d.tar``.
        maxcount (int): maximum number of records per shard.
        maxsize (float): maximum size of each shard.
        post (Callable): post process function to each shard file.
        start_shard (int): start number of shard.
        **kw: other options passed to :class:`TarWriter <wds.TarWriter>`.
    """

    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Callable = None,
        start_shard: int = 0,
        verbose=True,
        **kw,
    ):
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.stream = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self._tik = time.perf_counter()
        self._total_size = 0
        if ishdfs(pattern):
            self.stream = FileSystem.from_uri(pattern)[0]
        self.next_stream()

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.pattern % self.shard
        self.shard += 1
        if self.stream is None:
            stream = open(self.fname, "wb")
        else:
            stream = self.stream.open_output_stream(self.fname)
        self.tarstream = wds.TarWriter(stream, **self.kw)
        self.count = 0
        self.size = 0

    def write(self, obj: Any):
        r"""Write a sample.

        Args:
            obj (Any): sample to be written
        """

        if (
            self.tarstream is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self):
        """Finish all writing (use close instead)."""

        if self.verbose and self.fname:
            elapsed = time.perf_counter() - self._tik
            self._total_size += self.size
            tp = (self._total_size / (1 << 30)) / elapsed
            size = self.size / (1 << 30)
            logger.info(
                f"{self.fname}|{size:4.1f}GB|{self.count / 1024:.0f}k|"
                f"{self.total / 1024:.0f}k|{elapsed / 3600:.4f}h|{tp:.4f}GB/s|"
            )

        if self.tarstream is not None:
            self.tarstream.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.tarstream = None

    def close(self):
        """Close the stream."""
        self.finish()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()
