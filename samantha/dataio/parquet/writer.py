import logging
import os
import time

import pyarrow
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetWriter as _ParquetWriter

logger = logging.getLogger(__name__)

PLACEHOLDER = "__placeholder__"


class ParquetWriter:
    r"""A writer that writes data to a parquet file."""

    def __init__(
        self,
        filename,
        row_group_size=64,
        need_row_group_no=False,
        filesystem=None,
        verbose=False,
    ):
        self.filename = filename
        self.writer = None
        self.scheme = None
        self.total = 0
        self.fs = filesystem or get_filesystem(self.filename)
        self.verbose = verbose
        self.row_group_size = row_group_size
        self.tb = []
        self.need_row_group_no = need_row_group_no
        self.row_group_no = 0
        self.count = 0
        self._tik = time.perf_counter()

    def write(self, item):
        r"""Write an item to parquet."""
        if item is None:
            return
        item = self._try_insert_row_group_no(item)
        if self.writer is None:
            self.scheme = pyarrow.Table.from_pylist([item]).schema
            self.writer = _ParquetWriter(self.filename, self.scheme, filesystem=self.fs)
        item = self._try_insert_row_group_no(item)
        self.tb.append(item)
        if len(self.tb) == self.row_group_size:
            tb = pyarrow.Table.from_pylist(self.tb)
            self.writer.write_table(tb)
            self.total += self.row_group_size
            self.row_group_no += 1
            self.tb = []
            return self.total

    def _try_insert_row_group_no(self, item):
        if self.need_row_group_no:
            item.update({"row_group_no": self.row_group_no})
        return item

    def write_last(self):
        if self.writer is not None and self.tb:
            tb = pyarrow.Table.from_pylist(self.tb)
            self.writer.write_table(tb)
            self.count += len(self.tb)
            self.total += len(self.tb)
            self.tb = []
            return self.total

    def finish(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.verbose and self.filename and self.fs.exists(self.filename):
            elapsed = time.perf_counter() - self._tik
            size = self.fs.size(self.filename)
            tp = (size / (1 << 30)) / elapsed
            size = size / (1 << 30)
            logger.info(
                f"{self.filename}|grp_no={self.row_group_no}{size:4.1f}GB|{self.count / 1024:.0f}k|"  # noqa
                f"{self.total / 1024:.0f}k|{elapsed / 3600:.4f}h|{tp:.4f}GB/s|"
            )

    def close(self):
        r"""Close the stream."""
        self.write_last()
        self.finish()
        return self.total

    def __enter__(self):
        r"""Enter context."""
        return self

    def __exit__(self, *args, **kw):
        r"""Exit context."""
        self.close()


class ShardWriter:
    r"""A writer that writes data to multiple parquet files.

    It writes data to multiple parquet files, each of which contains
    no more than `maxcount` rows.

    Args:
        output_pattern (str): output pattern of parquet files, e.g. "xxx-%04d.parquet".
        maxcount (int): max number of rows in each parquet file.
        row_group_size (int): number of rows in each row group.
        need_row_group_no (bool): whether to add row_group_no to each row.
    """

    def __init__(
        self, output_pattern, maxcount=2048, row_group_size=64, need_row_group_no=False
    ):
        self.output_pattern = output_pattern
        self.maxcount = maxcount
        self.row_group_size = row_group_size
        self.need_row_group_no = need_row_group_no
        assert (
            maxcount % row_group_size == 0
        ), "maxcount must be a multiple of row_group_size"
        self.fs = get_filesystem(self.output_pattern)
        self.count = 0
        self.shard = 0
        self.writer = None
        self.filename = None
        self._tik = time.perf_counter()
        self.total = 0
        self.total_size = 0

    def write(self, item):
        r"""Write an item to shards."""
        if item is None:
            return
        if self.writer is None or self.count == self.maxcount:
            self._next_stream()
        self.writer.write(item)
        self.count += 1

    def close(self):
        r"""Close the stream."""
        self._finish()
        del self.writer
        del self.shard
        del self.count

    def _next_stream(self):
        self._finish()
        self.filename = self.output_pattern % self.shard
        self.writer = ParquetWriter(
            self.filename,
            row_group_size=self.row_group_size,
            need_row_group_no=self.need_row_group_no,
        )
        self.shard += 1
        self.count = 0

    def _finish(self):
        if self.writer is not None:
            self.total += self.writer.close()
            self.writer = None

        if self.filename:
            elapsed = time.perf_counter() - self._tik
            size = self.fs.size(self.filename)
            self.total_size += size
            tp = (self.total_size / (1 << 30)) / elapsed
            size = size / (1 << 30)
            logger.info(
                f"{self.filename}|{size:4.1f}GB|{self.count / 1024:.0f}k|"
                f"{self.total / 1024:.0f}k|{elapsed / 3600:.4f}h|{tp:.4f}GB/s|"
            )

    def __enter__(self):
        r"""Enter context."""
        return self

    def __exit__(self, *args, **kw):
        r"""Exit context."""
        self.close()


class IndexShardWriter:
    r"""A writer that writes data to multiple parquet files with index.

    It writes data to multiple parquet files, each of which contains
    no more than `maxcount` rows and a corresponding index file.

    Args:
        output_root (str): output root of parquet files.
        idx_version (int): version of index file.
        partitions (list): list of partition folders.
        filename_pattern (str): output pattern of parquet files,
            e.g. "shard-%04d.parquet".
        maxcount (int): max number of rows in each parquet file.
        row_group_size (int): number of rows in each row group.
        need_row_group_no (bool): whether to add row_group_no to each row.
    """

    def __init__(
        self,
        output_root,
        idx_version=1,
        partitions=["part=00000"],
        filename_pattern="shard-%05d.parquet",
        maxcount=2048,
        row_group_size=64,
    ):
        self.output_root = output_root
        self.partitions = partitions
        partition_path = "/".join(partitions)
        self.data_pattern = os.path.join(
            output_root, "data", partition_path, filename_pattern
        )
        idx_pattern = filename_pattern.replace(
            ".parquet", f".index_{idx_version}.parquet"
        )
        self.idx_pattern = os.path.join(
            output_root, f"index_{idx_version}", partition_path, idx_pattern
        )
        self.data_writer = ShardWriter(
            self.data_pattern, maxcount, row_group_size, need_row_group_no=False
        )
        self.idx_writer = ShardWriter(
            self.idx_pattern, maxcount, row_group_size, need_row_group_no=True
        )

    def write(self, data_item: dict, idx_item: dict):
        r"""Write an item to shards."""
        self.data_writer.write(data_item)
        cd_path = "/".join([".."] * (len(self.partitions) + 1))
        idx_item.update(
            {"data_file": self.data_writer.filename.replace(self.output_root, cd_path)}
        )
        self.idx_writer.write(idx_item)

    def close(self):
        r"""Close the stream."""
        self.data_writer.close()
        self.idx_writer.close()

    def __repr__(self):
        return (
            f"IndexShardWriter(data_pattern={self.data_pattern}, "
            f"idx_pattern={self.idx_pattern})"
        )
