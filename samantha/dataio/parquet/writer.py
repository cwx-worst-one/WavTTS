import logging
import time

import pyarrow
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetWriter

logger = logging.getLogger(__name__)


class ShardWriter:
    def __init__(
        self, output_pattern, maxcount, row_group_size=64, need_row_group_no=False
    ):
        self.output_pattern = output_pattern
        self.maxcount = maxcount
        self.count = 0
        self.shard = 0
        self.writer = None
        self.scheme = None
        self.filename = None
        self.total = 0
        self.fs = get_filesystem(self.output_pattern)
        self._tik = time.perf_counter()
        self.row_group_size = row_group_size
        self.tb = []
        self.need_row_group_no = need_row_group_no
        self.row_group_no = 0
        self.total_size = 0

    def try_insert_row_group_no(self, item):
        if self.need_row_group_no:
            item.update({"row_group_no": self.row_group_no})
        return item

    def write(self, item):
        if item is None:
            return
        item = self.try_insert_row_group_no(item)
        if self.writer is None or self.count >= self.maxcount:
            self.scheme = pyarrow.Table.from_pylist([item]).schema
            self.next_stream()
        item = self.try_insert_row_group_no(item)
        self.tb.append(item)
        if len(self.tb) == self.row_group_size:
            tb = pyarrow.Table.from_pylist(self.tb)
            self.writer.write_table(tb)
            self.count += self.row_group_size
            self.total += self.row_group_size
            self.row_group_no += 1
            self.tb = []
            return self.total

    def write_last(self):
        if self.writer is not None and self.tb:
            tb = pyarrow.Table.from_pylist(self.tb)
            self.writer.write_table(tb)
            self.count += len(self.tb)
            self.total += len(self.tb)
            self.tb = []
            return self.total

    def next_stream(self):
        self.finish()
        self.filename = self.output_pattern % self.shard
        self.writer = ParquetWriter(self.filename, self.scheme, filesystem=self.fs)
        self.count = 0
        self.tb = []
        self.shard += 1
        self.row_group_no = 0

    def finish(self):

        if self.writer is not None:
            self.writer.close()
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

    def close(self):
        """Close the stream."""
        self.write_last()
        self.finish()
        del self.writer
        del self.shard
        del self.count

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()
