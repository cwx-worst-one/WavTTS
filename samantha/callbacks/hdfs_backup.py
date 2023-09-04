import logging
import os
from functools import cached_property
from multiprocessing.pool import ThreadPool
from typing import Any

import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import samantha.utils.hdfs_helper as hh

logger = logging.getLogger(__name__)


class HDFSBackup(Callback):
    def __init__(self, src_dir, hdfs_dir=None, interval=1000):
        super().__init__()
        self.src_dir = src_dir
        if hdfs_dir is None:
            hdfs_dir = os.path.join(os.getenv("ARNOLD_OUTPUT"), "HDFSBackup")
        self.dst_dir = hdfs_dir
        self.interval = interval
        self.worker = ThreadPool(1)
        self._files = {}
        if not self.should_backup:
            logger.warning(
                f"Destination directory {hdfs_dir} is not hdfs path, "
                f"ignoring hdfs backup."
            )
        logger.info(
            f"HDFSBackup Callback will sync all files in {src_dir=} to {hdfs_dir=}"
        )
        hh.mkdir(self.dst_dir)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.should_backup and trainer.global_step % self.interval == 0:
            self.backup_files()

    @cached_property
    def should_backup(self):
        return hh.ishdfs(self.dst_dir)

    @rank_zero_only
    def backup_files(self):
        for file in self.list_files():
            self.worker.apply_async(func=hh.sync_hdfs, args=(file, self.dst_dir))

    def list_files(self):
        changed_files = []
        for fn in os.listdir(self.src_dir):
            abs_fn = os.path.join(self.src_dir, fn)
            mtime = os.path.getmtime(abs_fn)
            if mtime == self._files.get(fn, None):
                continue
            self._files[fn] = mtime
            changed_files.append(abs_fn)
        return changed_files

    def teardown(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        self.worker.close()
        self.worker.join()
