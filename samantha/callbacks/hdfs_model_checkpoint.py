import logging
import os
import tempfile
from datetime import timedelta
from multiprocessing.pool import Pool
from typing import Any, Optional

import pytorch_lightning as pl
from bytedance import easycycle
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT

from samantha.utils.hdfs_tools import hdfs_mkdir, hdfs_put

logger = logging.getLogger(__name__)


class HDFSModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        hdfs_path: Optional[_PATH] = None,
        num_sync_process: int = 5,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
    ):
        if hdfs_path is None:
            hdfs_path = os.getenv("ARNOLD_OUTPUT", None)
        self.hdfs_path = os.path.join(hdfs_path, "checkpoints")

        if dirpath is None:
            dirpath = tempfile.mkdtemp()
            rank_zero_warn(
                f"save ckeckpoints to tempdir={dirpath} cause it's not provided, "
                f"but will still sync all checkpoints into hdfs: {self.hdfs_path}"
            )

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        # record last.ckpt modify time.
        # use dict to support last-v1.ckpt, last-v2.ckpt, etc.
        # when reusing the same local checkpoint directory.
        self._last_mtime = dict()
        self._ckpt_history = set()
        self._worker_pool = Pool(num_sync_process)

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        fs = get_filesystem(self.dirpath)
        if fs.protocol != "file":
            raise ValueError(
                f"Only support local path to save checkpoints, but got {self.dirpath}."
            )
        trainer.strategy.broadcast(self.dirpath)

        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)
            hdfs_mkdir(self.hdfs_path)
            self.register_model()

    def register_model(self):
        req = easycycle.RegisterRawModelReq()
        req.name = os.getenv("ModelName", "TestModel")
        req.model_type = os.getenv("ModelType", "Common")
        req.owner = os.getenv("ARNOLD_TRIAL_OWNER", "samantha")
        easycycle.register_raw_model(req)

    @rank_zero_only
    def check_and_sync_checkpoints(self):
        def maybe_sync_last(ckpt_path):
            fn = os.path.basename(ckpt_path)
            if "last" not in fn:
                return False

            mtime = os.path.getmtime(ckpt_path)
            if mtime == self._last_mtime.get(fn, None):
                return False
            self._last_mtime.update({fn: mtime})
            return True

        ckpts = self.list_checkpoints()
        for ckpt in ckpts:
            if maybe_sync_last(ckpt) or ckpt not in self._ckpt_history:
                fn = os.path.basename(ckpt)
                self._ckpt_history.add(ckpt)
                hdfs_path = os.path.join(self.hdfs_path, fn)
                self._worker_pool.apply_async(
                    func=self._sync_checkpoint,
                    args=(ckpt, hdfs_path, "last" in fn),
                    error_callback=lambda e: logger.warning(f"{e}"),
                    callback=lambda e: logger.info(f"{e}"),
                )

    @classmethod
    def _sync_checkpoint(cls, local_path, hdfs_path, force):
        hdfs_put(local_path, hdfs_path, force=force)
        logger.info(f"Synced {local_path=} to {hdfs_path=}.")
        if force:
            return
        req = easycycle.RegisterCkptsReq()
        req.raw_model_name = os.getenv("ModelName", "TestModel")
        req.creator = os.getenv("ARNOLD_TRIAL_OWNER", "samantha")
        req.train_dataset_id = os.getenv("DatasetID", "null")
        req.train_task_id = os.getenv("ARNOLD_TRIAL_ID", "null")
        req.train_task_type = "MERLIN" if os.getenv("MERLIN_JOB_ID", None) else "ARNOLD"

        req.checkpoints = [
            easycycle.Checkpoint(
                name=os.path.basename(hdfs_path), hdfs=hdfs_path, extra={}
            )
        ]
        easycycle.register_ckpts(req)

    def list_checkpoints(self):
        ckpt_path = self.dirpath
        if self._fs.exists(ckpt_path):
            return {os.path.normpath(p) for p in self._fs.ls(ckpt_path, detail=False)}
        return set()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.check_and_sync_checkpoints()

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.check_and_sync_checkpoints()

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.check_and_sync_checkpoints()

    def teardown(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        self._worker_pool.close()
        self._worker_pool.join()

    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
        if (
            self.save_top_k != 0
            and self._fs.isdir(dirpath)
            and len(self._fs.ls(dirpath)) > 0
        ):
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
