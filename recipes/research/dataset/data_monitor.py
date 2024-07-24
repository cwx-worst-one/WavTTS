import os
from typing import Optional, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from webdataset import WebDataset

from recipes.research.dataset.parquet_dataset import AudioParquetDataset
from samantha.data.audio.types import AudioDataResult
from samantha.dataio.dataset import MultiIterableDataset
from samantha.models.base import LightningModuleBase
from samantha.utils.hdfs_tools import hdfs_cp, hdfs_mkdir
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


class DataMonitor(Callback):

    def __init__(
        self,
        hdfs_log_dir: Optional[str] = None,
        local_log_dir: Optional[str] = None,
        sync_every_n_steps: int = 1000,
    ):

        if hdfs_log_dir is not None:
            self.hdfs_log_dir = os.path.join(hdfs_log_dir, "monitor")
            hdfs_mkdir(self.hdfs_log_dir)

        if local_log_dir is None:
            self.local_dir = "monitor"
        else:
            self.local_dir = local_log_dir
        os.makedirs(self.local_dir, exist_ok=True)

        self.sync_every_n_steps = sync_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModuleBase) -> None:
        train_dataset: Union[MultiIterableDataset, AudioParquetDataset, WebDataset] = (
            trainer.train_dataloader.dataset
        )

        global_rank = trainer.global_rank

        fn = f"data_urls_{global_rank}.csv"
        local_fp = os.path.join(self.local_dir, fn)
        with open(local_fp, "w") as f:

            if type(train_dataset) == MultiIterableDataset:
                for dataloader_idx, dataset in enumerate(train_dataset.datasets):
                    for url in dataset.urls:
                        f.write(f"{dataloader_idx};{url}\n")
            else:
                if hasattr(train_dataset, "urls"):
                    for url in train_dataset.urls:
                        f.write(f"0;{url}\n")

        if hasattr(self, "hdfs_log_dir"):
            hdfs_fp = f"{self.hdfs_log_dir}/{fn}"
            hdfs_cp(local_fp, hdfs_fp, override=True)
            logger.info(f"Synchronised data url files to {self.hdfs_log_dir}")

    def record(self, trainer: Trainer, batch: AudioDataResult, batch_idx: int):
        batch_size = len(batch.segment_info)
        epoch = trainer.current_epoch
        step = trainer.global_step
        local_rank = trainer.local_rank
        global_rank = trainer.global_rank
        node_rank = trainer.node_rank
        fp = os.path.join(self.local_dir, f"data_monitor_{global_rank}.csv")

        with open(fp, "a+") as f:
            for idx in range(batch_size):
                id = batch.key[idx]
                shard_url = batch.shard_info[idx].url
                data_type = batch.segment_info[idx].data_type
                row = [
                    epoch,
                    step,
                    batch_idx,
                    idx,
                    local_rank,
                    global_rank,
                    node_rank,
                    id,
                    shard_url,
                    data_type,
                ]

                try:
                    f.write(";".join(map(str, row)) + "\n")
                except Exception as e:
                    logger.error(e)

    def sync(self, trainer: Trainer):
        if hasattr(self, "hdfs_log_dir"):
            logger.info(f"Synchronising data monitor directory to {self.hdfs_log_dir}")

            fn = f"data_monitor_{trainer.global_rank}.csv"
            hdfs_fp = f"{self.hdfs_log_dir}/{fn}"

            try:
                hdfs_cp(os.path.join(self.local_dir, fn), hdfs_fp, override=True)
            except Exception as e:
                logger.error(e)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: AudioDataResult,
        batch_idx: int,
    ) -> None:
        self.record(trainer, batch, batch_idx)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.sync(trainer)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: AudioDataResult,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.record(trainer, batch, batch_idx)

        if batch_idx % self.sync_every_n_steps == 0:
            self.sync(trainer)
