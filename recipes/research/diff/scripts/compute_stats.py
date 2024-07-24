import os
from glob import glob
from typing import List

import torch
import torch.distributed
import torch.nn as nn
from pytorch_lightning import Trainer

from recipes.research.dataset.collection import ShutterStockFeatureParquetDataset
from recipes.research.dataset.data_monitor import DataMonitor
from recipes.research.diff.normalize import FeatureNormalizer
from samantha.data.audio.dataset import AudioFolderDataModule
from samantha.data.audio.types import AudioDataResult
from samantha.models.base import DefaultTrainingBaseModule, TrainingResultBase
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


class RunningStats:
    def __init__(self):
        self.n_items = 0
        self.m = None
        self.s = None

    def __repr__(self):
        return f"{self.mean=}, {self.std=}, {self.var=}, {self.n_items=}"

    @property
    def mean(self) -> torch.Tensor:
        return self.m if self.n_items else torch.tensor([0.0])

    @property
    def var(self) -> torch.Tensor:
        return self.s / (self.n_items - 1) if self.n_items else torch.tensor([0.0])

    @property
    def std(self) -> torch.Tensor:
        return self.var.sqrt()

    def update(self, x: torch.Tensor) -> None:
        assert x.ndim == 2
        if self.n_items == 0:
            self.m = torch.zeros_like(x)
            self.s = torch.zeros_like(x)

        self.n_items += 1
        prev_m = self.m.clone()
        self.m += (x - self.m) / self.n_items
        self.s += (x - prev_m) * (x - self.m)


class StatsModule(DefaultTrainingBaseModule):

    def __init__(self, out_fp: str):
        super().__init__()
        self.out_fp = out_fp
        self.model = torch.nn.Linear(1, 1)
        self.stats = RunningStats()
        self.normalize = nn.Identity()

    def step(self, batch: AudioDataResult, batch_idx: int, return_loss: bool):

        feature = batch.audio  # [D, T]
        # for idx in range(feature.shape[0]):
        #     if feature[idx].std() > 2.0:
        #         print(feature[idx].std(), batch.key[idx])
        #         torch.save(batch, f"{batch.key[idx]}.pt")

        feature = self.normalize(feature)

        for f in feature:
            self.stats.update(f)

        if batch_idx and batch_idx % 1000 == 0:
            logger.debug(self.stats)
            self.write()

        return TrainingResultBase(
            loss={"loss": torch.tensor([0.0], requires_grad=True)}
        )

    def write(self):
        stats_fp = f"{self.out_fp}_rank={self.global_rank}"
        FeatureNormalizer.write(
            stats_fp,
            self.stats.mean.mean(),
            self.stats.std.mean(),
            self.stats.var.mean(),
            self.stats.n_items,
        )

    def on_predict_end(self) -> None:
        self.write()


if __name__ == "__main__":
    devices = 8
    num_workers = 12
    limit_predict_batches = None
    batch_size = 1

    dataset = ShutterStockFeatureParquetDataset(
        segment_duration=60, resampled=False, shardshuffle=False, crop_from_start=True
    )

    datamodule = AudioFolderDataModule(
        [dataset],
        [],
        [],
        weights=None,
        batch_size=batch_size,
        shuffle=None,
        num_workers=num_workers,
        prefetch_factor=2,
        batch_drop_duplicates=True,
    )
    dataloader = datamodule.train_dataloader()

    stats_fp = f"recipes/research/dataset/stats/ShutterStockFeatureParquetDataset_AudioCodec_7c355ea_64l.stats.pt"

    monitor = DataMonitor()
    module = StatsModule(out_fp=stats_fp)
    # module.normalize = FeatureNormalizer("recipes/research/dataset/stats/ShutterStockFeatureParquetDataset_AudioCodec_f81b3fa_64l.stats.pt")

    trainer = Trainer(
        accelerator="cpu",
        devices=devices,
        num_nodes=int(os.getenv("ARNOLD_WORKER_NUM", 1)),
        strategy="ddp_find_unused_parameters_false",
        use_distributed_sampler=False,  # NOTE: use this?
        callbacks=[monitor],
        limit_predict_batches=limit_predict_batches,
        max_epochs=1,
    )

    # 1. Calculate feature statistics
    trainer.predict(module, dataloader)

    # 2. Merge stats
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    stat_fps = list(glob(f"{stats_fp}_rank=*"))
    FeatureNormalizer.merge(stat_fps, stats_fp)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 2. Re-run with obtained statistics to verify:
    module = StatsModule(out_fp=stats_fp)
    module.normalize = FeatureNormalizer(stats_fp)
    trainer.predict(module, dataloader)

    stats_dict = {
        "mean": module.stats.mean.mean(),
        "std": module.stats.std.mean(),
        "var": module.stats.var.mean(),
        "n_items": module.stats.n_items,
    }
    logger.info(stats_dict)
