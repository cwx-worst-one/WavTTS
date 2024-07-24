import torch.nn as nn
from recipes.research.dataset.collection import FrankSinatraParquetDataset, RadioheadParquetDataset
from samantha.data.audio.dataset import AudioFolderDataModule
from pytorch_lightning import Trainer
from recipes.research.dataset.data_monitor import DataMonitor

from samantha.models.base import LightningModuleBase

class DummyModule(LightningModuleBase):

    def __init__(self):
        super().__init__()

        self.model = nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return None

if __name__ == "__main__":
    sample_rate = 44100
    segment_duration = 30
    num_workers = 4

    frank_sinatra = FrankSinatraParquetDataset(
        sample_rate=sample_rate,
        channels=2,
        segment_duration=segment_duration,
        resampled=False,
        shardshuffle=False,
        crop_from_start=False,
    )

    radiohead = RadioheadParquetDataset(
        sample_rate=sample_rate,
        channels=2,
        segment_duration=segment_duration,
        resampled=False,
        shardshuffle=False,
        crop_from_start=False,
    )

    pl_datamodule = AudioFolderDataModule(
        [frank_sinatra, radiohead],
        [],
        [],
        batch_size=1,
        shuffle=None,
        num_workers=num_workers,
        weights=None,
        batch_drop_duplicates=True,
    )

    pl_module = DummyModule()

    data_monitor = DataMonitor()

    trainer = Trainer(
        accelerator="cpu",
        devices=8,
        num_nodes=1,
        callbacks=[data_monitor],
        max_epochs=1,
    )
    trainer.fit(pl_module, pl_datamodule)
