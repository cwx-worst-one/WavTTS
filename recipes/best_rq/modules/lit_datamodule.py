import torch
import pytorch_lightning as pl
import webdataset as wds
from webdataset.pipeline import DataPipeline
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_dataset=None,
        validation_dataset=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.num_workers)
