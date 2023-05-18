import pytorch_lightning as pl
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset.pipeline import DataPipeline


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 100,
        train_dataset=None,
        validation_dataset=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.to_tuple("audio.npy"),
            wds.batched(self.batch_size),
        )
        return DataLoader(
            train_dataset_batched, batch_size=None, num_workers=self.num_workers
        )

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.to_tuple("audio.npy"),
            wds.batched(self.batch_size),
        )
        return DataLoader(
            validation_dataset_batched, batch_size=None, num_workers=self.num_workers
        )
