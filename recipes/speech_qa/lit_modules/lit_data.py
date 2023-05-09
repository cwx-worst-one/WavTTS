from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from webdataset import WebLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        predict_dataloader=None,
    ):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self._predict_dataloader = predict_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def predict_dataloader(self):
        return self._predict_dataloader


class WrappedDataModule(LightningDataModule):
    def __init__(
        self,
        dataset=None,
        batch_size=1,
        num_workers=8,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return dataloader


class WebDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return WebLoader(
            dataset=self.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
