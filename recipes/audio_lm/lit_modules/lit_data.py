import webdataset as wds
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import recipes.audio_lm.datasets.utils as utils


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
        train_dataset,
        val_dataset,
        batch_size=1,
        val_batch_size=1,
        sample_buffer_size=600,
        num_workers=8,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.sample_buffer_size = sample_buffer_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        val_loaders = [
            DataLoader(
                wds.DataPipeline(
                    val,
                    wds.shuffle(self.sample_buffer_size),
                    wds.batched(self.val_batch_size, collation_fn=utils.collate_fn),
                ),
                batch_size=None,
                num_workers=self.num_workers,
            )
            for val in self.val_dataset
        ]
        return val_loaders


class WDSDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size=1,
        val_batch_size=1,
        sample_buffer_size=600,
        num_workers=8,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.sample_buffer_size = sample_buffer_size
        self.num_workers = num_workers

    def train_dataloader(self):
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.sample_buffer_size),
            wds.batched(self.batch_size, collation_fn=utils.collate_fn),
        )
        return DataLoader(
            train_dataset_batched, batch_size=None, num_workers=self.num_workers
        )

    def val_dataloader(self):
        val_loaders = [
            DataLoader(
                wds.DataPipeline(
                    val,
                    wds.shuffle(self.sample_buffer_size),
                    wds.batched(self.val_batch_size, collation_fn=utils.collate_fn),
                ),
                batch_size=None,
                num_workers=self.num_workers,
            )
            for val in self.val_dataset
        ]
        return val_loaders


class InferenceModule(LightningDataModule):
    def __init__(self, train_dataset, batch_size=72, sample_buffer_size=600):
        super().__init__()
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.sample_buffer_size = sample_buffer_size

    def train_dataloader(self):
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.sample_buffer_size),
            wds.batched(self.batch_size, collation_fn=utils.collate_fn),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=4)
