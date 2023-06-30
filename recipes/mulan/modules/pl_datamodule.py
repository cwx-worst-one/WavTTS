import pytorch_lightning as pl
import webdataset as wds
from torch.utils.data import DataLoader, DistributedSampler

import recipes.mulan.dataset.utils as utils


class MuLanDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size=72,
        val_batch_size=128,
        sample_buffer_size=600,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.sample_buffer_size = sample_buffer_size

    def train_dataloader(self):
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.sample_buffer_size),
            wds.batched(self.batch_size, collation_fn=utils.collate_fn),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=4)

    def val_dataloader(self):
        val_loaders = [
            DataLoader(
                val,
                sampler=DistributedSampler(val, shuffle=False),
                batch_size=self.val_batch_size,
                num_workers=4,
            )
            for val in self.val_dataset
        ]
        return val_loaders


class MuLanMCCDataModule(pl.LightningDataModule):
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
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.to_tuple("audio"),
            wds.batched(self.batch_size),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        val_loaders = [
            DataLoader(
                val,
                sampler=DistributedSampler(val, shuffle=False),
                batch_size=self.val_batch_size,
                num_workers=4,
            )
            for val in self.val_dataset
        ]
        return val_loaders


class MuLanYMVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size=72,
        val_batch_size=128,
        shuffle_buffer_size=600,
        num_workers=4,
        pin_memory = True
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.sample_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.sample_buffer_size),
            wds.batched(self.batch_size, collation_fn=utils.collate_fn),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=4, collate_fn=lambda x:x)

    def val_dataloader(self):
        val_loaders = [
            DataLoader(
                val,
                sampler=DistributedSampler(val, shuffle=False),
                batch_size=self.val_batch_size,
                num_workers=4,
            )
            for val in self.val_dataset
        ]
        return val_loaders


class MuLanMMEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size=72,
        val_batch_size=128,
        shuffle_buffer_size=600,
        num_workers=4,
        pin_memory = True
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.sample_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.sample_buffer_size),
            wds.batched(self.batch_size, collation_fn=utils.collate_fn),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=4, collate_fn=lambda x:x)

    def val_dataloader(self):
        val_loaders = [
            DataLoader(
                val,
                sampler=DistributedSampler(val, shuffle=False),
                batch_size=self.val_batch_size,
                num_workers=4,
            )
            for val in self.val_dataset
        ]
        return val_loaders
