from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ReverseDataset(Dataset):
    def __init__(self, n_datapoints: int, vocab_size: int, seq_len: int):
        super().__init__()
        self.n_datapoints = n_datapoints
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.data = torch.randint(
            self.vocab_size, size=(self.n_datapoints, self.seq_len)
        )

    def __len__(self):
        return self.n_datapoints

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels


class ReverseHierarchyDataset(ReverseDataset):
    def __init__(
        self, n_datapoints: int, vocab_size: int, n_hierarchies: int, seq_len: int
    ):
        super().__init__(n_datapoints, vocab_size, seq_len)
        self.n_hierarchies = n_hierarchies
        self.data = torch.randint(
            self.vocab_size, size=(self.n_datapoints, self.n_hierarchies, self.seq_len)
        )

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(1,))
        return inp_data, labels


class ReverseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_datapoints: int,
        vocab_size: int,
        n_hierarchies: int,
        seq_len: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        dataset = partial(
            ReverseHierarchyDataset,
            vocab_size=vocab_size,
            n_hierarchies=n_hierarchies,
            seq_len=seq_len,
        )

        train_size = int(n_datapoints * 0.8)
        valid_test_size = int(n_datapoints * 0.1)
        self.train_dataset = dataset(n_datapoints=train_size)
        self.validation_dataset = dataset(n_datapoints=valid_test_size)
        self.test_dataset = dataset(n_datapoints=valid_test_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
