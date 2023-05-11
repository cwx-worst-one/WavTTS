import random

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from recipes.byteformers_example.tokenizer import LlamaTokenizer


class TokenDataset(Dataset):
    def __init__(self, fp: str, seq_len: int):
        self.data = np.memmap(fp, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.total = len(self.data)

    @staticmethod
    def prepare(text: str, tokenizer, out_fp: str):
        ids = np.array(tokenizer.encode(text), dtype=np.uint16)
        ids.tofile(out_fp)

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        idx = random.randint(0, self.total - self.seq_len - 1)
        x = torch.from_numpy(self.data[idx : idx + self.seq_len + 1].astype(np.int64))
        y = x[1:]
        x = x[:-1]
        return x, y


class TokenDataModule(LightningDataModule):
    def __init__(
        self,
        data_fp: str,
        tokenizer: LlamaTokenizer,
        seq_len: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.data_fp = data_fp
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        with open(self.data_fp, "r") as f:
            text = f.read()

        n = len(text)
        train_data = text[: int(n * 0.9)]
        val_data = text[int(n * 0.9) :]

        # Train our own tokenizer:
        # vocab_size = 100
        # LlamaTokenizer.train(input_file_path, ".", vocab_size=vocab_size)
        # tokenizer = LlamaTokenizer("tokenizer.model")

        # Or load a pre-trained tokenizer (e.g., LLaMA's)
        TokenDataset.prepare(train_data, self.tokenizer, "train.bin")
        TokenDataset.prepare(val_data, self.tokenizer, "valid.bin")
        self.train_dataset = TokenDataset("train.bin", seq_len=self.seq_len)
        self.valid_dataset = TokenDataset("valid.bin", seq_len=self.seq_len)

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
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
