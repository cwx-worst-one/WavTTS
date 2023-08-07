from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import webdataset as wds
from julius.resample import ResampleFrac
from torch.utils.data import DataLoader
from webdataset.pipeline import DataPipeline

from samantha.dataio.batching import BucketBatcher


def _load_waveform(path: str, exp_sample_rate: int):
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(
            f"sample rate should be {exp_sample_rate}, but got {sample_rate}"
        )
    return waveform, sample_rate


class BaseDataModule(pl.LightningDataModule):
    data_sample_rate = None
    def __init__(
        self,
        sample_rate: int,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        batcher: Optional[BucketBatcher] = None,
        collate_fn: Optional[Callable] = None
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batcher = batcher
        self.collate_fn = collate_fn
        self.resample_fn = ResampleFrac(self.data_sample_rate, sample_rate)

    def resample(self, audio: torch.Tensor) -> torch.Tensor:
        if self.data_sample_rate != self.sample_rate:
            return self.resample_fn(audio)
        return audio

    @property
    def _pytorch_dataloader_batch_size(self) -> int:
        return self.batch_size if self.batcher is None else None

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
        )
        return DataLoader(train_dataset_batched, batch_size=self._pytorch_dataloader_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self._pytorch_dataloader_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self._pytorch_dataloader_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
