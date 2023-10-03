import os
import subprocess
from typing import List, Optional
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from joblib import Parallel, delayed
from julius.resample import ResampleFrac
from torch.utils.data import DataLoader
from tqdm import tqdm
from webdataset.pipeline import DataPipeline
import webdataset as wds
from samantha.dataio.batching import BucketBatcher


def collate_batch(batch):
    keys = batch[0].keys()
    batch_dict = defaultdict(list)
    for b in batch:
        for k in keys:
            batch_dict[k].append(b[k])
    return dict(batch_dict)

def _load_waveform(path: str, exp_sample_rate: int):
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(
            f"sample rate should be {exp_sample_rate}, but got {sample_rate}"
        )
    return waveform, sample_rate


def resample_cmd(
    audio_fp: str, out_fp: str, sample_rate: int, mono: bool, overwrite: bool
):
    n_channels = 1 if mono else 2
    overwrite = "-y" if overwrite else "-n"
    return [
        "ffmpeg",
        overwrite,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_fp,
        "-ac",
        str(n_channels),
        "-ar",
        str(sample_rate),
        out_fp,
    ]


def resample(audio_fp: str, sample_rate: int, mono: bool, overwrite: bool = False):
    out_fp = audio_fp + f"-resampled_{sample_rate}hz.wav"
    if not overwrite and os.path.exists(out_fp):
        return out_fp

    if os.path.exists(audio_fp):
        p = subprocess.Popen(
            resample_cmd(audio_fp, out_fp, sample_rate, mono, overwrite=overwrite)
        )
        p.wait()
    return out_fp


def parallel_resample(
    audio_fps: List[str],
    sample_rate: int,
    mono: bool,
    n_jobs: int = 12,
    overwrite: bool = False,
) -> None:
    def resample_catch_error(audio_fp):
        try:
            resample(audio_fp, sample_rate, mono, overwrite=overwrite)
        except Exception as e:
            print(e)

    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(resample_catch_error)(fp) for fp in tqdm(audio_fps)
    )


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
        self.resample_fn = ResampleFrac(self.data_sample_rate, sample_rate)

    def resample(self, audio: torch.Tensor) -> torch.Tensor:
        if self.data_sample_rate != self.sample_rate:
            return self.resample_fn(audio)
        return audio

    @property
    def _pytorch_dataloader_batch_size(self) -> int:
        return self.batch_size if self.batcher is None else None

    def collate_fn(self):
        return None

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=self._pytorch_dataloader_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self._pytorch_dataloader_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self._pytorch_dataloader_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
