from abc import abstractmethod, abstractproperty
from typing import Any, Iterable, List

import webdataset as wds
import logging
import os
import subprocess
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
import torchaudio
import webdataset as wds
from joblib import Parallel, delayed
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from torchaudio_augmentations import Compose
from tqdm import tqdm

from samantha.dataio.batching import BucketBatcher
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)

BatchedStr = Union[str, List[str]]

logger = logging.getLogger(__name__)


@dataclass
class DataResult:
    shard: BatchedStr
    key: BatchedStr


def collate_batch(batch) -> DataResult:
    class_keys = vars(batch[0]).keys()
    init_args = {k: [] for k in class_keys}
    new_class = batch[0].__class__(**init_args)

    for b in batch:
        for k in class_keys:
            new_class.__dict__[k].append(b.__dict__[k])

    for k in class_keys:
        if type(new_class.__dict__[k][0]) == torch.Tensor:
            new_class.__dict__[k] = torch.stack(new_class.__dict__[k], dim=0)
    return new_class


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


class MIRDataModuleBase:
    _root = ""
    _splits = []
    _sample_rate = None

    def __init__(self, split: str, resample: bool):
        super().__init__()

        if split not in self._splits:
            raise NotImplementedError(f"{split} does not exist for this dataset")

        self.split = split
        self.resample = resample
        self.logger = logger

        self.init_provider()

    @abstractmethod
    def transform(self, items: Iterable[Any]) -> Iterable[Any]:
        pass

    @property
    def is_train(self):
        return self.split == "train"

    def init_provider(self):
        urls = self.pipe_shard_urls(self.shard_urls)

        provider = wds.DataPipeline(
            self.shard_sampler(urls),
            wds.tarfile_to_samples(),
            wds.shuffle(100),  #  TODO!!
            wds.decode(),
            self.transform,
        )
        self.provider = provider
        self.logger.info(
            f"Collected {len(urls)} shard urls for data split: {self.split}"
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def shard_sampler(self):
        if self.resample:
            return wds.ResampledShards
        return wds.SimpleShardList

    @abstractproperty
    def shard_urls(self) -> List[str]:
        return []

    @staticmethod
    def pipe_shard_urls(shard_urls):
        return [f"pipe: hdfs dfs -cat {url}" for url in shard_urls]


class LightningDataModuleBase(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        batcher: Optional[BucketBatcher] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batcher = batcher

    @abstractmethod
    def prepare_data(self) -> None:
        pass

    def collate_fn(self, batch):
        return default_collate(batch)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


class WebDataModuleBase(LightningDataModuleBase):
    def __init__(
        self,
        train_dataset,
        batch_size: int,
        shuffle_buffer_size: int,
        validation_dataset=None,
        test_dataset=None,
        predict_dataset=None,
        num_workers: int = 8,
        pin_memory: bool = True,
        batcher: Optional[BucketBatcher] = None,
        batch_size_valid: Optional[int] = None,
        batch_size_test: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.train_dataset = train_dataset
        self.shuffle_buffer_size = shuffle_buffer_size
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batcher = batcher
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test

        if self.batch_size_valid is None:
            self.batch_size_valid = self.batch_size
        if self.batch_size_test is None:
            self.batch_size_test = self.batch_size

    def collate_fn(self, batch):
        return collate_batch(batch)

    def finalize_pipeline(self, dataset: wds.DataPipeline, batch_size: int):
        dataset.pipeline.extend(
            [
                wds.shuffle(self.shuffle_buffer_size),
                wds.batched(batch_size, collation_fn=self.collate_fn, partial=False),
            ]
        )
        return dataset

    @staticmethod
    def DataLoader(pipeline: wds.DataPipeline, num_workers: int):
        return DataLoader(pipeline, batch_size=None, num_workers=num_workers)

    def train_dataloader(self):
        provider = self.train_dataset.provider
        return self.DataLoader(
            self.finalize_pipeline(provider, self.batch_size),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        provider = self.validation_dataset.provider
        return self.DataLoader(
            self.finalize_pipeline(provider, self.batch_size_valid),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        provider = self.test_dataset.provider
        return self.DataLoader(
            self.finalize_pipeline(provider, self.batch_size_test),
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        provider = self.predict_dataset.provider
        return self.DataLoader(
            self.finalize_pipeline(provider, self.batch_size),
            num_workers=self.num_workers,
        )


class BaseAudioTransform:
    def __init__(self):
        self.transform = Compose(
            [
                ToTensor(),
                NormalizeAudioToFloat32(),
                SetAudioDimensions(),
            ]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
