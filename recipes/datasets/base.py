import os
import subprocess
from typing import Any, Callable, List, Optional, Union, Iterable
from collections import defaultdict
from abc import abstractmethod
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds
from samantha.dataio.batching import BucketBatcher
from dataclasses import dataclass, fields
from torchaudio_augmentations import Compose
from samantha.dataio.batching import BucketBatcher
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)


def collate_batch(batch):
    keys = batch[0].keys()
    batch_dict = defaultdict(list)
    for b in batch:
        for k in keys:
            batch_dict[k].append(b[k])
    return dict(batch_dict)


BatchedStr = Union[str, List[str]]


@dataclass
class DataResult:
    shard: BatchedStr
    key: BatchedStr

    def get(self, name: str, default_val: Any = None):
        if not hasattr(self, name):
            return default_val
        return getattr(self, name)

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setitem__(self, name: str, value):
        setattr(self, name, value)

    def values(self):
        for field in fields(self):
            yield getattr(self, field.name)

def collate_batch_dataclass(batch) -> DataResult:
    class_keys = vars(batch[0]).keys()
    init_args = {k: [] for k in class_keys}
    new_class = batch[0].__class__(**init_args)

    for b in batch:
        for k in class_keys:
            new_class.__dict__[k].append(b.__dict__[k])

    for k in class_keys:
        if type(new_class.__dict__[k][0]) == torch.Tensor:
            try:
                new_class.__dict__[k] = torch.stack(new_class.__dict__[k], dim=0)
            except Exception as e:
                print(e, k)
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


def audio_batcher(
    sample_rate: int,
    batch_size: int,
    buckets_sec: List[int],
    length_fn: Callable,
):
    buckets_samples = list(map(lambda i: i * sample_rate, buckets_sec))
    return BucketBatcher(
        buckets=buckets_samples,
        dynamic_batch=False,
        batch_size=batch_size,
        length_fn=length_fn,
    )


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
        return collate_batch(batch)

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
        epoch_size: Optional[int] = None,
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
        self.epoch_size = epoch_size

        if self.batch_size_valid is None:
            self.batch_size_valid = self.batch_size
        if self.batch_size_test is None:
            self.batch_size_test = self.batch_size

    def collate_fn(self, batch):
        return collate_batch_dataclass(batch)

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                # TODO: this can be more elegant
                for audio_attr in ["input_audio", "target_audio"]:
                    if (
                        hasattr(batch[0], audio_attr)
                        and batch[0][audio_attr] is not None
                    ):
                        max_length = max(
                            [self.batcher.length_fn(item) for item in batch]
                        )
                        random_pad = RandomPad(max_length)
                        for idx in range(len(batch)):
                            batch[idx][audio_attr] = random_pad(batch[idx][audio_attr])
                yield batch

    def DataLoader(
        self,
        dataset: wds.DataPipeline,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        epoch_size: Optional[int] = None,
    ):
        if shuffle:
            dataset = dataset.compose(wds.shuffle(self.shuffle_buffer_size))

        if self.batcher is not None:
            dataset = dataset.compose(self.bucketize)

        if self.batcher is None:
            dataset = dataset.compose(
                wds.batched(batch_size, collation_fn=self.collate_fn, partial=False)
            )
            collate_fn = None
        else:
            collate_fn = self.collate_fn

        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        return DataLoader(
            dataset, batch_size=None, num_workers=num_workers, collate_fn=collate_fn
        )

    def train_dataloader(self):
        return self.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            epoch_size=self.epoch_size,
        )

    def val_dataloader(self):
        return self.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size_valid,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return self.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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
