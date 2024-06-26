import os
import subprocess
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Union

import torch
import torchaudio
import webdataset as wds
from joblib import Parallel, delayed
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose
from tqdm import tqdm

from samantha.data.audio.types import AudioDataResult, DataResult
from samantha.dataio.batching import BucketBatcher
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    Pad,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)


def PaddingStrategy(padding_strategy: str, n_samples: int) -> Union[Pad, RandomPad]:
    if padding_strategy == "pad":
        return Pad(n_samples=n_samples)
    elif padding_strategy == "random_pad":
        return RandomPad(n_samples=n_samples)
    else:
        raise NotImplementedError("Choose between [pad, random_pad]")


def collate_batch(
    batch, pad_value: Union[int, float], eq_len: bool = False
) -> DataResult:
    class_keys = vars(batch[0]).keys()
    init_args = {k: [] for k in class_keys}
    new_class = batch[0].__class__(**init_args)

    for b in batch:
        for k in class_keys:
            new_class.__dict__[k].append(b.__dict__[k])

    input_length = []
    for k in class_keys:
        if type(new_class.__dict__[k][0]) is torch.Tensor:

            # TODO: refactor this
            if eq_len:
                max_length = max([item.shape[-1] for item in new_class.__dict__[k]])
                pad = Pad(max_length, value=pad_value)
                for idx in range(len(new_class.__dict__[k])):
                    audio = new_class.__dict__[k][idx]
                    padded_audio = pad(audio)
                    new_class.__dict__[k][idx] = padded_audio
                    input_length.append(audio.shape[-1])
            new_class.__dict__[k] = torch.stack(new_class.__dict__[k], dim=0)
    new_class.input_length = torch.tensor(input_length, dtype=torch.long)
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
    sample_rate: int, batch_size: int, buckets_sec: List[int], length_fn: Callable
):
    buckets_samples = list(map(lambda i: i * sample_rate, buckets_sec))
    return BucketBatcher(
        buckets=buckets_samples,
        dynamic_batch=False,
        batch_size=batch_size,
        length_fn=length_fn,
    )


def token_batcher(
    frame_rate: int, batch_size: int, buckets_sec: List[int], length_fn: Callable
):
    buckets_frames = list(map(lambda i: i * frame_rate, buckets_sec))
    return BucketBatcher(
        buckets=buckets_frames,
        dynamic_batch=False,
        batch_size=batch_size,
        length_fn=length_fn,
    )


class LightningDataModuleBase(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        batch_size: int,
        shuffle: bool,
        validation_dataset=[],
        test_dataset=[],
        predict_dataset=[],
        num_workers: int = 8,
        pin_memory: bool = True,
        validation_batch_size: Optional[int] = None,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        if validation_batch_size is None:
            self.validation_batch_size = batch_size
        else:
            self.validation_batch_size = validation_batch_size

    @abstractmethod
    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.validation_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.validation_batch_size,
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

    def transfer_batch_to_device(
        self, batch: AudioDataResult, device: torch.device, dataloader_idx: int
    ) -> AudioDataResult:
        if type(batch.audio[0]) is torch.Tensor:
            batch.audio = batch.audio.to(device)
        return batch


class WebDataModuleBase(LightningDataModuleBase):
    def __init__(
        self,
        train_dataset,
        batch_size: int,
        shuffle: bool,
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
        # NOTE: datasets are deepcopy'd here, as we do in-place
        # operations in DataLoader() and these should be
        # done on each pointer independently.
        super().__init__(
            train_dataset=deepcopy(train_dataset),
            batch_size=batch_size,
            validation_dataset=deepcopy(validation_dataset),
            test_dataset=deepcopy(test_dataset),
            predict_dataset=deepcopy(predict_dataset),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batcher = batcher
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test
        self.epoch_size = epoch_size

        if self.batch_size_valid is None:
            self.batch_size_valid = self.batch_size
        if self.batch_size_test is None:
            self.batch_size_test = self.batch_size

    def collate_fn(self, batch):
        return collate_batch(batch)

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                # TODO: this can be more elegant
                for audio_attr in ["audio"]:
                    if batch[0].get(audio_attr) is not None:
                        max_length = max(
                            [self.batcher.length_fn(item) for item in batch]
                        )
                        pad = Pad(max_length, value=0.0)
                        for idx in range(len(batch)):
                            audio = batch[idx].get(audio_attr)
                            padded_audio = pad(audio)
                            batch[idx][audio_attr] = padded_audio
                            batch[idx]["input_length"] = audio.shape[-1]
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
            [ToTensor(), NormalizeAudioToFloat32(), SetAudioDimensions()]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
