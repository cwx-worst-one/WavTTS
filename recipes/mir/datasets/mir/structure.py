import os
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np
import torch
import webdataset as wds

from recipes.mir.datasets.mir import MIRDataModuleBase
from recipes.mir.datasets.mir.base import BaseAudioTransform, WebDataModuleBase
from samantha.transforms.audio import RandomPad, RandomResizedCrop
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.utils.logger import RankedLogger

logger = RankedLogger()

from sam.data.mir import MIRDataModuleBase

from recipes.mir.eval.structure.utils import get_structure_training_labels


@dataclass
class StructureDataResult:
    audio: torch.Tensor
    boundary_label: torch.Tensor
    function_label: torch.Tensor
    shard: str
    key: str


class StructureDataModule(MIRDataModuleBase):
    _splits = ["train", "validation", "test"]

    _directories = {
        "train": [
            "structure/billboard_structure_24kHz/train",  # TODO
            # "harmonix_structure_24kHz/train",
            # "rwc_structure_24kHz/train",
            # "salami_structure_24kHz/train",
            # "hooktheory_structure_24kHz/train",
            # "isophonic_structure_24kHz/train",
            # "pop909_structure_24kHz/train",
        ],
        "validation": [
            "structure/harmonix_structure_24kHz/validation",
            # "rwc_structure_24kHz/validation",
            # "salami_structure_24kHz/validation",
        ],
        "test": [
            "structure/pop909_structure_24kHz/test",
        ],
    }
    _sample_rate = 24000

    def __init__(
        self,
        split: str,
        duration: float,
        label_hop: float,
        chunk_per_sample: int,
        resample: bool,
        enable_short: bool = True,
    ):
        super().__init__(
            split=split,
            resample=resample,
        )
        self._duration = duration
        self._label_hop = label_hop
        self._chunk_per_sample = chunk_per_sample
        self._enable_short = enable_short
        self._duration_samples = int(self._sample_rate * self._duration)

        self.audio_transform = BaseAudioTransform()
        self.random_pad = RandomPad(self._duration_samples)
        self.random_crop = RandomResizedCrop(self._duration_samples)

    @property
    def shard_urls(self):
        urls = []
        for d in self._directories[self.split]:
            urls.extend(hdfs_ls(os.path.join(self._root, d)))
        return urls

    def train_transform(self, items: Iterable[Any]) -> Iterable[StructureDataResult]:
        for item in items:
            audio = self.audio_transform(item["audio.npy"])
            original_audio_len = audio.shape[1]

            audio_duration_in_s = original_audio_len / self.sample_rate
            (
                boundary_label,
                function_label,
                chorus_only,
                chorus_interval,
                structure_intervals,
            ) = get_structure_training_labels(
                item, audio_duration_in_s, self._label_hop, self._enable_short
            )
            chunk_per_sample = min(1, int(audio_duration_in_s // 9))

            for _ in range(chunk_per_sample):
                if self.is_train:
                    audio_chunk = self.random_pad(audio)
                    audio_chunk, start_idx = self.random_crop(
                        audio_chunk, return_sample_idx=True
                    )
                    sample_start_time = start_idx / self.sample_rate
                else:
                    audio_chunk = audio
                    sample_start_time = 0

                label_len = int(self._duration / self._label_hop)
                sample_idx = int(round(sample_start_time * (1 / self._label_hop)))
                boundary_label = boundary_label[sample_idx : sample_idx + label_len, :]
                function_label = function_label[sample_idx : sample_idx + label_len, :]
                pad_len = label_len - boundary_label.shape[0]

                if pad_len > 0:
                    boundary_label = np.pad(
                        boundary_label, [(0, pad_len), (0, 0)], "constant"
                    )
                    function_label = np.pad(
                        function_label, [(0, pad_len), (0, 0)], "constant"
                    )

                boundary_label = boundary_label[..., :1]

                shard = item["__url__"]
                key = item["__key__"]
                yield StructureDataResult(
                    audio=audio_chunk,
                    boundary_label=torch.tensor(boundary_label),
                    function_label=torch.tensor(function_label),
                    shard=shard,
                    key=key,
                )

    def val_transform(self, items: Iterable[Any]) -> Iterable[StructureDataResult]:
        for item in items:
            item["intervals.pickle"].pop(-1)
            item["labels.pickle"].pop(-1)

            for i, interval in enumerate(item["intervals.pickle"]):
                start, end = float(interval[0]), float(interval[1])
                if end - start <= 0:
                    item["intervals.pickle"].pop(i)
                    item["labels.pickle"].pop(i)

            audio = self.audio_transform(item["audio.npy"])
            audio_duration_in_s = audio.shape[1] / self.sample_rate
            (
                boundary_label,
                function_label,
                chorus_only,
                chorus_interval,
                structure_intervals,
            ) = get_structure_training_labels(
                item, audio_duration_in_s, self._label_hop, self._enable_short
            )
            shard = item["__url__"]
            key = item["__key__"]
            yield StructureDataResult(
                audio=audio,
                boundary_label=torch.tensor(structure_intervals),
                function_label=torch.tensor(function_label),
                shard=shard,
                key=key,
            )

    def transform(self, items: Iterable[Any]) -> Iterable[StructureDataResult]:
        if self.is_train:
            return self.train_transform(items)
        return self.val_transform(items)


class CombinedStructureDataModule(WebDataModuleBase):
    def __init__(
        self,
        duration: int,
        label_hop: float,
        chunk_per_sample: int,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        train_dataset = StructureDataModule(
            split="train",
            duration=duration,
            label_hop=label_hop,
            chunk_per_sample=chunk_per_sample,
            resample=True,
        )
        validation_dataset = StructureDataModule(
            split="validation",
            duration=duration,
            label_hop=label_hop,
            chunk_per_sample=chunk_per_sample,
            resample=False,
        )
        test_dataset = StructureDataModule(
            split="test",
            duration=duration,
            label_hop=label_hop,
            chunk_per_sample=chunk_per_sample,
            resample=False,
        )

        predict_dataset = test_dataset
        super().__init__(
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size_valid=1,  # TODO
            batch_size_test=1,  # TODO
        )
