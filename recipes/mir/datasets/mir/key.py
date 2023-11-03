import os
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from recipes.mir.eval.key_detection.utils import get_key_training_label, get_keymode_labels

from sam.data.base import BaseAudioTransform, WebDataModuleBase
from recipes.mir.datasets.mir.base import BaseAudioTransform, WebDataModuleBase
from samantha.utils.hdfs_helper import hdfs_ls
from recipes.mir.datasets.mir import MIRDataModuleBase
from samantha.transforms.audio import RandomPad, RandomResizedCrop
from samantha.utils.logger import RankedLogger

logger = RankedLogger()


@dataclass
class KeyDataResult:
    audio: torch.Tensor
    key_label: torch.Tensor
    shard: str
    key: str


class KeyDataModule(MIRDataModuleBase):
    _splits = ["train", "validation", "test"]

    _directories = {
        "train": [
            "key/billboard_key_24kHz/train",
            "key/isophonic_key_24kHz/train",
            "key/leadsheet_key_24kHz/train",
            "key/karaoke_key_24kHz/train",  # large dataset
            # "key/karaoke_key_vocal_24kHz/train", # Vocal key detection, different task
            # "key/MCC126_key_24kHz/train", # Vocal key detection, different task
        ],
        "validation": [
            "key/leadsheet_key_24kHz/validation",
            # "MCC126_key_24kHz/validation",
        ],
        "test": [
            "key/giantstep_key_24kHz/test",
            # "MCC32_key_24kHz/test",
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
    ):
        super().__init__(
            split=split,
            resample=resample,
        )
        self._duration = duration
        self._label_hop = label_hop
        self._chunk_per_sample = chunk_per_sample
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

    def transform(self, items: Iterable[Any]) -> Iterable[KeyDataResult]:
        # TODO validation preprocessing
        for item in items:
            audio = self.audio_transform(item["audio.npy"])
            original_audio_len = audio.shape[1]
            audio_duration_in_s = original_audio_len / self.sample_rate

            key_labels = get_keymode_labels(
                item["keys.pickle"],
                item["intervals.pickle"],
                audio_duration_in_s,
                self._label_hop,
            )

            for _ in range(self._chunk_per_sample):
                if self.is_train:
                    audio_chunk = self.random_pad(audio)
                    audio_chunk, start_idx = self.random_crop(
                        audio_chunk, return_sample_idx=True
                    )
                    sample_start_time = start_idx / self.sample_rate
                else:
                    audio_chunk = audio
                    sample_start_time = 0

                key_label = get_key_training_label(
                    key_labels=key_labels,
                    audio_duration_in_s=audio_duration_in_s,
                    sample_start_time_in_s=sample_start_time,
                    label_hop=self._label_hop,
                    sample_len_in_s=self._duration,
                    is_train=self.is_train,
                )

                shard = item["__url__"]
                key = item["__key__"]
                yield KeyDataResult(
                    audio=audio_chunk,
                    key_label=torch.tensor(key_label, dtype=torch.long),
                    shard=shard,
                    key=key,
                )


class CombinedKeyDataModule(WebDataModuleBase):
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
        train_dataset = KeyDataModule(
            split="train",
            duration=duration,
            label_hop=label_hop,
            chunk_per_sample=chunk_per_sample,
            resample=True,
        )
        validation_dataset = KeyDataModule(
            split="validation",
            duration=duration,
            label_hop=label_hop,
            chunk_per_sample=chunk_per_sample,
            resample=False,
        )
        test_dataset = KeyDataModule(
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
