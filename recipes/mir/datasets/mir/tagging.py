import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from recipes.mir.datasets.mir import MIRDataModuleBase
from recipes.mir.datasets.mir.base import BaseAudioTransform, WebDataModuleBase
from samantha.transforms.audio import RandomPad, RandomResizedCrop
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.utils.logger import RankedLogger

logger = RankedLogger()

from dataclasses import dataclass

import numpy as np
import torch

from recipes.mir.datasets.mir.base import MIRDataModuleBase


@dataclass
class TaggingDataResult:
    audio: torch.Tensor
    tag: torch.Tensor
    shard: str
    key: str


class TaggingDataModule(MIRDataModuleBase):
    _splits = ["train", "validation", "test"]

    _directories = {
        "train": [
            "tagging/mtat_tagging_24kHz/train",
        ],
        "validation": [
            "tagging/mtat_tagging_24kHz/validation",
        ],
        "test": [
            "tagging/mtat_tagging_24kHz/test",
        ],
    }

    _sample_rate = 24000

    _tag_names = np.array(
        (
            "guitar",
            "classical",
            "slow",
            "techno",
            "strings",
            "drums",
            "electronic",
            "rock",
            "fast",
            "piano",
            "ambient",
            "beat",
            "violin",
            "vocal",
            "synth",
            "female",
            "indian",
            "opera",
            "male",
            "singing",
            "vocals",
            "no vocals",
            "harpsichord",
            "loud",
            "quiet",
            "flute",
            "woman",
            "male vocal",
            "no vocal",
            "pop",
            "soft",
            "sitar",
            "solo",
            "man",
            "classic",
            "choir",
            "voice",
            "new age",
            "dance",
            "male voice",
            "female vocal",
            "beats",
            "harp",
            "cello",
            "no voice",
            "weird",
            "country",
            "metal",
            "female voice",
            "choral",
        )
    )
    _tag_names.flags.writeable = False

    def __init__(self, split: str, duration: int, resample: bool):
        super().__init__(
            split=split,
            resample=resample,
        )

        self.duration = duration
        self.audio_transform = BaseAudioTransform()

        self.duration_samples = int(self._sample_rate * self.duration)
        self.audio_pad = RandomPad(self.duration_samples)
        self.audio_crop = RandomResizedCrop(self.duration_samples)

    @property
    def shard_urls(self):
        urls = []
        for d in self._directories[self.split]:
            urls.extend(hdfs_ls(os.path.join(self._root, d)))
        return urls

    @property
    def tag_names(self):
        return self._tag_names

    def tag_indices_to_names(self, tags_multi_one_hot: torch.Tensor):
        if tags_multi_one_hot.ndim == 1:
            tags_multi_one_hot = tags_multi_one_hot[None, :]

        tags = []
        for i in tags_multi_one_hot:
            tag_indices = (i == 1).nonzero().flatten()
            tags.append(self._tag_names[tag_indices].tolist())
        return tags

    def transform(self, items: Iterable[Any]) -> Iterable[TaggingDataResult]:
        for item in items:
            audio = self.audio_transform(item["audio.npy"])
            tag = torch.tensor(item["tag_binary.npy"], dtype=torch.long)
            shard = item["__url__"]
            key = item["__key__"]

            audio = self.audio_pad(audio)
            audio = self.audio_crop(audio)
            yield TaggingDataResult(audio=audio, tag=tag, shard=shard, key=key)


class CombinedTaggingDataModule(WebDataModuleBase):
    def __init__(
        self,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        train_dataset = TaggingDataModule(
            split="train", duration=duration, resample=True
        )
        validation_dataset = TaggingDataModule(
            split="validation", duration=duration, resample=False
        )
        test_dataset = TaggingDataModule(
            split="test", duration=duration, resample=False
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
        )
