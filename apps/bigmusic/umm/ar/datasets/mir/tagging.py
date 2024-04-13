from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np
import torch
import webdataset as wds

from apps.bigmusic.umm.ar.datasets.base import BaseAudioTransform, DataResult, WebDataModuleBase
from samantha.transforms.audio import Pad, RandomResizedCrop
from samantha.utils.logger import RankedLogger
from samantha.utils.webdataset import return_self

logger = RankedLogger()

from dataclasses import dataclass

import numpy as np
import torch

from apps.bigmusic.umm.ar.datasets.mir.base import MIRDataModuleBase


@dataclass
class TaggingDataResult(DataResult):
    audio: torch.Tensor
    input_length: int
    tag: torch.Tensor
    tag_names: List[str]
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

    def __init__(self, split: str, duration: int, resampled: bool, shardshuffle: bool, nodesplitter = wds.shardlists.single_node_only):
        super().__init__(
            split=split,
            resampled=resampled,
            shardshuffle=shardshuffle,
            nodesplitter=nodesplitter,
        )

        self.duration = duration
        self.audio_transform = BaseAudioTransform()

        self.duration_samples = int(self._sample_rate * self.duration)
        self.audio_pad = Pad(self.duration_samples)
        self.audio_crop = RandomResizedCrop(self.duration_samples)

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
            input_length = audio.shape[1]

            tag = torch.tensor(item["tag_binary.npy"], dtype=torch.long)
            tag_names = self.tag_indices_to_names(tag)


            audio = self.audio_pad(audio)
            audio = self.audio_crop(audio)
            
            shard = item["__url__"]
            key = item["__key__"]
            yield TaggingDataResult(audio=audio, input_length=input_length, tag=tag, tag_names=tag_names, shard=shard, key=key)


class CombinedTaggingDataModule(WebDataModuleBase):
    def __init__(
        self,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        train_dataset = TaggingDataModule(
            split="train", duration=duration, resampled=resampled, shardshuffle=shardshuffle,
        )
        validation_dataset = TaggingDataModule(
            split="validation", duration=duration, resampled=False, shardshuffle=False, nodesplitter=return_self,
        )
        test_dataset = TaggingDataModule(
            split="test", duration=duration, resampled=False, shardshuffle=False, nodesplitter=return_self,
        )
        predict_dataset = test_dataset
        super().__init__(
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            train_dataset=train_dataset.dataset,
            validation_dataset=validation_dataset.dataset,
            test_dataset=test_dataset.dataset,
            predict_dataset=predict_dataset.dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size_valid=1,
            batch_size_test=1
        )
