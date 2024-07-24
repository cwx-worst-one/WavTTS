from typing import Any, Iterable, List

import numpy as np
import torch
import webdataset as wds
from webdataset.shardlists import split_by_node

from recipes.datasets.base import BaseAudioTransform, DataResult, WebDataModuleBase
from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo, ShardInfo
from samantha.transforms.audio import Pad, RandomResizedCrop
from samantha.utils.logger import RankedLogger

logger = RankedLogger()

from dataclasses import dataclass

import numpy as np
import torch

from recipes.datasets.mir.base import MIRDataModuleBase
from samantha.data.audio_utils import convert_audio


class TaggingDataModule(MIRDataModuleBase):
    _splits = ["train", "validation", "test"]

    _directories = {
        "train": ["tagging/mtat_tagging_24kHz/train"],
        "validation": ["tagging/mtat_tagging_24kHz/validation"],
        "test": ["tagging/mtat_tagging_24kHz/test"],
    }

    _data_sample_rate = 24000

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

    def __init__(
        self,
        sample_rate: int,
        n_channels: int,
        split: str,
        duration: int,
        resampled: bool,
        shardshuffle: bool,
        nodesplitter=wds.shardlists.single_node_only,
    ):
        super().__init__(
            split=split,
            resampled=resampled,
            shardshuffle=shardshuffle,
            nodesplitter=nodesplitter,
        )

        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.duration = duration
        self.audio_transform = BaseAudioTransform()

        self.duration_samples = int(self._data_sample_rate * self.duration)
        self.audio_pad = Pad(self.duration_samples, value=0.0)
        self.audio_crop = RandomResizedCrop(self.duration_samples)

    @property
    def tag_names(self):
        return self._tag_names

    def tag_names_to_indices(self, tag_names: List[str]) -> torch.Tensor:
        tag2idx = {tag: idx for idx, tag in enumerate(self._tag_names)}
        tag_indices = []
        for tag in tag_names:
            tag_indices.append(tag2idx[tag])
        return torch.tensor(tag_indices, dtype=torch.long)

    def tag_indices_to_names(self, tags_multi_one_hot: torch.Tensor):
        if tags_multi_one_hot.ndim == 1:
            tags_multi_one_hot = tags_multi_one_hot[None, :]

        tags = []
        for i in tags_multi_one_hot:
            tag_indices = (i == 1).nonzero().flatten()
            tags.append(self._tag_names[tag_indices].tolist())
        return tags

    def transform(self, items: Iterable[Any]) -> Iterable[AudioDataResult]:
        for item in items:
            audio = self.audio_transform(item["audio.npy"])

            audio = convert_audio(
                audio, self.data_sample_rate, self.sample_rate, self.n_channels
            )

            n_frames = audio.shape[1]
            duration = n_frames / self.sample_rate

            tag = torch.tensor(item["tag_binary.npy"], dtype=torch.long)
            tag_names = self.tag_indices_to_names(tag)

            audio = self.audio_pad(audio)
            audio, crop_idx = self.audio_crop(audio, return_sample_idx=True)
            seek_time = crop_idx / self.sample_rate


            shard = item["__url__"]
            key = item["__key__"]

            segment_info = SegmentInfo(
                meta=AudioMeta(path=key, duration=duration, sample_rate=self.sample_rate),
                seek_time=seek_time,
                n_frames=n_frames,
                total_frames=audio.shape[1],
                sample_rate=self.sample_rate,
                channels=audio.shape[0],
                data_type="music_vocal",
            )

            index = dict(
                tag=tag,
                tag_names=tag_names,
            )

            yield AudioDataResult(
                audio=audio,
                segment_info=segment_info,
                index=index,
                shard=shard,
                key=key,
                shard_info=ShardInfo(url=shard),
            )


class CombinedTaggingDataModule(WebDataModuleBase):
    def __init__(
        self,
        sample_rate: int,
        n_channels: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        train_dataset = TaggingDataModule(
            sample_rate=sample_rate,
            n_channels=n_channels,
            split="train",
            duration=duration,
            resampled=resampled,
            shardshuffle=shardshuffle,
        )
        validation_dataset = TaggingDataModule(
            sample_rate=sample_rate,
            n_channels=n_channels,
            split="validation",
            duration=duration,
            resampled=False,
            shardshuffle=False,
            nodesplitter=split_by_node,
        )
        test_dataset = TaggingDataModule(
            sample_rate=sample_rate,
            n_channels=n_channels,
            split="test",
            duration=duration,
            resampled=False,
            shardshuffle=False,
            nodesplitter=split_by_node,
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
            batch_size_test=1,
        )
