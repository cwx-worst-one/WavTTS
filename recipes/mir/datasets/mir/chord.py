import os
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from recipes.mir.datasets.mir import MIRDataModuleBase
from recipes.mir.datasets.mir.base import BaseAudioTransform, WebDataModuleBase
from samantha.transforms.audio import RandomPad, RandomResizedCrop
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.utils.logger import RankedLogger

logger = RankedLogger()

from sam.data.mir import MIRDataModuleBase


@dataclass
class ChordDataResult:
    audio: torch.Tensor
    chord_root: torch.Tensor
    chord_triad: torch.Tensor
    chord_seventh: torch.Tensor
    chord_bass: torch.Tensor
    chord_note: torch.Tensor
    chord_boundary: torch.Tensor
    chord_ignore: torch.Tensor
    dataset_name: str
    shard: str
    key: str


from mir_tools.chord.utils import get_chord_labels, get_chord_training_label


class ChordDataModule(MIRDataModuleBase):
    _splits = ["train", "validation", "test"]
    _directories = {
        "train": [
            # "chord/billboard_truth_chord_24kHz",
            "chord/billboard_chord_24kHz/train",
            "chord/isophonic_chord_24kHz/train",
            "chord/jaychou_chord_24kHz/train",
            "chord/leadsheet_chord_24kHz/train",
            "chord/pop909_chord_24kHz/train",
            "chord/rwc_chord_24kHz/train",
            "chord/uspop_chord_24kHz/train",
        ],
        "validation": [
            "chord/leadsheet_chord_24kHz/validation",
        ],
        "test": [
            "chord/leadsheet_chord_24kHz/validation",
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
        assert chunk_per_sample == 1  # TODO

        self._label_hop = label_hop
        self._chunk_per_sample = chunk_per_sample
        self._duration = duration
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

    def transform(self, items: Iterable[Any]) -> Iterable[ChordDataResult]:
        for item in items:
            audio = self.audio_transform(item["audio.npy"])
            original_audio_len = audio.shape[1]

            oup = get_chord_labels(
                item["chords.pickle"], item["intervals.pickle"], self._label_hop
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

                # compute for the entire audio, then chunk
                (
                    chord_root,
                    chord_triad,
                    chord_seventh,
                    chord_bass,
                    chord_note,
                    chord_boundary,
                    chord_ignore,
                ) = get_chord_training_label(
                    original_audio_len,
                    oup=oup,
                    sample_start_time=sample_start_time,
                    sample_rate=self.sample_rate,
                    label_hop=self._label_hop,  # framerate
                    sample_len=self._duration,
                    is_train=self.is_train,
                )
                shard = item["__url__"]
                key = item["__key__"]

                yield ChordDataResult(
                    audio=audio_chunk,
                    chord_root=torch.tensor(chord_root, dtype=torch.long),
                    chord_triad=torch.tensor(chord_triad, dtype=torch.long),
                    chord_seventh=torch.tensor(chord_seventh, dtype=torch.long),
                    chord_bass=torch.tensor(chord_bass, dtype=torch.long),
                    chord_note=torch.tensor(chord_note, dtype=torch.long),
                    chord_boundary=torch.tensor(chord_boundary),
                    chord_ignore=torch.tensor(chord_ignore),
                    dataset_name=item["dataset.txt"],
                    shard=shard,
                    key=key,
                )


class CombinedChordDataModule(WebDataModuleBase):
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
        train_dataset = ChordDataModule(
            split="train",
            duration=duration,
            label_hop=label_hop,
            chunk_per_sample=chunk_per_sample,
            resample=True,
        )
        validation_dataset = ChordDataModule(
            split="validation",
            duration=duration,
            label_hop=label_hop,
            chunk_per_sample=chunk_per_sample,
            resample=False,
        )
        test_dataset = ChordDataModule(
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
