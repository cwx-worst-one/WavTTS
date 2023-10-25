import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Union

import numpy as np
import torch
import logging

from recipes.datasets.mir.base import (
    BaseAudioTransform,
    WebDataModuleBase,
    MIRDataModuleBase,
)
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.transforms.audio import RandomPad, RandomResizedCrop

logger = logging.getLogger(__name__)


def get_beats_labels(times, song_duration, dataset, hop_in_sec=0.2):
    # TODO: tests
    hop_per_sec = 1 / hop_in_sec
    end_idx = np.ceil(song_duration * hop_per_sec).astype(np.int32)
    beat_labels = np.zeros((end_idx, 2))

    if dataset in ["smc_beat", "simac_beat", "hjdb_beat"]:
        only_beat = True
    else:
        only_beat = False

    for time in times:
        t = np.round(np.array(time[0]) * hop_per_sec).astype(np.int32)

        if t < end_idx:
            if not only_beat:
                if time[1] == 1:
                    beat_labels[t, 1] = 1
                else:
                    beat_labels[t, 0] = 1
            else:
                beat_labels[t, 0] = 1

    if (times[-1][0] - times[0][0]) == 0:
        tempo = 0
    else:
        tempo = int(
            round((beat_labels.sum(0)[0] - 1) / (times[-1][0] - times[0][0]) * 60)
        )
    return beat_labels, tempo


@dataclass
class BeatDataResult:
    audio: torch.Tensor
    beat_labels: torch.Tensor
    orig_beats: np.ndarray
    tempo: Union[int, List[int]]
    dataset_name: str
    shard: str
    key: str


class BeatDataModule(MIRDataModuleBase):
    _root = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/mir_benchmark/beat"
    _splits = ["train", "validation", "test"]

    _directories = {
        "train": [
            "ballroom_beat_24kHz/train",
            "beatles_beat_24kHz/train",
            "hainsworth_beat_24kHz/train",
            "hjdb_beat_24kHz/train",
            "karaoke_beat_24kHz/train",
            "rwc_beat_24kHz/train",
            "simac_beat_24kHz/train",
            "smc_beat_24kHz/train",
            "harmonix_beat_24kHz/train",
            # "karaoke_beat_vocal_24kHz/train", # Different task
        ],
        "validation": [
            "gtzan_beat_24kHz/validation",
            "clip500_beat_24kHz/validation",
            "bytebeat_beat_24kHz/validation",
            # "MCC126_beat_24kHz/validation", # Different task
        ],
        "test": [
            "gtzan_beat_24kHz/test",
            "clip500_beat_24kHz/test",
            "bytebeat_beat_24kHz/test",
            # "MCC32_beat_24kHz/test", # Different task
        ],
    }
    _sample_rate = 24000

    def __init__(
        self,
        split: str,
        duration: float,
        label_hop: float,
        resample: bool,
    ):
        super().__init__(
            split=split,
            resample=resample,
        )
        self._duration = duration
        self._label_hop = label_hop
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

    def transform(self, items: Iterable[Any]) -> Iterable[BeatDataResult]:
        for item in items:
            audio = self.audio_transform(item["audio.npy"])
            beats = item["beats.pickle"]

            original_audio_len = audio.shape[1]

            audio_duration_in_s = original_audio_len / self.sample_rate
            # convert beat to format for training
            beat_labels, tempo = get_beats_labels(
                beats, audio_duration_in_s, item["dataset.txt"], self._label_hop
            )
            label_len = int(self._duration / self._label_hop)
            if self.is_train:
                # sample audio
                audio_chunk = self.random_pad(audio)
                audio_chunk, start_idx = self.random_crop(
                    audio_chunk, return_sample_idx=True
                )

                # get beat labels in sampled audio
                sample_idx = int(
                    round((start_idx / self.sample_rate) * (1 / self._label_hop))
                )

                beat_labels = beat_labels[sample_idx : sample_idx + label_len, :]
            else:
                audio_chunk = audio
                beat_labels = beat_labels[: int(audio_duration_in_s / self._label_hop)]

            pad_len = label_len - beat_labels.shape[0]
            if pad_len > 0:
                beat_labels = np.pad(
                    beat_labels, [(0, int(pad_len)), (0, 0)], "constant"
                )

            shard = item["__url__"]
            key = item["__key__"]
            yield BeatDataResult(
                audio=audio_chunk,
                beat_labels=torch.tensor(beat_labels),
                orig_beats=beats,
                tempo=torch.tensor(tempo),
                dataset_name=item["dataset.txt"],
                shard=shard,
                key=key,
            )


class CombinedBeatDataModule(WebDataModuleBase):
    def __init__(
        self,
        duration: float,
        label_hop: float,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        train_dataset = BeatDataModule(
            split="train",
            duration=duration,
            label_hop=label_hop,
            resample=True,
        )
        validation_dataset = BeatDataModule(
            split="validation",
            duration=duration,
            label_hop=label_hop,
            resample=False,
        )
        test_dataset = BeatDataModule(
            split="test",
            duration=duration,
            label_hop=label_hop,
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
        )
