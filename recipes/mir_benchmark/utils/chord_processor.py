from abc import ABC
from random import randrange
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from recipes.chord.preprocess.common import get_chord_labels, get_chord_training_label
from samantha.dataio.preprocess import AudioLengthModifier


class ChordDatasetMixin(ABC):
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_chord_hop,
        chunk_per_sample,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._audio_length_modifier = AudioLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._label_chord_hop = label_chord_hop
        self._chunk_per_sample = chunk_per_sample

    def train_preprocess(self, x):
        np_audio = x["audio.npy"]
        oup = get_chord_labels(
            x["chords.pickle"], x["intervals.pickle"], self._label_chord_hop
        )

        data_queue = []
        for i in range(self._chunk_per_sample):
            output_dict = {}
            audio, _, start_idx = self._audio_length_modifier(
                torch.from_numpy(np_audio.reshape(1, -1))
            )
            sample_start_time = start_idx / self._audio_length_modifier.sampling_rate

            (
                output_dict["chord_root"],
                output_dict["chord_triad"],
                _,
                _,
                output_dict["chord_note"],
                output_dict["chord_boundary"],
                output_dict["chord_ignore"],
            ) = get_chord_training_label(
                len(np_audio),
                oup=oup,
                sample_start_time=sample_start_time,
                sample_rate=self._audio_length_modifier.sampling_rate,
                label_hop=self._label_chord_hop,
                sample_len=self._audio_length_modifier.target_duration_sec,
                is_train=True,
            )

            output_dict["audio"] = audio.squeeze()

            data_queue.append(output_dict)
        return data_queue

    def val_preprocess(self, x):
        oup = get_chord_labels(
            x["chords.pickle"], x["intervals.pickle"], self._label_chord_hop
        )
        output_dict = {}
        (
            output_dict["chord_root"],
            output_dict["chord_triad"],
            output_dict["chord_seventh"],
            output_dict["chord_bass"],
            output_dict["chord_note"],
            output_dict["chord_boundary"],
            output_dict["chord_ignore"],
        ) = get_chord_training_label(
            len(x["audio.npy"]),
            oup=oup,
            sample_start_time=0,
            sample_rate=self._audio_length_modifier.sampling_rate,
            label_hop=self._label_chord_hop,
            sample_len=self._audio_length_modifier.target_duration_sec,
            is_train=False,
        )
        output_dict["audio"] = torch.tensor(x["audio.npy"].squeeze())
        output_dict["dataset.txt"] = x["dataset.txt"]
        del x
        return output_dict


class ChordPreprocessor:
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_hop,
        sample_hop=1,
        num_train_samples_multiplier=1,
        *args,
        **kwargs,
    ):
        self._dataset_preprocessors = {}
        self._dataset_preprocessors = ChordDatasetMixin(
            target_duration_sec, sampling_rate, label_hop, 1, *args, **kwargs
        )
        self._target_duration_sec = target_duration_sec
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec
        self._sample_hop = sample_hop
        self._num_train_samples_multiplier = num_train_samples_multiplier

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            data_queue = preprocessor.train_preprocess(x)
            while len(data_queue) > 0:
                yield data_queue.pop(0)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)
