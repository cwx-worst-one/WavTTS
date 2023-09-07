from abc import ABC

import torch

from recipes.key_detection.preprocess.common import (
    get_key_training_label,
    get_keymode_labels,
)
from samantha.dataio.preprocess import AudioLengthModifier


class KeyDatasetMixin(ABC):
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_key_hop,
        chunk_per_sample,
        *args,
        **kwargs,
    ):
        super().__init__()  # forwards all unused arguments
        self._audio_length_modifier = AudioLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._label_key_hop = label_key_hop
        self._chunk_per_sample = chunk_per_sample

    def train_preprocess(self, x):
        np_audio = x["audio.npy"]
        audio_duration_in_s = len(np_audio) / self._audio_length_modifier.sampling_rate

        key_labels = get_keymode_labels(
            x["keys.pickle"],
            x["intervals.pickle"],
            audio_duration_in_s,
            self._label_key_hop,
        )

        data_queue = []
        for i in range(self._chunk_per_sample):
            output_dict = {}
            audio, _, start_idx = self._audio_length_modifier(
                torch.from_numpy(np_audio.reshape(1, -1))
            )
            sample_start_time = start_idx / self._audio_length_modifier.sampling_rate

            output_dict["key_label"] = get_key_training_label(
                key_labels=key_labels,
                audio_duration_in_s=audio_duration_in_s,
                sample_start_time_in_s=sample_start_time,
                label_hop=self._label_key_hop,
                sample_len_in_s=self._audio_length_modifier.target_duration_sec,
                is_train=True,
            )
            output_dict["audio"] = audio.squeeze()
            output_dict["__key__"] = x["__key__"]

            data_queue.append(output_dict)

        return data_queue

    def val_preprocess(self, x):
        output_dict = {}

        np_audio = x["audio.npy"]
        audio_duration_in_s = len(np_audio) / self._audio_length_modifier.sampling_rate

        key_labels = get_keymode_labels(
            x["keys.pickle"],
            x["intervals.pickle"],
            audio_duration_in_s,
            self._label_key_hop,
        )

        output_dict["key_label"] = get_key_training_label(
            key_labels=key_labels,
            audio_duration_in_s=audio_duration_in_s,
            sample_start_time_in_s=0,
            label_hop=self._label_key_hop,
            sample_len_in_s=self._audio_length_modifier.target_duration_sec,
            is_train=False,
        )
        output_dict["audio"] = torch.from_numpy(np_audio).squeeze()
        output_dict["__key__"] = x["__key__"]
        return output_dict


class KeyPreprocessor:
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_hop,
        num_train_samples_multiplier=1,
        *args,
        **kwargs,
    ):
        self._dataset_preprocessors = {}
        self._dataset_preprocessors = KeyDatasetMixin(
            target_duration_sec,
            sampling_rate,
            label_hop,
            num_train_samples_multiplier,
            *args,
            **kwargs,
        )
        self._target_duration_sec = target_duration_sec
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            data_queue = preprocessor.train_preprocess(x)
            while len(data_queue) > 0:
                yield data_queue.pop(0)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)
