from abc import ABC

import numpy as np
import scipy
import torch

from recipes.mir_benchmark.utils.beat_processor import CustomLengthModifier
from recipes.vocal2midi.datasets.common import get_label_segment, note2array

SLAKH_INSTRUMENTS = [
    "Bass",
    "Brass",
    "Chromatic Percussion",
    "Drums",
    "Guitar",
    "Organ",
    "Piano",
    "Pipe",
    "Reed",
    "Strings",
    "Synth Lead",
    "Synth Pad",
]


class TranscriptionDatasetMixin(ABC):
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_time_resolution,
        pitch_resolution=1,
        pitch_start=21,
        pitch_range=88,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._audio_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self.label_time_resolution = label_time_resolution
        self.sampling_rate = sampling_rate
        self.pitch_resolution = pitch_resolution
        self.pitch_start = pitch_start
        self.pitch_range = pitch_range
        self.target_duration_sec = target_duration_sec

    def train_preprocess(self, x):
        n_frame = int(self.target_duration_sec / self.label_time_resolution)

        onset_output, pitch_output, audio_output = [], [], []
        for instrument in SLAKH_INSTRUMENTS:
            if instrument in x["audio.pickle"]:
                np_audio = x["audio.pickle"][instrument]
                len_audio = (np_audio.shape[0]) / self.sampling_rate
                pitch_label = note2array(
                    x["notes.pickle"][instrument],
                    time_resolution=self.label_time_resolution,
                    end_time=len_audio,
                    pitch_range=self.pitch_range,
                    pitch_resolution=self.pitch_resolution,
                    pitch_start=self.pitch_start,
                )
                pitch_roll = np.roll(pitch_label, 1, 0)
                onset_label = pitch_label - pitch_roll
                onset_label[onset_label < 1] = 0

                audio, _, start_idx = self._audio_length_modifier(
                    torch.from_numpy(np_audio.reshape(1, -1))
                )

                sample_start_time = start_idx / self.sampling_rate
                start_idx = int(
                    round(sample_start_time * (1 / self.label_time_resolution))
                )
                pitch_label = get_label_segment(pitch_label, start_idx, n_frame)
                onset_label = get_label_segment(onset_label, start_idx, n_frame)

                onset_label = scipy.ndimage.gaussian_filter1d(
                    onset_label[:, :], 1, axis=0, mode="constant"
                )
            else:
                audio = torch.zeros(
                    (1, int(self.target_duration_sec * self.sampling_rate))
                )
                onset_label = np.zeros((n_frame, 129))
                pitch_label = np.zeros((n_frame, 129))

            onset_output.append(onset_label)
            pitch_output.append(pitch_label)
            audio_output.append(audio.squeeze().unsqueeze(0))

        onset_output = np.array(onset_output)[..., 1:]
        pitch_output = np.array(pitch_output)[..., 1:]
        audio_output = torch.cat(audio_output, 0)

        output_dict = {}
        output_dict["audio"] = audio_output
        output_dict["note_labels"] = pitch_output
        output_dict["onset_labels"] = onset_output

        return output_dict

    def val_preprocess(self, x):
        np_audio = 0
        for instrument in SLAKH_INSTRUMENTS:
            if instrument in x["audio.pickle"]:
                np_audio += x["audio.pickle"][instrument]

        output_dict = {}
        output_dict["audio"] = torch.from_numpy(np_audio).squeeze()
        output_dict["note_seq"] = x["notes.pickle"]

        return output_dict


class TranscriptionPreprocessor:
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_resolution,
        pitch_resolution=1,
        pitch_start=1,
        pitch_range=128,
        *args,
        **kwargs,
    ):
        self._dataset_preprocessors = {}
        self._dataset_preprocessors = TranscriptionDatasetMixin(
            target_duration_sec,
            sampling_rate,
            label_resolution,
            pitch_resolution=pitch_resolution,
            pitch_start=pitch_start,
            pitch_range=pitch_range,
            *args,
            **kwargs,
        )

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.train_preprocess(x)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)
