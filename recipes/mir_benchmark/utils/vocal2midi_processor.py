from abc import ABC

import scipy
import torch

from recipes.mir_benchmark.utils.beat_processor import CustomLengthModifier
from recipes.vocal2midi.datasets.common import get_label_segment, note2array


class Vocal2MidiDatasetMixin(ABC):
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
        np_audio = x["audio.npy"]
        len_audio = (np_audio.shape[0]) / self.sampling_rate
        pitch_label = note2array(
            x["note_seq.pickle"],
            time_resolution=self.label_time_resolution,
            end_time=len_audio,
            pitch_range=self.pitch_range,
            pitch_resolution=self.pitch_resolution,
            pitch_start=self.pitch_start,
        )
        onset_label = note2array(
            x["onset_seq.pickle"],
            time_resolution=self.label_time_resolution,
            end_time=len_audio,
            onset_len=2,
            pitch_range=self.pitch_range,
            pitch_resolution=self.pitch_resolution,
            pitch_start=self.pitch_start,
        )

        audio, _, start_idx = self._audio_length_modifier(
            torch.from_numpy(np_audio.reshape(1, -1))
        )

        n_frame = int(self.target_duration_sec / self.label_time_resolution)
        sample_start_time = start_idx / self.sampling_rate
        start_idx = int(round(sample_start_time * (1 / self.label_time_resolution)))
        pitch_label = get_label_segment(pitch_label, start_idx, n_frame)
        onset_label = get_label_segment(onset_label, start_idx, n_frame)

        onset_label = scipy.ndimage.gaussian_filter1d(
            onset_label[:, :], 1, axis=0, mode="constant"
        )

        output_dict = {}
        output_dict["audio"] = audio.squeeze()
        output_dict["note_labels"] = pitch_label
        output_dict["onset_labels"] = onset_label

        return output_dict

    def val_preprocess(self, x):
        np_audio = x["audio.npy"]

        output_dict = {}
        output_dict["audio"] = torch.from_numpy(np_audio).squeeze()
        output_dict["note_seq"] = x["note_seq.pickle"]
        output_dict["dataset.txt"] = x["dataset.txt"]

        return output_dict


class Vocal2midiPreprocessor:
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
        self._dataset_preprocessors = Vocal2MidiDatasetMixin(
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
