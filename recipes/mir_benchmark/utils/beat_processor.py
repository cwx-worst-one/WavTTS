from abc import ABC
from random import randrange

import numpy as np
import torch
import torch.nn.functional as F

from recipes.beat.preprocess.common import get_beats_labels
from samantha.dataio.preprocess import AudioLengthModifier


class CustomLengthModifier(AudioLengthModifier):
    def __init__(
        self,
        sampling_rate: int,
        target_duration_sec: float,
        is_random_crop: bool = True,
    ):
        super().__init__(sampling_rate, target_duration_sec, is_random_crop)

        if target_duration_sec < 0:
            self._raise_config_error("target_duration_sec must be positive.")

    def __call__(self, audio: torch.Tensor):
        self._check_audio(audio)
        num_channels, source_frames = audio.shape
        target_frames = int(self.target_duration_sec * self.sampling_rate)

        start_idx = 0
        if source_frames < target_frames:
            output_audio = F.pad(
                input=audio,
                pad=(0, target_frames - source_frames),
                mode="constant",
                value=0,
            )
            padding_mask = F.pad(
                input=torch.zeros_like(audio),
                pad=(0, target_frames - source_frames),
                mode="constant",
                value=1,
            )
        else:
            if self.is_random_crop and (source_frames > target_frames):
                start_idx = randrange(source_frames - target_frames)
            output_audio = audio.clone()[:, start_idx : start_idx + target_frames]
            padding_mask = torch.zeros((num_channels, target_frames))
        return output_audio, padding_mask, start_idx


class BeatDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, label_hop, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._label_hop = label_hop
        self._audio_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._label_hop = label_hop

    def train_preprocess(self, x):
        out = x.copy()
        np_audio, beats, out["tempo_label"] = self._sample_audio_and_beat(x)
        out["audio.npy"] = torch.tensor(np_audio, dtype=torch.float32)
        out["beats.pickle"] = torch.tensor(beats)
        return out

    def val_preprocess(self, x):
        orig_beats = x["beats.pickle"]
        beat_label, tempo_label = get_beats_labels(
            x["beats.pickle"],
            len(x["audio.npy"]) / self._audio_length_modifier.sampling_rate,
            x["dataset.txt"],
            self._label_hop,
        )
        beat_label = beat_label[
            : int(
                len(x["audio.npy"])
                / self._audio_length_modifier.sampling_rate
                * (1 / self._label_hop)
            )
        ]

        pad_len = (
            self._audio_length_modifier.target_duration_sec * int(1 / self._label_hop)
            - beat_label.shape[0]
        )

        if pad_len > 0:
            beat_label = np.pad(beat_label, [(0, int(pad_len)), (0, 0)], "constant")
        x["orig_beats"] = orig_beats
        x["beats.pickle"] = torch.tensor(beat_label)
        x["tempo_label"] = torch.tensor(tempo_label)
        x["audio.npy"] = torch.tensor(x["audio.npy"])
        return x

    def _sample_audio_and_beat(self, x):
        beats = x["beats.pickle"]
        np_audio = x["audio.npy"]
        audio_duration_in_s = (
            np_audio.shape[0] / self._audio_length_modifier.sampling_rate
        )
        # convert beat to format for training
        beat_labels, tempo = get_beats_labels(
            beats, audio_duration_in_s, x["dataset.txt"], self._label_hop
        )
        label_len = int(
            self._audio_length_modifier.target_duration_sec / self._label_hop
        )

        # sample from audio
        audio, _, start_idx = self._audio_length_modifier(
            torch.from_numpy(np_audio.reshape(1, -1))
        )

        # get beat labels in sampled audio
        sample_idx = int(
            round(
                (start_idx / self._audio_length_modifier.sampling_rate)
                * (1 / self._label_hop)
            )
        )
        beat_labels = beat_labels[sample_idx : sample_idx + label_len, :]
        pad_len = label_len - beat_labels.shape[0]
        if pad_len > 0:
            beat_labels = np.pad(beat_labels, [(0, pad_len), (0, 0)], "constant")

        sampled_np_audio = np.squeeze(audio.numpy())

        return sampled_np_audio, beat_labels, tempo


class BeatPreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, label_hop, *args, **kwargs):
        self._dataset_preprocessors = BeatDatasetMixin(
            target_duration_sec, sampling_rate, label_hop, *args, **kwargs
        )
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec
        self._hop_factor = 1  # for beat

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.train_preprocess(x)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)
