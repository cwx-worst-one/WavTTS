import logging
from abc import ABC
from random import randrange
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

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


class GenreDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._audio_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self.genres = [
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        ]

    def train_preprocess(self, x):
        out = x.copy()
        np_audio, out["genre_label"] = self._sample_audio_and_genre(x)
        out["audio.npy"] = torch.tensor(np_audio, dtype=torch.float32)
        return out

    def val_preprocess(self, x):
        genre = np.array([self.genres.index(x["genre.txt"])])

        x["genre_label"] = torch.tensor(genre)
        x["audio.npy"] = torch.tensor(x["audio.npy"])
        return x

    def _sample_audio_and_genre(self, x):
        genre = x["genre.txt"]
        np_audio = x["audio.npy"]
        audio_duration_in_s = (
            np_audio.shape[0] / self._audio_length_modifier.sampling_rate
        )

        # sample from audio
        audio, _, start_idx = self._audio_length_modifier(
            torch.from_numpy(np_audio.reshape(1, -1))
        )

        sampled_np_audio = np.squeeze(audio.numpy())

        # genre index
        genre = np.array([self.genres.index(genre)])

        return sampled_np_audio, genre


class GenreTransformFactory:
    """The factory class for creating beat transform objects based on dataset."""

    registry = {}
    """ Internal registry for available beat transform objects based on dataset. """

    @classmethod
    def register(cls, name: str) -> Callable:
        """Class method to register MSS model classes to the internal registry.
        Args:
            name (str): The name of the MSS model.
        Returns:
            The MSS model class itself.
        """

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logging.warning("Executor %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Factory command to create the MSS model.
        This method gets the appropriate MSS model class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the MSS model to create.
        Returns:
            An instance of the MSS model that is created.
        """

        if name not in cls.registry:
            logging.warning("Executor %s does not exist in the registry", name)
            return None

        exec_class = cls.registry[name]
        executor = exec_class(*args, **kwargs)
        return executor


class GenrePreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        self._dataset_preprocessors = GenreDatasetMixin(
            target_duration_sec, sampling_rate, *args, **kwargs
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
