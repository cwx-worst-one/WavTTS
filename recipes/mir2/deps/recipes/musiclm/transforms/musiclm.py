from typing import Dict, Generator, Optional

import torch
from torchaudio_augmentations import Compose

from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
)
from recipes.musiclm.transforms.base import TransformBase


class MusicLMTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        audio_key: str,
        sample_range_key: Optional[str] = None,
        min_volume_threshold: Optional[float] = 0.0,
        num_crops: int = 1,
        num_tries: int = 5,
    ) -> None:
        super().__init__()
        self.audio_key = audio_key
        self.sample_range_key = sample_range_key
        self.min_volume_threshold = min_volume_threshold
        self.num_crops = num_crops
        self.num_tries = num_tries

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )

        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        audio = self.base_transform(x[self.audio_key])

        if self.sample_range_key is not None:
            sample_range = self.base_transform(x[self.sample_range_key])
            audio = self.normalize_audio(audio, norm_tensor=sample_range)

        audio = self.random_pad(audio)
        # Return up to self.num_crops crops
        num_crops = 0
        num_tries = 0
        while num_crops < self.num_crops and num_tries < self.num_tries:
            cropped_audio = self.random_crop(audio)
            if (
                self.min_volume_threshold is None
                or torch.mean(torch.abs(cropped_audio)) >= self.min_volume_threshold
            ):
                yield {"audio.npy": cropped_audio}
                num_crops += 1
            num_tries += 1
        self._update_stats(num_crops == 0)
