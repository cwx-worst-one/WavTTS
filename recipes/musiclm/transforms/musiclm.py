from typing import Any, Dict, List, Generator, Optional
import io
import sys
import torch
import random
from torchaudio_augmentations import Compose

from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    LoudnessCheck,
    ReadMP3,
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


class MCCTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        sample_rate: int,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.5,
        aed_filtered: bool = False,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        max_num_crops: Optional[int] = None,    # if None, auto set based on audio length
        crop_step_size: Optional[int] = None,   # if None, auto set based on n_samples
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.aed_filtered = aed_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = set(exclude_licenses)
        self.max_num_crops = max_num_crops
        if crop_step_size is None:
            crop_step_size = self.n_samples // 2
        assert crop_step_size <= self.n_samples
        self.crop_step_size = crop_step_size
        self.crop_wing_span = self.n_samples // self.crop_step_size

        self.is_loud = LoudnessCheck(
            self.sample_rate,
            self.min_volume_threshold,
            self.loudness_ratio_threshold
        )
        self.read_mp3 = ReadMP3(self.sample_rate)
        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [
                self.read_mp3,
                self.to_tensor,
                self.audio_dim,
                self.normalize_audio_fp32
            ]
        )

        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def is_metadata_good(self, metadata: Dict[str, Any]) -> bool:
        # Apply AED filtering if applicable
        if self.aed_filtered and not metadata.get("aed_filtered", False):
            return False
        # Avoid sound effect if applicable
        if self.avoid_sound_effect and metadata.get("final_theme") == "Sound Effect":
            return False
        # Apply license-based filtering if applicable
        if len(self.exclude_licenses) > 0:
            for license in metadata.get("license_types", []):
                if license in self.exclude_licenses:
                    return False
        return True

    def __call__(self, x: Dict[str, Any]) -> Generator:
        if not self.is_metadata_good(x["__index_data__"]):
            self._update_stats(skipped=True)
            return

        try:
            audio = self.base_transform(io.BytesIO(x[self.audio_key]))
        except Exception as e:
            print(f"[MP3 decoding error] {e}")
            self._update_stats(skipped=True)
            return
        if audio.size(1) < self.n_samples * 0.95:
            self._update_stats(skipped=True)
            return
        else:            
            audio = self.random_pad(audio)
        
        # Determine possible crop starting points
        num_windows = 1 + (audio.size(1) - self.n_samples) // self.crop_step_size
        window_ids = list(range(num_windows))
        random.shuffle(window_ids)
        # Return up to max_num_crops
        max_num_crops = self.max_num_crops
        if max_num_crops is None:
            max_num_crops = max(1, audio.size(1) // self.n_samples // 4)
        num_crops = 0
        taboo = set()
        for wid in window_ids:
            if wid in taboo:
                continue
            st_sample = wid * self.crop_step_size
            cropped_audio = audio[:, st_sample : st_sample + self.n_samples]
            if not self.is_loud(cropped_audio):
                continue
            yield {"audio.npy": cropped_audio}
            num_crops += 1
            if num_crops >= max_num_crops:
                break
            # Avoid overlapping segments
            for overlapped_wid in range(
                max(0, wid - self.crop_wing_span + 1),
                min(num_windows, wid + self.crop_wing_span),
            ):
                taboo.add(overlapped_wid)
        self._update_stats(skipped=num_crops == 0)
