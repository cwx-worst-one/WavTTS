from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torchaudio_augmentations import Compose

from recipes.audio_diffusion.modules.transforms.audio import (
    NormalizeAudio,
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
    crop_1d,
    get_random_crop_idx,
    rms,
)


class SingSongTransforms(nn.Module):
    def __init__(
        self,
        n_samples: int,
        sample_rate: int,
        soundstream_frame_rate: int,
        semantic_frame_rate: int = 25,
        include_emb: bool = True,
        audio_key: str = "audio.npy",
        audio_key2: str = None,
        sample_range_key: str = "sample_range.npy",
        sample_range_key2: str = None,
        semantic_key: str = None,
        semantic_key2: str = None,
        min_volume_threshold: Optional[float] = None,
        white_noise_amplitude_stddev: Optional[float] = None,
        vocal_relative_gain_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.semantic_frame_rate = semantic_frame_rate
        self.soundstream_frame_rate = soundstream_frame_rate
        self.include_emb = include_emb
        self.audio_key = audio_key
        self.audio_key2 = audio_key2
        self.semantic_key = semantic_key
        self.semantic_key2 = semantic_key2
        self.sample_range_key = sample_range_key
        self.sample_range_key2 = sample_range_key2
        if self.audio_key2 is not None:
            assert self.sample_range_key2 is not None, (
                "If another audio key specifed, then we need to also specify sample"
                " range key"
            )
        self.min_volume_threshold = min_volume_threshold
        self.white_noise_amplitude_stddev = white_noise_amplitude_stddev
        self.vocal_relative_gain_threshold = vocal_relative_gain_threshold
        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )
        self.block_rate = np.lcm(soundstream_frame_rate, semantic_frame_rate)

    # Gets random_idx and length for audio and semantic tokens
    def get_random_idxs_and_lengths(self, audio_length_samples):
        audio_length_blocks = self.block_rate * audio_length_samples // self.sample_rate
        crop_length_blocks = self.block_rate * self.n_samples // self.sample_rate
        crop_length_audio_samples = (
            crop_length_blocks * self.sample_rate // self.block_rate
        )
        crop_length_semantic_frames = (
            crop_length_blocks * self.semantic_frame_rate // self.block_rate
        )

        random_idx_blocks = get_random_crop_idx(
            audio_length_blocks - crop_length_blocks
        )
        random_idx_audio_samples = (
            random_idx_blocks * self.sample_rate // self.block_rate
        )
        random_idx_semantic_frames = (
            random_idx_blocks * self.semantic_frame_rate // self.block_rate
        )
        return (
            random_idx_audio_samples,
            crop_length_audio_samples,
            random_idx_semantic_frames,
            crop_length_semantic_frames,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio = self.base_transform(x[self.audio_key])
        audio_length_samples = audio.shape[-1]
        if self.audio_key2 is not None:
            audio_length_samples = min(
                audio_length_samples, x[self.audio_key].shape[-1]
            )

        if audio_length_samples < self.n_samples:
            ret = {"__skip__": True}
            return ret

        (
            random_idx_audio_samples,
            crop_length_audio_samples,
            random_idx_semantic_frames,
            crop_length_semantic_frames,
        ) = self.get_random_idxs_and_lengths(audio_length_samples)

        sample_range = self.base_transform(x[self.sample_range_key])
        # TODO: investigate what's the effect of applying
        # vocal relative gain threshold in normalized audio vs unnormalized
        if self.vocal_relative_gain_threshold:
            norm_audio = audio
        else:
            norm_audio = self.normalize_audio(audio, norm_tensor=sample_range)
        # TODO: there is some rare edge case which is not handled properly by handling
        try:
            norm_audio = crop_1d(
                norm_audio,
                start_idx=random_idx_audio_samples,
                n_samples=crop_length_audio_samples,
            )
        except IndexError:
            ret["__skip__"] = True
            return ret

        ret = {"audio.npy": norm_audio, "audio_mix.npy": norm_audio}
        if self.audio_key2 is not None:
            audio2 = self.base_transform(x[self.audio_key2])
            sample_range2 = self.base_transform(x[self.sample_range_key2])
            # TODO: investigate what's the effect of applying
            # vocal relative gain threshold in normalized audio vs unnormalized
            if not self.vocal_relative_gain_threshold:
                norm_audio2 = self.normalize_audio(audio2, norm_tensor=sample_range2)
                # TODO: there is some rare edge case which is not handled properly
            try:
                norm_audio2 = crop_1d(
                    audio2,
                    start_idx=random_idx_audio_samples,
                    n_samples=crop_length_audio_samples,
                )
            except IndexError:
                ret["__skip__"] = True
                return ret

            # NOTE: J.S. I moved this up, so that we're filtering low-volume
            # vocals/accomopaniment before adding noise
            if self.min_volume_threshold is not None:
                # TODO: think use different kernel_size for rms than default 1000?
                either_is_silent = (
                    rms(norm_audio).max() < self.min_volume_threshold
                    or rms(norm_audio2).max() < self.min_volume_threshold
                )
                if either_is_silent:
                    ret["__skip__"] = True
                else:
                    if self.vocal_relative_gain_threshold is not None:
                        rms_audio = rms(norm_audio).max()
                        rms_audio2 = rms(norm_audio2).max()
                        ret["__skip__"] = (
                            rms_audio2 / rms_audio >= self.vocal_relative_gain_threshold
                        )
                    else:
                        ret["__skip__"] = False

            if self.white_noise_amplitude_stddev is not None:
                norm_audio2 += (
                    torch.randn_like(norm_audio2) * self.white_noise_amplitude_stddev
                )
            ret["audio2.npy"] = norm_audio2  # TODO: should this be a key?
            ret["audio_mix.npy"] = (norm_audio + norm_audio2) / 2
        if self.semantic_key is not None:
            semantic_frames = x[self.semantic_key]
            semantic_frames = semantic_frames[
                random_idx_semantic_frames : random_idx_semantic_frames
                + crop_length_semantic_frames
            ]
            ret["semantic.npy"] = semantic_frames
        if self.semantic_key2 is not None:
            semantic_frames2 = x[self.semantic_key2]
            semantic_frames2 = semantic_frames2[
                random_idx_semantic_frames : random_idx_semantic_frames
                + crop_length_semantic_frames
            ]
            ret["semantic2.npy"] = semantic_frames2
        if self.include_emb:
            ret["t2m_emb.npy"] = x["t2m_emb.npy"]
        else:
            # Include this here anyways since many components expect it
            ret["t2m_emb.npy"] = torch.zeros(1)

        # Hotifx: vocal/acc data not being of the same shape
        if norm_audio.shape[1] != norm_audio2.shape[1]:
            ret["__skip__"] = True
        return ret
