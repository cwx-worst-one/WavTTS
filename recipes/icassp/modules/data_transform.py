import io
import random
import soundfile as sf
from typing import Dict

import torch
import torchaudio
import torch.nn as nn
from nnAudio.features import CQT1992v2
from torchaudio_augmentations import Compose

from recipes.musicfm.modules.audio_utils import convert_audio_ffmpeg_bytes_to_bytes
from recipes.musicfm.modules.features import STFT, MelSTFT, MFCC, Chromagram
from recipes.audio_diffusion.modules.transforms.audio import (
    SetAudioDimensions,
    ToTensor,
    NormalizeAudioToFloat32
)


def shift_pitch(audio, audio_sr, pitch_shift):
    """
    Randomly shift the pitch up or down in the audio given the pitch_shift_range.

    Args:
        audio (torch.tensor): audio array in the shape of (ch, t).
        audio_sr (int): the sample rate of the audio.
        pitch_shift (int): pitch shifting.
    Return:
        audio (torch.tensor): the audio array after pitch shifting (ch, t).
    """

    len_in_sample = audio.shape[1]
    pitch_shift_in_cent = pitch_shift * 100
    shifted_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
        audio,
        audio_sr,
        [["pitch", f"{pitch_shift_in_cent}"], ["rate", f"{audio_sr}"]],
        channels_first=True,
    )  # sox is using cent as unit

    if shifted_audio.shape[1] != len_in_sample:
        shifted_audio = torch.nn.functional.pad(
            input=shifted_audio,
            pad=(0, len_in_sample - shifted_audio.shape[1]),
            mode="constant",
            value=0,
        )

    return shifted_audio


class FMAAudioTransforms(nn.Module):
    def __init__(
        self,
        length_seconds: int = 30,
        audio_sample_rate: int = 24000,
        is_random_pitch_shift: bool=False,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.n_audio_samples = self.audio_sample_rate * self.length_seconds

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.to_float32 = NormalizeAudioToFloat32()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.to_float32]
        )
        self.is_random_pitch_shift = is_random_pitch_shift

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}

        # byte to numpy
        np_audio = x["audio.npy"]
        audio = self.base_transform(np_audio)
        
        if self.n_audio_samples > audio.size(-1):
            output["__skip__"] = True
            return output

        # Crop audio
        audio_chunks = audio.unfold(1, self.n_audio_samples, self.audio_sample_rate)
        idx = random.randint(
            0, audio_chunks.size(1) - 1
        )
        output["audio"] = audio_chunks[:, idx, :]

        # Random pitch shift
        if self.is_random_pitch_shift:
            pitch_shift = random.sample([-4, -3, -2, -1, 1, 2, 3, 4], k=1)[0]
            output["shifted_audio"] = shift_pitch(output["audio"], self.audio_sample_rate, pitch_shift)
            output["pitch_shift"] = pitch_shift
        
        return output


class FMAAudioTransformsForStat(nn.Module):
    def __init__(
        self,
        length_seconds: int = 30,
        audio_sample_rate: int = 24000,
        hop_length: int = 240,
        n_fft: int=2048,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.n_audio_samples = self.audio_sample_rate * self.length_seconds

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.to_float32 = NormalizeAudioToFloat32()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.to_float32]
        )
        self.stft = STFT(n_fft, hop_length)
        self.mel_stft = MelSTFT(audio_sample_rate, n_fft, hop_length)
        self.mfcc = MFCC(audio_sample_rate, n_fft, hop_length)
        self.chromagram = Chromagram(audio_sample_rate, n_fft, hop_length)
        self.cqt = CQT1992v2(sr=audio_sample_rate, hop_length=hop_length)


    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}

        # byte to numpy
        np_audio = x["audio.npy"]
        audio = self.base_transform(np_audio)

        if self.n_audio_samples > audio.size(-1):
            output["__skip__"] = True
            return output

        # Crop audio
        audio_chunks = audio.unfold(1, self.n_audio_samples, self.audio_sample_rate)
        idx = random.randint(
            0, audio_chunks.size(1) - 1
        )
        random_chunk = audio_chunks[:, idx, :]
        
        # chromatic stft
        output["spec"] = self.stft(random_chunk)
        output["mel_spec"] = self.mel_stft(random_chunk)
        output["cqt"] = self.cqt(random_chunk)
        output["mfcc"] = self.mfcc(random_chunk)
        output["chromagram"] = self.chromagram(random_chunk)
        
        return output

