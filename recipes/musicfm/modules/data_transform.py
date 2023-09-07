import io
import random
from typing import Dict

import soundfile as sf
import torch
import torch.nn as nn
from nnAudio.features import CQT1992v2
from torchaudio_augmentations import Compose

from recipes.audio_diffusion.modules.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)
from recipes.musicfm.modules.audio_utils import convert_audio_ffmpeg_bytes_to_bytes
from recipes.musicfm.modules.features import (
    MFCC,
    STFT,
    Chromagram,
    ChromaticSTFT,
    MelSTFT,
)


class AudioTransforms(nn.Module):
    def __init__(
        self, length_seconds: int = 30, audio_sample_rate: int = 24000
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.n_audio_samples = self.audio_sample_rate * self.length_seconds

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.base_transform = Compose([self.to_tensor, self.audio_dim])

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}

        # byte to numpy
        try:
            audio_as_bytes = convert_audio_ffmpeg_bytes_to_bytes(
                x["mp3"], mono=True, sampling_rate=24000
            )
        except:  # byte read error
            output["__skip__"] = True
            return output
        np_audio, _ = sf.read(io.BytesIO(audio_as_bytes))
        audio = self.base_transform(np_audio.astype("float32"))

        if self.n_audio_samples > audio.size(-1):
            output["__skip__"] = True
            return output

        # Crop audio
        audio_chunks = audio.unfold(1, self.n_audio_samples, self.audio_sample_rate)
        idx = random.randint(0, audio_chunks.size(1) - 1)
        output["audio"] = audio_chunks[:, idx, :]

        return output


class AudioTransformsForStat(nn.Module):
    def __init__(
        self,
        length_seconds: int = 30,
        audio_sample_rate: int = 24000,
        hop_length: int = 240,
        n_fft: int = 2048,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.n_audio_samples = self.audio_sample_rate * self.length_seconds

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.base_transform = Compose([self.to_tensor, self.audio_dim])
        self.stft = STFT(audio_sample_rate, n_fft, hop_length)
        self.mel_stft = MelSTFT(audio_sample_rate, n_fft, hop_length)
        self.chromatic_stft = ChromaticSTFT(audio_sample_rate, n_fft, hop_length)
        self.mfcc = MFCC(audio_sample_rate)
        self.chromagram = Chromagram(audio_sample_rate, n_fft, hop_length)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}

        # byte to numpy
        audio_as_bytes = convert_audio_ffmpeg_bytes_to_bytes(
            x["mp3"], mono=True, sampling_rate=24000
        )
        np_audio, _ = sf.read(io.BytesIO(audio_as_bytes))
        audio = self.base_transform(np_audio.astype("float32"))

        if self.n_audio_samples > audio.size(-1):
            output["__skip__"] = True
            return output

        # Crop audio
        audio_chunks = audio.unfold(1, self.n_audio_samples, self.audio_sample_rate)
        idx = random.randint(0, audio_chunks.size(1) - 1)
        random_chunk = audio_chunks[:, idx, :]

        # chromatic stft
        output["spec"] = self.stft(random_chunk)
        output["mel_spec"] = self.mel_stft(random_chunk)
        output["chromatic_spec"] = self.chromatic_stft(random_chunk)
        output["mfcc"] = self.mfcc(random_chunk)
        output["chromagram"] = self.chromagram(random_chunk)

        return output


class PlaylistAudioTransforms(nn.Module):
    def __init__(
        self,
        length_seconds: int = 30,
        audio_sample_rate: int = 24000,
        hop_length: int = 240,
        n_fft: int = 2048,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.n_audio_samples = self.audio_sample_rate * self.length_seconds

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.to_float32 = NormalizeAudioToFloat32()
        self.base_transform = Compose([self.to_tensor, self.audio_dim, self.to_float32])

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
        idx = random.randint(0, audio_chunks.size(1) - 1)
        output["audio"] = audio_chunks[:, idx, :]

        return output


class PlaylistAudioTransformsForStat(nn.Module):
    def __init__(
        self,
        length_seconds: int = 30,
        audio_sample_rate: int = 24000,
        hop_length: int = 240,
        n_fft: int = 2048,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.n_audio_samples = self.audio_sample_rate * self.length_seconds

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.to_float32 = NormalizeAudioToFloat32()
        self.base_transform = Compose([self.to_tensor, self.audio_dim, self.to_float32])
        self.stft = STFT(audio_sample_rate, n_fft, hop_length)
        self.mel_stft = MelSTFT(audio_sample_rate, n_fft, hop_length)
        self.mfcc = MFCC(audio_sample_rate)
        self.chromagram = Chromagram(audio_sample_rate)
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
        idx = random.randint(0, audio_chunks.size(1) - 1)
        random_chunk = audio_chunks[:, idx, :]

        # chromatic stft
        output["spec"] = self.stft(random_chunk)
        output["mel_spec"] = self.mel_stft(random_chunk)
        output["cqt"] = self.cqt(random_chunk)
        output["mfcc"] = self.mfcc(random_chunk)
        output["chromagram"] = self.chromagram(random_chunk)

        return output
