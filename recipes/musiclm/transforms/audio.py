import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import avg_pool1d, pad
from torchaudio_augmentations import Compose

from recipes.musiclm.utils.math import safe_divide

from pydub import AudioSegment


def to_tensor(x: np.ndarray):
    return torch.from_numpy(x)


def power_to_db(x: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(x)


def amplitude_to_db(x: torch.Tensor) -> torch.Tensor:
    return power_to_db(x.abs().pow(2))


def normalize_audio_to_float32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor

    info = torch.iinfo(tensor.dtype)
    abs_max = 2 ** (info.bits - 1)
    return tensor.float() / abs_max


def rms(x: torch.Tensor, kernel_size: int = 1000) -> torch.Tensor:
    """TODO test

    Args:
        x (torch.Tensor): _description_
        kernel_size (int, optional): _description_. Defaults to 1000.

    Returns:
        torch.Tensor: _description_
    """
    return torch.sqrt(avg_pool1d(x**2, kernel_size=kernel_size, stride=1))


class Identity:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ToTensor:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        """Converts numpy array to torch.Tensor

        Args:
            x (np.ndarray): Audio array

        Returns:
            torch.Tensor: Audio tensor
        """
        return to_tensor(x)


class SetAudioDimensions:
    def __init__(self):
        pass

    def __call__(self, x):
        if x.ndim == 1:
            return x[None, :]
        return x


def get_random_idx(n_samples: int):
    return random.randint(0, n_samples)


def crop_1d(audio: torch.Tensor, start_idx: int, n_samples: int):
    if audio.shape[1] == n_samples:
        return audio

    if (start_idx + n_samples) > audio.shape[1]:
        raise IndexError(
            f"The number of samples needed({start_idx + n_samples}) to crop exceeds the"
            f" max_samples({audio.shape[1]}) in the audio"
        )
    return audio[..., start_idx : start_idx + n_samples]


class ResizedCrop(torch.nn.Module):
    def __init__(self, start_idx: int, n_samples: int):
        super().__init__()
        self.start_idx = start_idx
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return crop_1d(x, self.start_idx, self.n_samples)


class RandomResizedCrop(torch.nn.Module):
    def __init__(self, n_samples: int):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x):
        max_samples = x.shape[-1]
        rand_idx = get_random_idx(max_samples - self.n_samples)
        return crop_1d(x, rand_idx, self.n_samples)


class NormalizeAudioToFloat32:
    def __init__(self):
        pass

    def __call__(self, x):
        return normalize_audio_to_float32(x)


class Pad(nn.Module):
    def __init__(self, n_samples: int) -> None:
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D input (n_channels x n_samples), got {x.ndim}D input"
            )

        if x.shape[-1] >= self.n_samples:
            return x

        return pad(x, (0, self.n_samples - x.shape[-1]))


def pad_1d(audio: torch.Tensor, start_idx: int, n_samples: int):
    y = torch.zeros(audio.shape[0], n_samples, device=audio.device)
    y[:, start_idx : start_idx + audio.shape[-1]] = audio
    return y


class RandomPad(nn.Module):
    def __init__(self, n_samples: int) -> None:
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D input (n_channels x n_samples), got {x.ndim}D input"
            )

        if x.shape[-1] >= self.n_samples:
            return x

        rand_idx = random.randint(0, self.n_samples - x.shape[-1])
        return pad_1d(x, rand_idx, self.n_samples)


class NormalizeAudio(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, norm_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if norm_tensor is None:
            denom = x.abs().max()
        else:
            denom = norm_tensor.abs().max()
        return safe_divide(x, denom)


class SplitView(nn.Module):
    def __init__(self, views: List[Compose]) -> None:
        super().__init__()
        self.views = views

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return tuple(map(lambda v: v(x), self.views))


class LoudnessCheck:
    def __init__(
        self,
        sample_rate: int,
        threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
    
    def __call__(self, audio: torch.Tensor) -> bool:
        window_size = int(self.sample_rate * 0.1)

        frames = audio.unfold(1, window_size, window_size)
        energy = torch.max(torch.abs(frames), dim=-1)[0]
        ratio = torch.sum(energy > self.threshold) / energy.size(1)

        if ratio < self.loudness_ratio_threshold:
            return False
        return True


class ReadMP3:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def __call__(self, mp3: bytes) -> np.ndarray:
        audio = AudioSegment.from_file(mp3, format="mp3")
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(self.sample_rate)
        wav = np.asarray(audio.get_array_of_samples())
        return wav
