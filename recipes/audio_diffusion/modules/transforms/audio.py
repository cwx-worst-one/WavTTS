import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import avg_pool1d, pad
from torchaudio_augmentations import Compose

from ..util.math import safe_divide


def to_tensor(x: np.ndarray):
    return torch.from_numpy(x)


def normalize_audio_to_float32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        pass
    elif tensor.dtype == torch.int32:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 2147483647.0
        tensor[tensor < 0] /= 2147483648.0
    elif tensor.dtype == torch.int16:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 32767.0
        tensor[tensor < 0] /= 32768.0
    elif tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) - 128
        tensor[tensor > 0] /= 127.0
        tensor[tensor < 0] /= 128.0
    return tensor


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


def get_random_crop_idx(n_samples: int):
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
        rand_idx = get_random_crop_idx(max_samples - self.n_samples)
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

        y = torch.zeros(x.shape[0], self.n_samples, device=x.device)
        rand_indx = random.randint(0, self.n_samples - x.shape[-1])
        y[:, rand_indx : rand_indx + x.shape[-1]] = x
        return y


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


def rms(x: torch.Tensor, kernel_size: int = 1000) -> torch.Tensor:
    return torch.sqrt(avg_pool1d(x**2, kernel_size=kernel_size, stride=1))
