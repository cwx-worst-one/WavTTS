import io
import random
from typing import Callable, List, Optional, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad
from torchaudio.functional.functional import _get_spec_norms
from torchaudio.transforms import MelScale


def safe_log(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(x + eps)


def fp32_to_int16(audio: torch.Tensor) -> torch.Tensor:
    max_val = torch.iinfo(torch.int16).max
    return (audio * max_val).to(torch.int16)


def to_tensor(x: np.ndarray):
    if torch.is_tensor(x):
        return x
    if isinstance(x, bytes):
        x, _ = librosa.load(io.BytesIO(x), sr=None)

    return torch.from_numpy(x)


def power_to_db(x: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(x)


def amplitude_to_db(x: torch.Tensor) -> torch.Tensor:
    return power_to_db(x.abs().pow(2))


def normalize_audio_to_float32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor

    if tensor.dtype == torch.float64:
        return tensor.float()

    info = torch.iinfo(tensor.dtype)
    abs_max = 2 ** (info.bits - 1)
    return tensor.float() / abs_max


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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Sets the expected audio dimensions according to
        PyTorch standards: (n_channels, n_samples)

        Args:
            x (torch.Tensor): Tensor containing audio samples of size:
                - [n_samples] -> [1, n_samples]
                - [n_channels, n_samples] -> [n_channels, n_samples]
                - [n_samples, n_channels] -> [n_channels, n_samples]

        Returns:
            torch.Tensor: Tensor containing audio samples of size:
        """
        if x.ndim == 1:
            return x[None, :]
        elif x.ndim == 2:
            if x.shape[0] > x.shape[1]:
                x = x.permute(1, 0)
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


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.exp(torch.log(torch.clamp(x, min=clip_val) * C))


def stft(
    waveform: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    pad: int,
    power: float,
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
):
    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    x_stft = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm,
        onesided=onesided,
        return_complex=True,
    )

    # unpack batch
    x_stft = x_stft.reshape(shape[:-1] + x_stft.shape[-2:])
    x_phs = torch.angle(x_stft)

    if window_norm:
        x_stft /= window.pow(2.0).sum().sqrt()

    if power is not None:
        if power == 1.0:
            x_stft = x_stft.abs()
        else:
            x_stft = x_stft.abs().pow(power)
    return x_stft, x_phs


def plot_spectrogram(
    spec: torch.Tensor,
    plot_log: bool,
    mel: bool = False,
    title: Optional[str] = None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if title is not None:
        ax.set_title(title)

    ylabel = "bin" if mel else "Hz"
    ax.set_ylabel(ylabel)

    if plot_log:
        spec = spec.log()

    ax.imshow(spec, origin="lower", aspect="auto", interpolation="nearest")


def batch_plot_spectrogram(
    spec: torch.Tensor,
    plot_log: bool,
    mel: bool = False,
    title: Union[str, List[str], None] = None,
    ax=None,
):
    if spec.shape[0] > 1:
        if isinstance(title, str):
            title = [title] * spec.shape[0]
        if ax is None:
            _, ax = plt.subplots(spec.shape[0], 1)

        ax = ax.flatten()
        assert len(ax) == spec.shape[0]
        for s, a, t in zip(spec, ax, title):
            plot_spectrogram(
                s.squeeze().cpu(), plot_log=plot_log, mel=mel, title=t, ax=a
            )
    else:
        plot_spectrogram(
            spec.squeeze().cpu(), plot_log=plot_log, mel=mel, title=title, ax=ax
        )


class Magnitude(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs()


class Spectrogram(nn.Module):
    """Create a spectrogram from a audio signal.
    Args:
        n_fft (int, optional):
            Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional):
            Window size. (Default: ``n_fft``)
        hop_length (int or None, optional):
            Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional):
            Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional):
            A function to create a window
            tensor that is applied/multiplied to each frame/window.
            (Default: ``torch.hann_window``)
        power (float or None, optional):
            Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool or str, optional):
            Whether to normalize by magnitude after stft. If input is str,
            choices are ``"window"`` and ``"frame_length"``, if specific
            normalization type is desirable. ``True`` maps to ``"window"``.
            (Default: ``False``)
        window_kwargs (dict or None, optional):
            Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the t-th frame is centered at time
            t * hop_length.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy (Default: ``True``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = torchaudio.transforms.Spectrogram(n_fft=800)
        >>> spectrogram = transform(waveform)
    """

    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        pad: int = 0,
        power: Optional[float] = 2.0,
        normalized: Union[bool, str] = False,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        window_kwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_phase: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = (
            window_fn(self.win_length)
            if window_kwargs is None
            else window_fn(self.win_length, **window_kwargs)
        )

        self.register_buffer("window", window)
        self.hop_length = hop_length
        self.pad = pad
        self.window_fn = window_fn
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_phase = return_phase

    def plot(
        self,
        x: torch.Tensor,
        plot_log: bool = True,
        title: Optional[str] = None,
        ax=None,
    ):
        spec = self.forward(x)
        batch_plot_spectrogram(spec, plot_log=plot_log, mel=False, title=title, ax=ax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        x_stft, x_phase = stft(
            x,
            window=self.window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad=self.pad,
            power=self.power,
            normalized=self.normalized,
            center=self.center,
            pad_mode=self.pad_mode,
            onesided=self.onesided,
        )

        if self.return_phase:
            return x_stft, x_phase
        return x_stft


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        window_kwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        return_phase: bool = False,
    ):
        super().__init__()
        if f_max is not None and f_max > (sample_rate // 2):
            raise Exception("f_max is above the nyquist frequency (sample_rate / 2")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.normalized = normalized
        self.power = power
        self.return_phase = return_phase
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            window_kwargs=window_kwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
            return_phase=True,
        )
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            norm,
            mel_scale,
        )

    def plot(
        self,
        x: torch.Tensor,
        plot_log: bool = True,
        title: Optional[str] = None,
        ax=None,
    ):
        spec = self.forward(x)
        batch_plot_spectrogram(spec, plot_log=plot_log, mel=True, title=title, ax=ax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram, x_phase = self.spectrogram(x)
        # calling contiguous fun to make torchscript work correctly
        mel_specgram = self.mel_scale(specgram.contiguous())
        if self.return_phase:
            return mel_specgram, x_phase
        return mel_specgram
