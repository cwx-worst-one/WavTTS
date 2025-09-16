import io
import math
import random
from typing import Callable, List, Optional, Tuple, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad
from torchaudio.functional import loudness as torchaudio_loudness
from torchaudio.transforms import MelScale

from samantha.data.audio_utils import convert_audio, normalize_audio

MIN_LOUDNESS = -70
GAIN_FACTOR = np.log(10) / 20


def safe_log(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(x + eps)


def safe_cliplog(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


def safe_log10(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log10(x + eps)


def fp32_to_int16(audio: torch.Tensor) -> torch.Tensor:
    max_val = torch.iinfo(torch.int16).max
    return (audio * max_val).to(torch.int16)


def to_tensor(x: np.ndarray, mono=True):
    if torch.is_tensor(x):
        return x
    if isinstance(x, bytes):
        x, _ = librosa.load(io.BytesIO(x), mono=mono, sr=None)
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
    def __init__(self, mono=True) -> None:
        self.mono = mono

    def __call__(self, x):
        """Converts numpy array to torch.Tensor

        Args:
            x (np.ndarray): Audio array

        Returns:
            torch.Tensor: Audio tensor
        """
        return to_tensor(x, mono=self.mono)


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
    if audio.shape[-1] == n_samples:
        return audio

    if (start_idx + n_samples) > audio.shape[-1]:
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

    def forward(self, x, return_sample_idx: bool = False):
        max_samples = x.shape[-1]
        rand_idx = get_random_idx(max_samples - self.n_samples)

        x = crop_1d(x, rand_idx, self.n_samples)
        if return_sample_idx:
            return x, rand_idx
        return x


class NormalizeAudioToFloat32:
    def __init__(self):
        pass

    def __call__(self, x):
        return normalize_audio_to_float32(x)


class Pad(nn.Module):
    def __init__(self, n_samples: int, value: Union[int, float] = 0.0) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] >= self.n_samples:
            return x

        return pad(x, (0, self.n_samples - x.shape[-1]), value=self.value)


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


class Mono(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            return x
        if x.ndim == 2:
            return x.mean(dim=0, keepdim=True)
        elif x.ndim == 3:
            return x.mean(dim=1, keepdim=True)


class Normalize(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        loudness_headroom_db: float,
        strategy: str = "loudness",
        peak_clip_headroom_db: float = 1.0,
        rms_headroom_db: float = 18,
        loudness_compressor: bool = False,
        log_clipping: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.loudness_headroom_db = loudness_headroom_db
        self.strategy = strategy
        self.peak_clip_headroom_db = peak_clip_headroom_db
        self.rms_headroom_db = rms_headroom_db
        self.loudness_compressor = loudness_compressor
        self.log_clipping = log_clipping

    def forward(self, x: Tensor) -> Tensor:
        return normalize_audio(
            x,
            normalize=True,
            strategy=self.strategy,
            peak_clip_headroom_db=self.peak_clip_headroom_db,
            rms_headroom_db=self.rms_headroom_db,
            loudness_headroom_db=self.loudness_headroom_db,
            loudness_compressor=self.loudness_compressor,
            log_clipping=self.log_clipping,
            sample_rate=self.sample_rate,
        )


def ensure_max_of_audio(x: Tensor, max: float = 1.0, raise_exception: bool = False):
    """Ensures that ``abs(audio_data) <= max``.

    Parameters
    ----------
    max : float, optional
        Max absolute value of signal, by default 1.0

    Returns
    -------
    AudioSignal
        Signal with values scaled between -max and max.
    """

    peak = x.abs().max(dim=-1, keepdims=True)[0]

    if raise_exception and peak.max() > max:
        raise Exception(f"Audio exceeded limit: {peak.max()}")

    peak_gain = torch.ones_like(peak)
    peak_gain[peak > max] = max / peak[peak > max]
    return x * peak_gain


class RescaleAudio(nn.Module):
    def __init__(self, max: float = 1.0, raise_exception: bool = False):
        super().__init__()
        self.max = max
        self.raise_exception = raise_exception

    def forward(self, x: Tensor) -> Tensor:
        return ensure_max_of_audio(
            x, max=self.max, raise_exception=self.raise_exception
        )


class ConvertAudio(nn.Module):
    def __init__(self, src_sample_rate: int, target_sample_rate: int, n_channels: int):
        super().__init__()
        self.src_sample_rate = src_sample_rate
        self.target_sample_rate = target_sample_rate
        self.n_channels = n_channels

    def forward(self, x: Tensor) -> Tensor:
        return convert_audio(
            x, self.src_sample_rate, self.target_sample_rate, self.n_channels
        )  # TODO once stereo


class ShiftPhase(nn.Module):
    """Shifts the phase of the audio.

    Parameters
    ----------
    shift : tuple, optional
        How much to shift phase by, by default ("uniform", -np.pi, np.pi)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(self, min: float = -np.pi, max: float = np.pi):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor):
        raise NotImplementedError


class Loudness(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, sample_rate: int) -> Tensor:
        return torchaudio_loudness(x, sample_rate)


class MinMaxFilter(nn.Module):
    def __init__(self, transform: nn.Module, min: float, max: float):
        super().__init__()
        self.transform = transform
        self.min = min
        self.max = max

    def forward(self, x: Tensor, sample_rate) -> Tuple[bool, Tensor]:
        result = self.transform(x, sample_rate)
        if self.min <= result <= self.max:
            return False, result
        return True, result


class Filters(nn.Module):
    def __init__(self, filters: List[nn.Module]):
        super().__init__()
        self.filters = filters

    def forward(self, x: Tensor, sample_rate: int) -> bool:
        for filter in self.filters:
            f, result = filter(x, sample_rate)
            if f:
                return f, result
        return False, None


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.exp(torch.log(torch.clamp(x, min=clip_val) * C))


def compute_stft_padding(
    waveform: torch.Tensor, window_length: int, hop_length: int, match_stride: bool
):
    """Compute how the STFT should be padded, based on match_stride.

    Parameters
    ----------
    waveform: torch.Tensor
        Audio waveform
    window_length : int
        Window length of STFT.
    hop_length : int
        Hop length of STFT.
    match_stride : bool
        Whether or not to match stride, making the STFT have the same alignment as
        convolutional layers.

    Returns
    -------
    tuple
        Amount to pad on either side of audio.
    """

    length = waveform.shape[-1]

    if match_stride:
        assert (
            hop_length == window_length // 4
        ), "For match_stride, hop must equal n_fft // 4"
        right_pad = math.ceil(length / hop_length) * hop_length - length
        pad = (window_length - hop_length) // 2
    else:
        right_pad = 0
        pad = 0

    return right_pad, pad


def _get_spec_norms(normalized: Union[str, bool]):
    """
    NOTE: this is altered to be compatible with torch.compile()
    `torch.jit.isinstance` is replaced with `isinstance`
    """
    frame_length_norm, window_norm = False, False
    if isinstance(normalized, str):
        if normalized not in ["frame_length", "window"]:
            raise ValueError("Invalid normalized parameter: {}".format(normalized))
        if normalized == "frame_length":
            frame_length_norm = True
        elif normalized == "window":
            window_norm = True
    elif isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError("Input type not supported")
    return frame_length_norm, window_norm


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
    return_phase: bool = False,
    match_stride: bool = False,  # align with nn.Conv
):
    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    if match_stride:
        right_pad, pad = compute_stft_padding(
            waveform, win_length, hop_length, match_stride
        )
        waveform = torch.nn.functional.pad(waveform, (pad, pad + right_pad), pad_mode)

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

    if return_phase:
        x_phs = torch.angle(x_stft)
    else:
        x_phs = None

    if window_norm:
        x_stft /= window.pow(2.0).sum().sqrt()

    if power is not None:
        if power == 1.0:
            x_stft = x_stft.abs()
        else:
            x_stft = x_stft.abs().pow(power)

    if match_stride:
        # TODO: check
        # Drop first two and last two frames, which are added
        # because of padding. Now num_frames * hop_length = num_samples.
        x_stft = x_stft[..., 2:-2]

    return x_stft, x_phs


def istft(
    spectrogram: torch.Tensor,
    length: Optional[int],
    pad: int,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float,
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
) -> torch.Tensor:
    r"""Create an inverse spectrogram or a batch of inverse spectrograms from the provided
    complex-valued spectrogram.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        spectrogram (Tensor): Complex tensor of audio of dimension (..., freq, time).
        length (int or None): The output length of the waveform.
        pad (int): Two sided padding of signal. It is only effective when ``length`` is provided.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        normalized (bool or str): Whether the stft output was normalized by magnitude. If input is str, choices are
            ``"window"`` and ``"frame_length"``, dependent on normalization mode. ``True`` maps to
            ``"window"``.
        center (bool, optional): whether the waveform was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. This parameter is provided for compatibility with the
            spectrogram function and is not used. Default: ``"reflect"``
        onesided (bool, optional): controls whether spectrogram was done in onesided mode.
            Default: ``True``

    Returns:
        Tensor: Dimension `(..., time)`. Least squares estimation of the original signal.
    """

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    if not spectrogram.is_complex():
        raise ValueError("Expected `spectrogram` to be complex dtype.")

    if window_norm:
        spectrogram = spectrogram * window.pow(2.0).sum().sqrt()

    # pack batch
    shape = spectrogram.size()
    spectrogram = spectrogram.reshape(-1, shape[-2], shape[-1])

    if power is not None:
        spectrogram = spectrogram.pow(1 / power)

    # default values are consistent with librosa.core.spectrum._spectrogram
    waveform = torch.istft(
        input=spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=frame_length_norm,
        onesided=onesided,
        length=length + 2 * pad if length is not None else None,
        return_complex=False,
    )

    if length is not None and pad > 0:
        # remove padding from front and back
        waveform = waveform[:, pad:-pad]

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform


def plot_spectrogram(
    spec: torch.Tensor,
    plot_log: bool,
    mel: bool = False,
    title: Optional[str] = None,
    ax=None,
    figsize=None,
    imshow_kwargs={},
):
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    if title is not None:
        ax.set_title(title)

    ylabel = "bin"
    ax.set_ylabel(ylabel)

    if plot_log:
        spec = spec.log()

    ax.imshow(
        spec, origin="lower", aspect="auto", interpolation="nearest", **imshow_kwargs
    )


def batch_plot_spectrogram(
    spec: torch.Tensor,
    plot_log: bool,
    mel: bool = False,
    title: Union[str, List[str]] = "",
    ax=None,
    figsize=None,
    grid=None,
    imshow_kwargs={},
):
    if spec.shape[0] > 1:
        if isinstance(title, str):
            title = [title] * spec.shape[0]
        if ax is None:
            if grid is None:
                grid = (spec.shape[0], 1)
            _, ax = plt.subplots(grid[0], grid[1], figsize=figsize)

        ax = ax.flatten()
        assert len(ax) == spec.shape[0]
        for s, a, t in zip(spec, ax, title):
            plot_spectrogram(
                s.cpu(),
                plot_log=plot_log,
                mel=mel,
                title=t,
                ax=a,
                imshow_kwargs=imshow_kwargs,
            )
    else:
        if ax is None:
            _, ax = plt.subplots(1, figsize=figsize)
        plot_spectrogram(
            spec[0].cpu(),
            plot_log=plot_log,
            mel=mel,
            title=title,
            ax=ax,
            imshow_kwargs=imshow_kwargs,
        )


def plot_amplitude_histogram(
    x: torch.Tensor,
    bins: int,
    title: Optional[str] = None,
    ax=None,
    figsize=None,
    imshow_kwargs: dict = {},
):
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    if title is not None:
        ax.set_title(title)

    min_val = x.min()
    max_val = x.max()

    hist, density = torch.histogram(
        x,
        bins=bins,
        range=(min_val.item(), max_val.item()),
        weight=None,
        density=True,
        out=None,
    )

    xlabel = torch.linspace(min_val, max_val, bins)

    ax.bar(range(bins), hist.cpu(), align="center", **imshow_kwargs)
    ax.set_xticks(range(bins), [f"{l:.4f}" for l in xlabel.tolist()])
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    return ax


def batch_plot_amplitude_histogram(
    x: torch.Tensor,
    bins: int,
    title: Union[str, List[str]] = "",
    ax=None,
    figsize=None,
    grid=None,
    imshow_kwargs={},
):
    if x.shape[0] > 1:
        if isinstance(title, str):
            title = [title] * x.shape[0]
        if ax is None:
            if grid is None:
                grid = (x.shape[0], 1)
            _, ax = plt.subplots(grid[0], grid[1], figsize=figsize)

        ax = ax.flatten()
        assert len(ax) == x.shape[0]
        for s, a, t in zip(x, ax, title):
            plot_amplitude_histogram(
                s.cpu(), bins, title=t, ax=a, imshow_kwargs=imshow_kwargs
            )
    else:
        if ax is None:
            _, ax = plt.subplots(1, figsize=figsize)
        plot_amplitude_histogram(
            x[0].cpu(), bins, title=title, ax=ax, imshow_kwargs=imshow_kwargs
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
        match_stride: bool = False,
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
        self.match_stride = match_stride

    @property
    def n_fft_bins(self) -> int:
        return self.n_fft // 2 + 1

    def plot(
        self,
        x: torch.Tensor,
        plot_log: bool = True,
        title: Optional[str] = None,
        ax=None,
    ):
        spec = self.forward(x)
        batch_plot_spectrogram(spec, plot_log=plot_log, mel=False, title=title, ax=ax)

    def inverse(self, spec: torch.Tensor, phase: torch.Tensor):
        imag = spec * torch.sin(phase)
        real = spec * torch.cos(phase)
        spec_complex = torch.complex(real, imag)
        return istft(
            spec_complex,
            length=None,
            pad=self.pad,
            window=self.window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=self.power,
            normalized=self.normalized,
        )

    def forward(self, x: torch.Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
            return_phase=self.return_phase,
            match_stride=self.match_stride,
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
        self.win_length = self.win_length = (
            win_length if win_length is not None else n_fft
        )
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
            return_phase=self.return_phase,
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

    def inverse(self, mel: torch.Tensor, driver: str = "gels"):
        fb = self.mel_scale.fb

        shape = mel.size()

        mel = mel.view(-1, shape[-2], shape[-1])

        time = shape[-1]
        freq, _ = fb.size()  # (freq, n_mels)

        specgram = torch.relu(
            torch.linalg.lstsq(fb.transpose(-1, -2)[None], mel, driver=driver).solution
        )
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        if self.return_phase:
            specgram, x_phase = self.spectrogram(x)
            return self.mel_scale(specgram), x_phase
        else:
            specgram = self.spectrogram(x)
            return self.mel_scale(specgram)


def generate_sine(
    sample_rate: int,
    duration: float,
    freq: float,
    amplitude: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    t = torch.linspace(0, duration, int(duration * sample_rate), device=device)
    t = t[None, :]
    return amplitude * torch.sin(2 * np.pi * freq * t)


def generate_silence(
    sample_rate: int, duration: float, device: torch.device = torch.device("cpu")
):
    return generate_sine(sample_rate, duration, freq=0, amplitude=0, device=device)
