from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchaudio.functional.functional import _get_spec_norms
from torchaudio.transforms import MelScale


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
        _, ax = plt.subplots(spec.shape[0], 1)

    if title is not None:
        ax.set_title(title)

    ylabel = "mel bin" if mel else "Hz"
    ax.set_ylabel(ylabel)

    if plot_log:
        spec = spec.log()

    ax.imshow(spec, origin="lower", aspect="auto", interpolation="nearest")


def batch_plot_spectrogram(
    spec: torch.Tensor,
    plot_log: bool,
    mel: bool = False,
    title: Optional[str] = None,
    ax=None,
):
    if spec.shape[0] > 1:
        ax = ax.flatten()
        assert len(ax) == spec.shape[0]
        for a in ax:
            plot_spectrogram(spec.cpu(), plot_log=plot_log, mel=mel, title=title, ax=a)
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
        mel_specgram = self.mel_scale(specgram)
        if self.return_phase:
            return mel_specgram, x_phase
        return mel_specgram
