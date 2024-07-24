from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal, STFTParams
from einops.layers.torch import Rearrange
from julius.resample import ResampleFrac
from torch.nn.utils import weight_norm

from samantha.models.base import LightningModuleBase


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):
    def __init__(self, period: int, n_channels: int):
        super().__init__()
        self.period = period
        self.rearrange_period = Rearrange("b c (l p) -> b c l p", p=self.period)
        self.convs = nn.ModuleList(
            [
                WNConv2d(n_channels, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        # TODO: output only mono for now
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

        # dicts = {
        #     2: {"subbands": 2, "taps": 62, "cutoff_ratio": 0.26699457, "beta": 9.0},
        #     3: {"subbands": 3, "taps": 62, "cutoff_ratio": 0.18366124, "beta": 9.0},
        #     5: {"subbands": 5, "taps": 72, "cutoff_ratio": 0.11463421, "beta": 9.0},
        #     7: {"subbands": 7, "taps": 82, "cutoff_ratio": 0.08427813, "beta": 9.0},
        #     11: {"subbands": 11, "taps": 92, "cutoff_ratio": 0.05690741, "beta": 9.0},
        # }
        # self.pqmf = PQMF(**dicts[self.period])

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)  # TODO: necessary?
        x = self.rearrange_period(x)  # NOTE: already done
        # b, c, t = x.shape
        # x = rearrange(x, "b c t -> (b c) 1 t")
        # x = self.pqmf(x)  # [B, D, T]
        # x = rearrange(x, "(b c) d t -> b c d t", c=c)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int, n_channels: int, sample_rate: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(n_channels, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        # TODO: output 1 channel for now
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate
        self.resample_fn = ResampleFrac(self.sample_rate, self.sample_rate // self.rate)

    def forward(self, x):
        x = self.resample_fn(x)
        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        n_channels: int,
        sample_rate: int,
        hop_factor: float = 0.25,
        n_filters: int = 32,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()
        self.n_channels = n_channels
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate

        # BEFORE:
        # self.spectrogram = Spectrogram(
        #     n_fft=window_length,
        #     win_length=window_length,
        #     hop_length=int(window_length * hop_factor),
        #     match_stride=True,
        #     power=None,  # get the complex-valued spectrogram
        #     return_phase=False,
        # )
        # AFTER:
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        self.rearrange_spec = Rearrange("b c f t real_imag -> b (c real_imag) t f")

        n_fft = window_length // 2 + 1
        # BEFORE
        # bands = [
        #     (
        #         int(b[0] * self.spectrogram.n_fft_bins),
        #         int(b[1] * self.spectrogram.n_fft_bins),
        #     )
        #     for b in bands
        # ]

        # AFTER
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        self.frames_per_band = [b[1] - b[0] for b in self.bands]
        # assert sum(self.frames_per_band) == self.spectrogram.n_fft_bins

        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2 * n_channels, n_filters, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(n_filters, n_filters, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(n_filters, n_filters, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(n_filters, n_filters, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(n_filters, n_filters, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        # TODO: output 1 channel for now
        self.conv_post = WNConv2d(
            n_filters, 1, (3, 3), (1, 1), padding=(1, 1), act=False
        )

    def split_bands(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Split into bands

        ## the following has equivalent logic:
        # return torch.split(x, self.frames_per_band, dim=-1)
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def audio_to_spectrogram_bands(self, x):
        # BEFORE:
        # x = self.spectrogram(x)

        # AFTER:
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = x.stft()

        x = torch.view_as_real(x)

        # arrange the real/imaginary parts to the channel dim
        x = self.rearrange_spec(x)
        return self.split_bands(x)

    def forward(self, x):
        x_bands = self.audio_to_spectrogram_bands(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


@dataclass
class DiscriminatorConfig:
    sample_rate: int = 44100
    n_channels: int = 1
    n_filters: int = 32
    rates: Tuple[int] = ()
    periods: Tuple[int] = (2, 3, 5, 7, 11)
    window_lengths: Tuple[int] = (2048, 1024, 512)
    bands: Tuple[Tuple[float, float]] = (
        (0.0, 0.1),
        (0.1, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
    )


class Discriminator(LightningModuleBase):
    def __init__(self, config: DiscriminatorConfig):
        """Discriminator that combines multiple discriminators.

        From the paper:

        Like prior work, we use multi-scale (MSD) and multi-period waveform discriminators (MPD)
        which lead to improved audio fidelity. However, spectrograms of generated audio can still appear
        blurry, exhibiting over-smoothing artifacts in high frequencies[16]. The multi-resolution spectrogram
        discriminator (MRSD) was proposed in UnivNet to fix these artifacts and BigVGAN [21] found
        that it also helps to reduce pitch and periodicity artifacts. However, using magnitude spectrograms
        discards phase information which could’ve been otherwise utilized by the discriminator to penalize
        phase modeling errors. Moreover, we find that high-frequency modeling is still challenging for these
        models especially at high sampling rates.

        To address these issues, we use a complex STFT discriminator [46] at multiple time-scales [8] and
        find that it works better in practice and leads to improved phase modeling. Additionally we find
        that splitting the STFT into sub-bands slightly improves high frequency prediction and mitigates
        aliasing artifacts, since the discriminator can learn discriminative features about a specific sub-band
        and provide a stronger gradient signal to the generator. Multi-band processing was earlier proposed
        in [43] to predict audio in sub-bands which are subsequently summed to produce the full-band audio


        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        window_lengths : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MPD(p, n_channels=config.n_channels) for p in config.periods]
        discs += [
            MSD(r, n_channels=config.n_channels, sample_rate=config.sample_rate)
            for r in config.rates
        ]
        discs += [
            MRD(
                w,
                n_channels=config.n_channels,
                sample_rate=config.sample_rate,
                n_filters=config.n_filters,
                bands=config.bands,
            )
            for w in config.window_lengths
        ]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        """Preprocesses the audio relevant for the spectral discriminators:
        Remove DC offset
        Peak normalize the volume of input audio (lower headroom)
        """
        y = y - y.mean(dim=-1, keepdims=True)
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


if __name__ == "__main__":
    n_channels = 2
    disc = Discriminator(DiscriminatorConfig(n_channels=n_channels))
    x = torch.zeros(1, n_channels, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
