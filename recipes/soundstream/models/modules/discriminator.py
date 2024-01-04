# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MS-STFT discriminator, provided here for reference."""

import typing as tp

import torch
import torchaudio
from einops import rearrange
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import weight_norm

from recipes.soundstream.models.modules.pqmf import PQMF

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


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

def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )

class MPD(nn.Module):
    def __init__(
        self, 
        period,
        in_channels: int = 1,
        adapt_hopper: bool = True,
    ):
        super().__init__()
        self.period = period
        if adapt_hopper:
            self.convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(0, 0, 2, 2), value=0),
                        WNConv2d(in_channels, 32, (5, 1), (3, 1)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(0, 0, 2, 2), value=0),
                        WNConv2d(32, 128, (5, 1), (3, 1)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(0, 0, 2, 2), value=0),
                        WNConv2d(128, 512, (5, 1), (3, 1)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(0, 0, 2, 2), value=0),
                        WNConv2d(512, 1024, (5, 1), (3, 1)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(0, 0, 2, 2), value=0),
                        WNConv2d(1024, 1024, (5, 1), 1),
                    ),
                ]
            )

            self.conv_post = nn.Sequential(
                nn.ConstantPad2d(padding=(0, 0, 1, 1), value=0),
                WNConv2d(1024, 1, kernel_size=(3, 1), act=False),
            )
        else:
            self.convs = nn.ModuleList(
                [
                    WNConv2d(in_channels, 32, (5, 1), (3, 1), padding=(2, 0)),
                    WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                    WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                    WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                    WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
                ]
            )
            self.conv_post = WNConv2d(
                1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
            )

        dicts = {
            2: {"subbands": 2, "taps": 62, "cutoff_ratio": 0.26699457, "beta": 9.0},
            3: {"subbands": 3, "taps": 62, "cutoff_ratio": 0.18366124, "beta": 9.0},
            5: {"subbands": 5, "taps": 72, "cutoff_ratio": 0.11463421, "beta": 9.0},
            7: {"subbands": 7, "taps": 82, "cutoff_ratio": 0.08427813, "beta": 9.0},
            11: {"subbands": 11, "taps": 92, "cutoff_ratio": 0.05690741, "beta": 9.0},
        }
        self.pqmf = PQMF(**dicts[self.period])

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        x = rearrange(x, "b c t -> (b c) 1 t")
        x = self.pqmf(x)  # [B, D, T]
        x = rearrange(x, "(b c) d t -> b c d t", c=c)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        # put final output in fmap
        fmap.append(x)

        return fmap

BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
class MRD(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        filters: int = 48, 
        win_length: int = 512,
        hop_factor: float = 0.25,
        bands: list = BANDS,
        adapt_hopper: bool = True,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        win_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * win_length``.
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.win_length = win_length
        self.hop_factor = hop_factor
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=win_length,
            hop_length=int(win_length * hop_factor),
            win_length=win_length,
            window_fn=torch.hann_window,
            center=True,
            pad_mode='reflect',
            power=None,
        )

        n_fft = win_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        if adapt_hopper:
            convs = lambda: nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(4, 4, 1, 1), value=0),
                        WNConv2d(2*in_channels, filters, (3, 9), (1, 1)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(4, 4, 1, 1), value=0),
                        WNConv2d(filters, filters, (3, 9), (1, 2)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(4, 4, 1, 1), value=0),
                        WNConv2d(filters, filters, (3, 9), (1, 2)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(4, 4, 1, 1), value=0),
                        WNConv2d(filters, filters, (3, 9), (1, 2)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0),
                        WNConv2d(filters, filters, (3, 3), (1, 1)),
                    ),
                ]
            )
        else:
            convs = lambda: nn.ModuleList(
                [
                    WNConv2d(2*in_channels, filters, (3, 9), (1, 1), padding=(1, 4)),
                    WNConv2d(filters, filters, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(filters, filters, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(filters, filters, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(filters, filters, (3, 3), (1, 1), padding=(1, 1)),
                ]
            )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        if adapt_hopper:
            self.conv_post = nn.Sequential(
                nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0),
                WNConv2d(filters, 1, (3, 3), (1, 1), act=False),
            )
        else:
            self.conv_post = WNConv2d(filters, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            x_stft = torch.view_as_real(self.spec_transform(x))
            x_stft = rearrange(x_stft, "b c f t cp -> b (c cp) t f")
        x_stft = x_stft.to(x.dtype)

        x_bands = [x_stft[..., b[0] : b[1]] for b in self.bands]
        
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        # put final output in fmap
        fmap.append(x)

        return fmap

class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        periods: tp.List = [2, 3, 5, 7, 11],
        n_ffts: tp.List[int] = [2048, 512, 128],
        hop_lengths: tp.List[int] = [512, 128, 32],
        win_lengths: tp.List[int] = [1024, 256, 64],
        adapt_hopper: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.mrds = nn.ModuleList(
            [
                MRD(
                    in_channels=in_channels,
                    filters=filters,
                    win_length=win_lengths[i],
                    adapt_hopper=adapt_hopper,
                )
                for i in range(len(n_ffts))
            ]
        )
        self.mpds = nn.ModuleList([MPD(p, in_channels, adapt_hopper) for p in periods])

    def preprocess(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return x

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, return_scales=False
    ) -> DiscriminatorOutput:
        logit_rs, logit_gs, fmap_rs, fmap_gs = [], [], [], []
        scales = []

        x = self.preprocess(x)
        y = self.preprocess(y)

        for mrd in self.mrds:
            fmap_r = mrd(x)
            fmap_g = mrd(y)

            logit_rs.append(fmap_r[-1])
            logit_gs.append(fmap_g[-1])
            fmap_rs.append(fmap_r[:-1])
            fmap_gs.append(fmap_g[:-1])

        for mpd in self.mpds:
            fmap_r = mpd(x)
            fmap_g = mpd(y)

            logit_rs.append(fmap_r[-1])
            logit_gs.append(fmap_g[-1])
            fmap_rs.append(fmap_r[:-1])
            fmap_gs.append(fmap_g[:-1])

        return logit_rs, logit_gs, fmap_rs, fmap_gs


def test():
    disc = MultiScaleSTFTDiscriminator(filters=32)
    y = torch.randn(1, 1, 24000)
    y_hat = torch.randn(1, 1, 24000)

    y_disc_r, fmap_r = disc(y)
    y_disc_gen, fmap_gen = disc(y_hat)
    assert (
        len(y_disc_r)
        == len(y_disc_gen)
        == len(fmap_r)
        == len(fmap_gen)
        == disc.num_discriminators
    )

    assert all([len(fm) == 5 for fm in fmap_r + fmap_gen])
    assert all([list(f.shape)[:2] == [1, 32] for fm in fmap_r + fmap_gen for f in fm])
    assert all([len(logits.shape) == 4 for logits in y_disc_r + y_disc_gen])


if __name__ == "__main__":
    test()
