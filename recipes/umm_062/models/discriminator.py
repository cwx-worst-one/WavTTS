import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm

from recipes.umm_062.models.pqmf import PQMF as PQMF_PWG
from recipes.umm_062.models.vocoder import LRELU_SLOPE, get_padding

BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        dicts = {
            2: {"subbands": 2, "taps": 62, "cutoff_ratio": 0.26699457, "beta": 9.0},
            3: {"subbands": 3, "taps": 62, "cutoff_ratio": 0.18366124, "beta": 9.0},
            5: {"subbands": 5, "taps": 72, "cutoff_ratio": 0.11463421, "beta": 9.0},
            7: {"subbands": 7, "taps": 82, "cutoff_ratio": 0.08427813, "beta": 9.0},
            11: {"subbands": 11, "taps": 92, "cutoff_ratio": 0.05690741, "beta": 9.0},
        }
        self.pqmf = PQMF_PWG(**dicts[self.period])

    def forward(self, x):
        fmap = []
        x = self.pqmf.analysis(x)  # [B, D, T]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiBandDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, fmap_depth=0):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm
        self.discriminators = nn.ModuleList([DiscriminatorS(), DiscriminatorS()])
        self.full_discriminator = DiscriminatorS()
        self.pqmf = PQMF_PWG(subbands=2, taps=62, cutoff_ratio=0.26699457, beta=9.0)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # full band discriminator
        y_d_r, fmap_r = self.full_discriminator(y)
        y_d_g, fmap_g = self.full_discriminator(y_hat)
        y_d_rs.append(y_d_r)
        fmap_rs.append(fmap_r)
        y_d_gs.append(y_d_g)
        fmap_gs.append(fmap_g)

        # multi band discriminator
        y = self.pqmf.analysis(y)
        y_hat = self.pqmf.analysis(y_hat)
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y[:, i : i + 1, :])
            y_d_g, fmap_g = d(y_hat[:, i : i + 1, :])
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
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

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        win_length = self.window_length
        n_fft = win_length
        hop_length = int(self.window_length * self.hop_factor)

        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=False,
        )  # [B, F, TT, 2]
        x = rearrange(x, "b f t c -> b c t f")
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
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

        return x, fmap


class MBPRD(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, ifmbd=False):
        super(MBPRD, self).__init__()
        self.ifmbd = ifmbd
        periods = [2, 3, 5, 7, 11]
        fft_sizes = [2048, 1024, 512]
        bands = BANDS

        self.extraD = (
            MultiBandDiscriminator(use_spectral_norm=use_spectral_norm)
            if ifmbd
            else DiscriminatorS(use_spectral_norm=use_spectral_norm)
        )
        self.mpd = nn.ModuleList(
            [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        )
        self.mrd = nn.ModuleList([MRD(f, bands=bands) for f in fft_sizes])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        if self.ifmbd:
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.extraD(y, y_hat)
        else:
            y_d_r, fmap_r = self.extraD(y)
            y_d_g, fmap_g = self.extraD(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.mpd):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.mrd):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
