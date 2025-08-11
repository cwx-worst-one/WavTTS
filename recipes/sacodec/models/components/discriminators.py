from typing import List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Spectrogram


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator module adapted from https://github.com/jik876/hifi-gan.
    Additionally, it allows incorporating conditional information with a learned embeddings table.

    Args:
        periods (tuple[int]): Tuple of periods for each discriminator.
        num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
            Defaults to None.
    """

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11), num_embeddings: Optional[int] = None, in_channels: int = 1, flatten_stereo=False):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(period=p, num_embeddings=num_embeddings, in_channels=in_channels, flatten_stereo=flatten_stereo) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        lrelu_slope: float = 0.1,
        num_embeddings: Optional[int] = None,
        flatten_stereo: bool = False
    ):
        super().__init__()
        self.period = period
        self.flatten_stereo = flatten_stereo
        if flatten_stereo:
            in_channels = 1
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv2d(in_channels, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(kernel_size // 2, 0))),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=1024)
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu_slope = lrelu_slope

    def preprocess(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return x

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.preprocess(x.float()) # added from soundstream
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        # Try batch by channel to make it harder for discriminator
        if self.flatten_stereo:
            x = x.view(b * c, 1, t // self.period, self.period)
        else:
            x = x.view(b, c, t // self.period, self.period)

        for i, l in enumerate(self.convs):
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            # if i > 0:
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (2048, 1024, 512),
        num_embeddings: Optional[int] = None,
        in_channels: int = 1,
        channels: int = 32,
        flatten_stereo: bool = False
    ):
        """
        Multi-Resolution Discriminator module adapted from https://github.com/descriptinc/descript-audio-codec.
        Additionally, it allows incorporating conditional information with a learned embeddings table.

        Args:
            fft_sizes (tuple[int]): Tuple of window lengths for FFT. Defaults to (2048, 1024, 512).
            num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
                Defaults to None.
        """

        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(window_length=w, num_embeddings=num_embeddings, in_channels=in_channels, channels=channels, flatten_stereo=flatten_stereo) for w in fft_sizes]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        window_length: int,
        num_embeddings: Optional[int] = None,
        in_channels: int = 1,
        channels: int = 32, # apcodec uses 64
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
        flatten_stereo: bool = False
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(
            n_fft=window_length, hop_length=int(window_length * hop_factor), win_length=window_length, power=None
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        self.flatten_stereo = flatten_stereo
        if flatten_stereo:
            in_channels = 1
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2*in_channels, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def preprocess(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return x

    def spectrogram(self, x):
        x = self.preprocess(x)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        assert len(x.shape) == 5
        if self.flatten_stereo:
            x = rearrange(x, "b ca f t c -> (b ca) c t f") # ca = audio channels
        else:
            x = rearrange(x, "b ca f t c -> b (ca c) t f") # ca = audio channels
        # x = rearrange(x, "b f t c -> b c t f") # non-channel stereo case

        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
        with torch.autocast(device_type="cuda", enabled=False):
            x_bands = self.spectrogram(x.float())
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                # if i > 0:
                fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h

        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    """
    Combined Period and Resolution Discriminators
    """
    def __init__(
        self,
        # MultiPeriodDiscriminator params
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        # MultiResolutionDiscriminator params
        fft_sizes: Tuple[int, ...] = (2048, 1024, 512),
        # shared
        num_embeddings: Optional[int] = None,
        in_channels: int = 1,
    ):
        self.mpd = MultiPeriodDiscriminator(periods=periods, num_embeddings=num_embeddings, in_channels=in_channels)
        self.mrd = MultiResolutionDiscriminator(fft_sizes=fft_sizes, num_embeddings=num_embeddings, in_channels=in_channels)

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        # y_d_rs, y_d_gs, fmap_rs, fmap_gs 
        mpd_res = self.mpd(y, y_hat, bandwidth_id)
        mrd_res = self.mrd(y, y_hat, bandwidth_id)
        
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = (p + r for p, r in zip(mpd_res, mrd_res))
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs