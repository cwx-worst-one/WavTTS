import typing as tp
import torchaudio
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d

from recipes.umm.vocoder.pqmf import PQMF


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]
LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, fmap_depth=0):
        super().__init__()
        self.period = period
        self.fmap_depth = fmap_depth
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(1,
                       32, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(32,
                       128, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(128,
                       512, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(512,
                       1024, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        dicts = {
            2: {
                "subbands": 2,
                "taps": 62,
                "cutoff_ratio": 0.26699457,
                "beta": 9.0
            },
            3: {
                "subbands": 3,
                "taps": 62,
                "cutoff_ratio": 0.18366124,
                "beta": 9.0
            },
            5: {
                "subbands": 5,
                "taps": 72,
                "cutoff_ratio": 0.11463421,
                "beta": 9.0
            },
            7: {
                "subbands": 7,
                "taps": 82,
                "cutoff_ratio": 0.08427813,
                "beta": 9.0
            },
            11: {
                "subbands": 11,
                "taps": 92,
                "cutoff_ratio": 0.05690741,
                "beta": 9.0
            }
        }
        self.pqmf = PQMF(**dicts[self.period])

    def forward(self, x):
        fmap = []
        x = self.pqmf(x)  # [B, D, T]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            if i >= self.fmap_depth:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(torch.nn.Module):

    def __init__(self, fmap_depth=0):
        super().__init__()
        self.fmap_depth = fmap_depth
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 512, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(512, 512, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(512, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            if i >= self.fmap_depth:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class NormConv2d(nn.Module):

    def __init__(self,
                 *args,
                 norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs),
                                               norm)
        self.norm = get_norm_module(self.conv,
                                    causal=False,
                                    norm=norm,
                                    **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def apply_parametrization_norm(module: nn.Module,
                               norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        return module


CONV_NORMALIZATIONS = frozenset([
    'none', 'weight_norm', 'spectral_norm', 'time_layer_norm', 'layer_norm',
    'time_group_norm'
])


def get_norm_module(module: nn.Module,
                    causal: bool = False,
                    norm: str = 'none',
                    **norm_kwargs) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
            assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_2d_padding(kernel_size: tp.Tuple[int, int],
                   dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):

    def __init__(self,
                 filters: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 max_filters: int = 1024,
                 filters_scale: int = 1,
                 kernel_size: tp.Tuple[int, int] = (3, 9),
                 dilations: tp.List = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 2),
                 normalized: bool = True,
                 norm: str = 'weight_norm',
                 activation: str = 'LeakyReLU',
                 activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels,
                       self.filters,
                       kernel_size=kernel_size,
                       padding=get_2d_padding(kernel_size)))
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale**(i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(in_chs,
                           out_chs,
                           kernel_size=kernel_size,
                           stride=stride,
                           dilation=(dilation, 1),
                           padding=get_2d_padding(kernel_size, (dilation, 1)),
                           norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale**(len(dilations) + 1)) * self.filters,
                      max_filters)
        self.convs.append(
            NormConv2d(in_chs,
                       out_chs,
                       kernel_size=(kernel_size[0], kernel_size[0]),
                       padding=get_2d_padding(
                           (kernel_size[0], kernel_size[0])),
                       norm=norm))
        self.conv_post = NormConv2d(out_chs,
                                    self.out_channels,
                                    kernel_size=(kernel_size[0],
                                                 kernel_size[0]),
                                    padding=get_2d_padding(
                                        (kernel_size[0], kernel_size[0])),
                                    norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):

    def __init__(self,
                 filters: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_ffts: tp.List[int] = [2048, 512, 128],
                 hop_lengths: tp.List[int] = [256, 128, 32],
                 win_lengths: tp.List[int] = [1024, 256, 64],
                 pretrain=False,
                 **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.pretrain = pretrain
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              n_fft=n_ffts[i],
                              win_length=win_lengths[i],
                              hop_length=hop_lengths[i],
                              **kwargs) for i in range(len(n_ffts))
        ])
        self.mpds = nn.ModuleList([
            DiscriminatorP(2, fmap_depth=2),
            DiscriminatorP(3, fmap_depth=2),
            DiscriminatorP(5, fmap_depth=2),
        ])
        if not self.pretrain:
            self.msds = nn.ModuleList([
                DiscriminatorS(fmap_depth=2),
            ])
            self.meanpool = AvgPool1d(4, 2, padding=2)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                return_scales=False) -> DiscriminatorOutput:
        logit_rs, logit_gs, fmap_rs, fmap_gs = [], [], [], []
        scales = []

        for disc in self.discriminators:
            logit_r, fmap_r = disc(x)
            logit_g, fmap_g = disc(y)

            logit_rs.append(logit_r)
            logit_gs.append(logit_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
            scales.append(1.0)

        for mpd in self.mpds:
            logit_r, fmap_r = mpd(x)
            logit_g, fmap_g = mpd(y)

            logit_rs.append(logit_r)
            logit_gs.append(logit_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
            scales.append(1.0)

        if not self.pretrain:
            for i, msd in enumerate(self.msds):
                logit_r, fmap_r = msd(x)
                logit_g, fmap_g = msd(y)
                x = self.meanpool(x)
                y = self.meanpool(y)

                logit_rs.append(logit_r)
                logit_gs.append(logit_g)
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)
                scales.append(0.1)

        if return_scales:
            return logit_rs, logit_gs, fmap_rs, fmap_gs, scales
        return logit_rs, logit_gs, fmap_rs, fmap_gs
