import julius
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from recipes.soundstream.models.modules.mrd import MultiResolutionDiscriminator
from recipes.soundstream.models.modules.pqmf import PQMF
from recipes.soundstream.models.modules.resblocks import SineGen
from recipes.soundstream.utils.utils import get_padding, init_weights

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self, period, kernel_size=5, stride=3, use_spectral_norm=False, fmap_depth=0
    ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.fmap_depth = fmap_depth
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
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
    def __init__(self, use_spectral_norm=False, fmap_depth=0):
        super(DiscriminatorS, self).__init__()
        self.fmap_depth = fmap_depth
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 512, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(512, 512, 5, 1, padding=2)),
            ]
        )
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


class Discriminator(nn.Module):
    def __init__(self, hp, use_spectral_norm=False):
        super().__init__()
        self.hp = hp
        fmap_depth = hp.fmap_depth
        self.mpds = nn.ModuleList(
            [
                DiscriminatorP(2, fmap_depth=fmap_depth),
                DiscriminatorP(3, fmap_depth=fmap_depth),
                DiscriminatorP(5, fmap_depth=fmap_depth),
                DiscriminatorP(7, fmap_depth=fmap_depth),
                DiscriminatorP(11, fmap_depth=fmap_depth),
            ]
        )
        # self.mrd = MultiResolutionDiscriminator(hp)
        self.msd = DiscriminatorS(use_spectral_norm=False, fmap_depth=fmap_depth)

    def forward(self, y, y_hat, f0s=None):
        y_org = y
        y_hat_org = y_hat

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # mpds
        for i, d in enumerate(self.mpds):
            y_d_r, fmap_r = d(y_org)
            y_d_g, fmap_g = d(y_hat_org)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        """# mrds
        res_r = self.mrd(y_org)
        res_g = self.mrd(y_hat_org)
        for score, fmap in res_r:
            y_d_rs.append(score)
            fmap_rs.append(fmap)
        for score, fmap in res_g:
            y_d_gs.append(score)
            fmap_gs.append(fmap)
        """
        # msd single layer
        y_d_r, fmap_r = self.msd(y_org)
        y_d_g, fmap_g = self.msd(y_hat_org)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
