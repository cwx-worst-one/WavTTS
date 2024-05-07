import math
from torch import nn
import torch.nn.functional as F

from recipes.soundstream.models.modules.basic_blocks import (
    Snake1d,
    WNConv1d,
    WNConvTranspose1d,
    ResidualUnit
)

class NearestUpsample(nn.Module):
    def __init__(self, up_scale, in_channels, out_channels, adapt_hopper):
        super().__init__()
        self.up_scale = up_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adapt_hopper = adapt_hopper
        if adapt_hopper:
            self.pad = nn.ConstantPad1d(padding=up_scale - 1, value=0)
            self.conv = WNConv1d(
                in_channels,
                out_channels,
                kernel_size=2 * up_scale - 1,
            )
        else:
            self.conv = WNConv1d(
                in_channels,
                out_channels,
                kernel_size=2 * up_scale - 1,
                padding_mode='zeros',
                padding=up_scale - 1,
            )

    def forward(self, x):
        x_size = x.size(2)
        x = F.interpolate(x, scale_factor=self.up_scale, mode="nearest")
        if self.adapt_hopper:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, adapt_hopper: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            NearestUpsample(
                up_scale=stride,
                in_channels=input_dim,
                out_channels=output_dim,
                adapt_hopper=adapt_hopper,
            ),
            ResidualUnit(output_dim, dilation=1, adapt_hopper=adapt_hopper),
            ResidualUnit(output_dim, dilation=3, adapt_hopper=adapt_hopper),
            ResidualUnit(output_dim, dilation=9, adapt_hopper=adapt_hopper),
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        adapt_hopper: bool = True,
        last_act: bool = True,
    ):
        super().__init__()

        # Add first conv layer
        if adapt_hopper:
            layers = [
                nn.ConstantPad1d(padding=3, value=0),
                WNConv1d(input_channel, channels, kernel_size=7)
            ]
        else:
            layers = [
                WNConv1d(input_channel, channels, kernel_size=7, padding=3)
            ]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, adapt_hopper)]

        # Add final conv layer
        if adapt_hopper:
            layers += [
                Snake1d(output_dim),
                nn.ConstantPad1d(padding=3, value=0),
                WNConv1d(output_dim, d_out, kernel_size=7),
            ]
        else:
            layers += [
                Snake1d(output_dim),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            ]

        if last_act:
            layers += [nn.Tanh()]


        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)