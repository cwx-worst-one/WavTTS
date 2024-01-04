import math
from torch import nn

from recipes.soundstream.models.modules.basic_blocks import (
    Snake1d,
    WNConv1d,
    ResidualUnit
)

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, adapt_hopper: bool = True):
        super().__init__()

        self.block = [
            ResidualUnit(dim // 2, dilation=1, adapt_hopper=adapt_hopper),
            ResidualUnit(dim // 2, dilation=3, adapt_hopper=adapt_hopper),
            ResidualUnit(dim // 2, dilation=9, adapt_hopper=adapt_hopper),
            Snake1d(dim // 2),
        ]
        if adapt_hopper:
            self.block += [
                nn.ConstantPad1d(padding=stride, value=0),
                WNConv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride + 1,
                    stride=stride,
                ),
            ]
        else:
            self.block += [
                WNConv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride + 1,
                    stride=stride,
                    padding_mode="zeros",
                    padding=stride,
                ),
            ]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        adapt_hopper: bool = True,
    ):
        super().__init__()
        # Create first convolution
        if adapt_hopper:
            self.block = [
                nn.ConstantPad1d(padding=3, value=0),
                WNConv1d(n_channels, d_model, kernel_size=7),
            ]
        else:
            self.block = [
                WNConv1d(n_channels, d_model, kernel_size=7, padding=3),
            ]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, adapt_hopper=adapt_hopper)]

        # Create last convolution
        if adapt_hopper:
            self.block += [
                Snake1d(d_model),
                nn.ConstantPad1d(padding=1, value=0),
                WNConv1d(d_model, d_model, kernel_size=3),
            ]
        else:
            self.block += [
                Snake1d(d_model),
                WNConv1d(d_model, d_model, kernel_size=3, padding=1),
            ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)

