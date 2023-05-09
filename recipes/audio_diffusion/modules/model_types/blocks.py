import torch
import torch.nn as nn

from .layers import GRN, FilmLayer


class Conv1d(nn.Conv1d):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class DepthWiseConvBlock1d(nn.Module):  # pragma: no cover
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=49,
        activation=nn.GELU(),
        cond_channels=512,
    ):
        super().__init__()
        self.film = FilmLayer(in_channels, cond_channels)
        hidden_channels = in_channels * 4
        self.depth_wise_conv = Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels,
        )
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.hidden_mapping = Conv1d(in_channels, hidden_channels, 1)
        self.activation = activation
        self.grn = GRN(hidden_channels)
        self.output_mapping = Conv1d(hidden_channels, out_channels, 1)
        self.skip = Conv1d(in_channels, out_channels, 1)

    def forward(self, x, cond):
        skip = self.skip(x)
        x = self.film(x, cond)
        x = self.depth_wise_conv(x)
        x = self.group_norm(x)
        x = self.hidden_mapping(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.output_mapping(x)
        return x + skip


class ConvNeXTUpBlock(nn.Module):  # pragma: no cover
    def __init__(
        self,
        in_channels,
        out_channels,
        factor,
        kernel_length,
        depth=3,
        cond_channels=512,
    ):
        super().__init__()

        self.factor = factor
        self.channel_mapping = Conv1d(2 * in_channels, in_channels, 1)
        self.resampling_layer = nn.ConvTranspose1d(
            in_channels, in_channels, factor, stride=factor, groups=in_channels
        )
        self.convs = nn.ModuleList([])
        for _ in range(depth):
            self.convs.append(
                DepthWiseConvBlock1d(
                    in_channels, in_channels, kernel_length, cond_channels=cond_channels
                )
            )
        self.output_mapping = Conv1d(in_channels, out_channels, 1)

    def forward(self, x, x_dblock, cond):
        x = torch.cat([x, x_dblock], dim=1)
        x = self.channel_mapping(x)
        for layer in self.convs:
            x = layer(x, cond)
        x = self.resampling_layer(x)
        x = self.output_mapping(x)
        return x


class ConvNeXTDownBlock(nn.Module):  # pragma: no cover
    def __init__(
        self,
        in_channels,
        out_channels,
        factor,
        kernel_length,
        depth=3,
        cond_channels=512,
    ):
        super().__init__()
        self.factor = factor
        self.resampling_layer = nn.Conv1d(
            out_channels, out_channels, factor, stride=factor, groups=out_channels
        )
        self.channel_mapping = Conv1d(in_channels, out_channels, 1)
        self.convs = nn.ModuleList([])
        for _ in range(depth):
            self.convs.append(
                DepthWiseConvBlock1d(
                    out_channels,
                    out_channels,
                    kernel_length,
                    cond_channels=cond_channels,
                )
            )

    def forward(self, x, cond):
        x = self.channel_mapping(x)
        x = self.resampling_layer(x)
        for layer in self.convs:
            x = layer(x, cond)
        return x
