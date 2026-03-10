from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ChannelLastConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        return x.permute(0, 2, 1)


class ConvMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w3 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvMLPOutProjection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        kernel_size: int = 7,
        padding: int = 3,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = in_dim * 4 if hidden_dim is None else hidden_dim
        self.pre = nn.Sequential(
            ChannelLastConv1d(in_dim, in_dim, kernel_size=kernel_size, padding=padding),
            nn.SELU(),
            ConvMLP(
                in_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                multiple_of=multiple_of,
            ),
        )
        self.out = ChannelLastConv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.output_layer = self.out

    def forward(self, x: torch.Tensor):
        return self.out(self.pre(x))
