""" Complex layer
"""
import torch
from torch import nn
from panther_utilities.torch_module.panther_module import PantherLog10

# pylint: disable='dangerous-default-value'
class ComplexConv2d(nn.Module):
    """ComplexConv2d"""

    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        kernel_size=[3, 1],
        padding=[1, 0],
        stride=[1, 1],
        bias=False,
    ):
        super().__init__()
        self.conv_re = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias
        )
        self.conv_im = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias
        )

    def forward(self, x):
        """forward process"""
        x_re = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        x_im = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        return torch.stack([x_re, x_im], dim=-1)


class PhaseEncoderLog(nn.Module):
    """phase encoder using ComplexConv2d"""

    def __init__(self):
        super().__init__()
        self.complex_conv = ComplexConv2d(in_channels=6, out_channels=6)

    def forward(self, x):
        """forward"""
        x = self.complex_conv(x)
        x = (x.pow(2).sum(dim=-1) + 1e-8) ** 0.5
        x = torch.log10(x + 1e-8)
        return x


class PhaseEncoderLogCh(nn.Module):
    """phase encoder using ComplexConv2d which can change output channel"""

    def __init__(self, in_ch=4, out_ch=4, kernel_size=[3, 1], padding=[1, 0]):
        super().__init__()

        self.complex_conv = ComplexConv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding
        )
        self.log10 = PantherLog10()

    def forward(self, x):
        """froward"""
        x = self.complex_conv(x)
        x = (x.pow(2).sum(dim=-1) + 1e-8) ** 0.5
        x = self.log10(x + 1e-8)
        return x
