import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from recipes.semantic_coder.models.utils import get_padding, init_weights


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class WNResBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        padding_mode="zeros",
        act_fn="leaky_relu",
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, 1),
                ),
                WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, 1),
                ),
                WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, 1),
                ),
            ]
        )
        self.convs2.apply(init_weights)

        n_act = len(self.convs1) + len(self.convs2)
        if act_fn == "leaky_relu":
            self.act_fns = nn.ModuleList(
                [torch.nn.LeakyReLU(0.1) for i in range(n_act)]
            )
        else:
            self.act_fns = nn.ModuleList([Snake1d(channels) for i in range(n_act)])

    def forward(self, x, fixed_noises=None):
        i = 0
        for c1, c2 in zip(self.convs1, self.convs2):
            # xt = F.leaky_relu(x, 0.1)
            xt = self.act_fns[2 * i](x)
            xt = c1(xt)
            # xt = F.leaky_relu(xt, 0.1)
            xt = self.act_fns[2 * i + 1](xt)
            xt = c2(xt)
            x = xt + x
            i += 1
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ModulatedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=0,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if kernel_size * dilation - dilation - 2 * padding == 0:
            self.padding = padding
        else:
            raise Exception
        # weight
        weight = torch.randn(
            size=[out_channels, in_channels, kernel_size], dtype=torch.float32
        ) / np.sqrt(out_channels)
        weight.requires_grad = True
        self.register_parameter("weight", nn.Parameter(weight))
        # bias
        if bias:
            bias = torch.zeros(size=[out_channels], dtype=torch.float32)
            weight.requires_grad = True
            self.register_parameter("bias", nn.Parameter(bias))
        else:
            self.bias = None
        # style
        style = torch.ones(size=[out_channels], dtype=torch.float32)
        style.requires_grad = True
        self.register_parameter("style", nn.Parameter(style))
        self.infer_mode = False

    def forward(self, x):
        if not self.infer_mode:
            w = self.weight
            w = w * (w.square().sum(dim=[1, 2], keepdim=True) + 1e-8).rsqrt()  # [OIk]
            w = w * self.style.unsqueeze(1).unsqueeze(2)
        else:
            w = self.w_norm
        x = F.conv1d(
            x,
            w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def remove_weight_norm(self):
        self.infer_mode = True
        w = self.weight
        w = w * (w.square().sum(dim=[1, 2], keepdim=True) + 1e-8).rsqrt()  # [OIk]
        w = w * self.style.unsqueeze(1).unsqueeze(2)
        self.register_parameter("w_norm", nn.Parameter(w))


class ModulatedResBlock(torch.nn.Module):
    def __init__(
        self, channels, kernel_size=3, dilation=(1, 3, 5), act_fn="leaky_relu"
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                ModulatedConv(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                ModulatedConv(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                ModulatedConv(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                ModulatedConv(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ModulatedConv(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ModulatedConv(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
            ]
        )

        n_act = len(self.convs1) + len(self.convs2)
        if act_fn == "leaky_relu":
            self.act_fns = nn.ModuleList(
                [torch.nn.LeakyReLU(0.1) for i in range(n_act)]
            )
        else:
            self.act_fns = nn.ModuleList([Snake1d(channels) for i in range(n_act)])

    def forward(self, x):
        i = 0
        for c1, c2 in zip(self.convs1, self.convs2):
            # xt = F.leaky_relu(x, 0.1)
            xt = self.act_fns[2 * i](x)
            xt = c1(xt)
            # xt = F.leaky_relu(xt, 0.1)
            xt = self.act_fns[2 * i + 1](xt)
            xt = c2(xt)
            x = xt + x
            i += 1
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            l.remove_weight_norm()
        for l in self.convs2:
            l.remove_weight_norm()


class CausalWNResBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        padding_mode="zeros",
        act_fn="leaky_relu",
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )
        for conv in self.convs1:
            conv.conv.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, 1),
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, 1),
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding_mode=padding_mode,
                    padding=get_padding(kernel_size, 1),
                ),
            ]
        )
        for conv in self.convs2:
            conv.conv.apply(init_weights)

        n_act = len(self.convs1) + len(self.convs2)
        if act_fn == "leaky_relu":
            self.act_fns = nn.ModuleList(
                [torch.nn.LeakyReLU(0.1) for i in range(n_act)]
            )
        else:
            self.act_fns = nn.ModuleList([Snake1d(channels) for i in range(n_act)])

    def forward(self, x):
        i = 0
        for c1, c2 in zip(self.convs1, self.convs2):
            # xt = F.leaky_relu(x, 0.1)
            xt = self.act_fns[2 * i](x)
            xt = c1(xt)
            # xt = F.leaky_relu(xt, 0.1)
            xt = self.act_fns[2 * i + 1](xt)
            xt = c2(xt)
            x = xt + x
            i += 1
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        padding=0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.conv = WNConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            padding=0,
            device=device,
            dtype=dtype,
        )
        self.pad_layer = nn.ConstantPad1d([2 * padding, 0], 0.0)

    def forward(self, x):
        x = self.conv(self.pad_layer(x))
        return x
