import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from recipes.soundstream.utils.utils import get_padding, init_weights

LRELU_SLOPE = 0.1
padding_mode = "zeros"

__all__ = [
    "SineGen",
    "NoiseFC",
    "WNResBlock1",
    "ModulatedResBlock1",
    "CausalWNResBlock1",
]


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
        add_noise=False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.add_noise = add_noise

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            if self.add_noise:
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            else:
                noise_amp = uv * self.noise_std
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class NoiseFC(nn.Module):
    def __init__(self, dim, use_noise=True, trunc_noise=True):
        super().__init__()

        self.use_noise = use_noise
        if not use_noise:
            return

        # inspired by stylegan
        self.fcs = nn.Sequential(
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
            # weight_norm(nn.Conv1d(512, dim, kernel_size=1)),
        )
        self.noise_res = None
        self.max_length = -1
        self.trunc_noise = trunc_noise

    def forward(self, x, fixed_noise=None):
        if not self.use_noise:
            return 0.0

        """
        # 在推理中，使用固定的noise，可以节省非常多的计算量
        if fixed_noise is not None:
            b, d, t = fixed_noise.size()
            t_x = x.size(-1)
            repeat_num = int(np.ceil(t_x / t))
            res = torch.cat([fixed_noise] * repeat_num, dim=-1)
            res = res[:, :, 0:t_x]
            return res
        elif not self.training:
            with torch.no_grad():
                b, d, t = x.size()
                if t > self.max_length:
                    self.max_length = t
                    noise = torch.randn(size=[1, d, t], device=x.device)
                    if self.trunc_noise:
                        noise = torch.nn.init.trunc_normal_(noise) * 0.1
                    else:
                        noise = noise * 0.1
                    self.noise_res = self.fcs(noise)
            return self.noise_res[:, :, 0:t]
        else:
            noise = torch.randn_like(x, device=x.device)
            if self.trunc_noise:
                noise = torch.nn.init.trunc_normal_(noise) * 0.1
            else:
                noise = noise * 0.1
            res = self.fcs(noise)
            self.noise_res = None
            self.max_length = -1
            return res
        """
        noise = torch.randn_like(x, device=x.device)
        if self.trunc_noise:
            noise = torch.nn.init.trunc_normal_(noise) * 0.1
        else:
            noise = noise * 0.1
        res = self.fcs(noise)
        self.noise_res = None
        self.max_length = -1
        return res

    def get_fixed_noise(self, size):
        # size: [B, D, T]
        b, d, t = size
        noise = torch.randn(size=[1, d, t], device="cpu")
        if self.trunc_noise:
            noise = torch.nn.init.trunc_normal_(noise) * 0.1
        else:
            noise = noise * 0.1
        self.fixed_noise = self.fcs(noise)
        return self.fixed_noise

    def remove_weight_norm(self):
        for l in self.fcs:
            remove_weight_norm(l)


class WNResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), trunc_noise=False):
        super().__init__()
        norm_f = weight_norm
        self.convs1 = nn.ModuleList(
            [
                norm_f(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding_mode=padding_mode,
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                norm_f(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding_mode=padding_mode,
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                norm_f(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding_mode=padding_mode,
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                norm_f(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding_mode=padding_mode,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm_f(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding_mode=padding_mode,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm_f(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding_mode=padding_mode,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)
        self.noise_convs = nn.ModuleList(
            [
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
            ]
        )

    def forward(self, x, fixed_noises=None):
        i = 0
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if fixed_noises is not None:
                xt = c1(xt) + self.noise_convs[2 * i](xt, fixed_noises[2 * i])
            else:
                xt = c1(xt) + self.noise_convs[2 * i](xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if fixed_noises is not None:
                xt = c2(xt) + self.noise_convs[2 * i + 1](xt, fixed_noises[2 * i + 1])
            else:
                xt = c2(xt) + self.noise_convs[2 * i + 1](xt)
            x = xt + x
            i += 1
        return x

    def get_fixed_noises(self, x):
        _, d, t = x.size()
        fixed_noises = []
        for layer in self.noise_convs:
            fixed_noises.append(layer.get_fixed_noise([1, d, t]))
        return fixed_noises

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
        for l in self.noise_convs:
            l.remove_weight_norm()


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


class ModulatedResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), trunc_noise=False):
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
        self.noise_convs = nn.ModuleList(
            [
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
                NoiseFC(channels, trunc_noise),
            ]
        )

    def forward(self, x, fixed_noises=None):
        i = 0
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if fixed_noises is not None:
                xt = c1(xt) + self.noise_convs[2 * i](xt, fixed_noises[2 * i])
            else:
                xt = c1(xt) + self.noise_convs[2 * i](xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if fixed_noises is not None:
                xt = c2(xt) + self.noise_convs[2 * i + 1](xt, fixed_noises[2 * i + 1])
            else:
                xt = c2(xt) + self.noise_convs[2 * i + 1](xt)
            x = xt + x
            i += 1
        return x

    def get_fixed_noises(self, x):
        _, d, t = x.size()
        fixed_noises = []
        for layer in self.noise_convs:
            fixed_noises.append(layer.get_fixed_noise([1, d, t]))
        return fixed_noises

    def remove_weight_norm(self):
        for l in self.convs1:
            l.remove_weight_norm()
        for l in self.convs2:
            l.remove_weight_norm()
        for l in self.noise_convs:
            l.remove_weight_norm()


class CausalWNResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5], trunc_noise=False):
        super().__init__()

        layers = nn.ModuleList([])

        for dilation in dilations:
            layer = nn.ModuleList([])

            conv_0 = CausalConv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation,
                padding_mode=padding_mode,
                padding=get_padding(kernel_size, dilation),
            )
            conv_0.conv.apply(init_weights)
            layer.append(conv_0)

            conv_1 = CausalConv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding_mode=padding_mode,
                padding=get_padding(kernel_size, 1),
            )
            conv_1.conv.apply(init_weights)
            layer.append(conv_1)

            for _ in range(2):
                layer.append(NoiseFC(channels, trunc_noise))
            layers.append(layer)

        self.layers = layers

    def forward(self, x, fixed_noises=None):

        for i, layer in enumerate(self.layers):
            conv_0, conv_1, noise_0, noise_1 = layer

            xt = F.leaky_relu(x, LRELU_SLOPE)
            if fixed_noises is not None:
                xt = conv_0(xt) + noise_0(xt, fixed_noises[2 * i])
            else:
                xt = conv_0(xt) + noise_0(xt)

            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if fixed_noises is not None:
                xt = conv_1(xt) + noise_1(xt, fixed_noises[2 * i + 1])
            else:
                xt = conv_1(xt) + noise_1(xt)

            x = xt + x

        return x

    def get_fixed_noises(self, x):
        _, d, t = x.size()
        fixed_noises = []
        for layer in self.noise_convs:
            fixed_noises.append(layer.get_fixed_noise([1, d, t]))
        return fixed_noises

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
        for l in self.noise_convs:
            l.remove_weight_norm()


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
        self.conv = weight_norm(
            nn.Conv1d(
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
        )
        self.pad_layer = nn.ConstantPad1d([2 * padding, 0], 0.0)

    def forward(self, x):
        x = self.conv(self.pad_layer(x))
        return x
