import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from math import sqrt, ceil


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=num_features,
            momentum=momentum,
            affine=affine,
            eps=eps,
            track_running_stats=track_running_stats,
        )
        # self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats)
        # self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats)

    def forward(self, x):
        mag_in = torch.sqrt(torch.sum(torch.pow(x, 2.0), dim=-1))
        mag_out = self.bn(mag_in)
        x_scale = mag_out / (mag_in + 1e-12)
        real = x[..., 0] * x_scale
        imag = x[..., 1] * x_scale
        output = torch.stack((real, imag), dim=-1)
        return output


class SVConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        batch_norm=False,
        act=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if (type(kernel_size) is not int) else (kernel_size, kernel_size)
        )
        self.kernels = kernels
        self.stride = stride if (type(stride) is not int) else (stride, stride)
        self.padding = padding if (type(padding) is not int) else (padding, padding)
        self.dilation = dilation if (type(dilation) is not int) else (dilation, dilation)
        self.dilated_kernel_size = [
            (self.kernel_size[i] - 1) * self.dilation[i] + 1 for i in range(2)
        ]
        self.bias = bias
        self.batch_norm = batch_norm
        self.act = act
        self.kernel = nn.Parameter(
            torch.FloatTensor(out_channels, in_channels, kernels, *self.kernel_size),
            requires_grad=True,
        )
        if bias:
            self.bia = nn.Parameter(
                torch.FloatTensor(1, out_channels, 1, kernels), requires_grad=True
            )
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, a=sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bia, -bound, bound)

    def forward(self, x):
        assert x.ndim == 4
        x = torch.constant_pad_nd(
            x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        )
        x_stride = x.stride()
        x_shape = x.shape
        shape = (
            *x_shape[:-2],
            (x_shape[-2] - self.dilated_kernel_size[0]) // self.stride[0] + 1,
            (x_shape[-1] - self.dilated_kernel_size[1]) // self.stride[1] + 1,
            *self.kernel_size,
        )
        strides = (
            *x_stride[:-2],
            x_stride[-2] * self.stride[0],
            x_stride[-1] * self.stride[1],
            x_stride[-2] * self.dilation[0],
            x_stride[-1] * self.dilation[1],
        )
        x = torch.as_strided(x, size=shape, stride=strides)
        x = torch.einsum('bitfmn,oifmn->botf', x, self.kernel)
        if self.bias:
            x += self.bia
        if self.batch_norm:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ComplexSVConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()

        self.conv_re = SVConv2d(
            in_channels,
            out_channels,
            kernel_size,
            kernels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.conv_im = SVConv2d(
            in_channels,
            out_channels,
            kernel_size,
            kernels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):  # shape of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexSVMobileConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        batch_norm=False,
        act=None,
        groups=-1,
    ):
        super().__init__()

        self.conv_re = SVMobileConv2d(
            in_channels,
            out_channels,
            kernel_size,
            kernels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.conv_im = SVMobileConv2d(
            in_channels,
            out_channels,
            kernel_size,
            kernels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.cpx_batchnorm = ComplexBatchNorm2d(out_channels)
        self.act = act
        self.batch_norm = batch_norm

    def forward(self, x):  # shape of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        if self.batch_norm is not None:
            output = self.cpx_batchnorm(output)
        if self.act is not None:
            out_abs = torch.sqrt(torch.sum(torch.pow(output, 2), dim=-1))
            act_out = self.act(out_abs)
            output_real = output[..., 0] / (out_abs + 1e-12) * act_out
            output_imag = output[..., 1] / (out_abs + 1e-12) * act_out
            output = torch.stack((output_real, output_imag), dim=-1)
        return output


class SVMobileConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        batch_norm=True,
        act_depthwise=None,
        act_pointwise=None,
    ):
        super().__init__()
        self.act_depthwise = act_depthwise
        self.act_pointwise = act_pointwise
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if (type(kernel_size) is not int) else (kernel_size, kernel_size)
        )
        self.kernels = kernels
        self.stride = stride if (type(stride) is not int) else (stride, stride)
        self.padding = padding if (type(padding) is not int) else (padding, padding)
        self.dilation = dilation if (type(dilation) is not int) else (dilation, dilation)
        self.dilated_kernel_size = [
            (self.kernel_size[i] - 1) * self.dilation[i] + 1 for i in range(2)
        ]
        self.bias = bias
        self.depthwise_kernel = nn.Parameter(
            torch.FloatTensor(in_channels, kernels, *self.kernel_size), requires_grad=True
        )
        self.pointwise_kernel = nn.Parameter(
            torch.FloatTensor(out_channels, in_channels, kernels), requires_grad=True
        )
        if bias:
            self.depthwise_bias = torch.nn.Parameter(
                torch.FloatTensor(1, in_channels, 1, kernels), requires_grad=True
            )
            self.pointwise_bias = torch.nn.Parameter(
                torch.FloatTensor(1, out_channels, 1, kernels), requires_grad=True
            )
        if batch_norm:
            self.depthwise_norm = nn.BatchNorm2d(in_channels)
            self.pointwise_norm = nn.BatchNorm2d(out_channels)
        self.reset_parameters()
        self.batch_norm = batch_norm

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.depthwise_kernel, a=sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.depthwise_kernel)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.depthwise_bias, -bound, bound)

        nn.init.kaiming_uniform_(self.pointwise_kernel, a=sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pointwise_kernel)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.pointwise_bias, -bound, bound)

    def forward(self, x):
        assert x.ndim == 4
        x = torch.constant_pad_nd(
            x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        )
        x_stride = x.stride()
        x_shape = x.shape
        shape = (
            *x_shape[:-2],
            (x_shape[-2] - self.dilated_kernel_size[0]) // self.stride[0] + 1,
            (x_shape[-1] - self.dilated_kernel_size[1]) // self.stride[1] + 1,
            *self.kernel_size,
        )
        strides = (
            *x_stride[:-2],
            x_stride[-2] * self.stride[0],
            x_stride[-1] * self.stride[1],
            x_stride[-2] * self.dilation[0],
            x_stride[-1] * self.dilation[1],
        )
        x = torch.as_strided(x, size=shape, stride=strides)
        x = torch.einsum('bitfmn,ifmn->bitf', x, self.depthwise_kernel)
        if self.bias:
            x += self.depthwise_bias
        if self.batch_norm:
            x = self.depthwise_norm(x)
        if self.act_depthwise is not None:
            x = self.act_depthwise(x)
        # x = F.leaky_relu(x)
        x = torch.einsum('bitf,oif->botf', x, self.pointwise_kernel)
        if self.bias:
            x += self.pointwise_bias
        if self.batch_norm:
            x = self.pointwise_norm(x)
        if self.act_pointwise is not None:
            x = self.act_pointwise(x)
        return x


class SVMobileConv2d_group(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels,
        stride=1,
        padding=0,
        dilation=1,
        groups=-1,
        bias=True,
        batch_norm=True,
        act_depthwise=None,
        act_pointwise=None,
    ):
        super().__init__()
        self.act_depthwise = act_depthwise
        self.act_pointwise = act_pointwise
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if (type(kernel_size) is not int) else (kernel_size, kernel_size)
        )
        self.kernels = kernels
        self.stride = stride if (type(stride) is not int) else (stride, stride)
        self.padding = padding if (type(padding) is not int) else (padding, padding)
        self.dilation = dilation if (type(dilation) is not int) else (dilation, dilation)
        self.dilated_kernel_size = [
            (self.kernel_size[i] - 1) * self.dilation[i] + 1 for i in range(2)
        ]
        self.bias = bias
        self.groups = groups if 0 < groups < kernels else kernels
        self.num_kernels_per_group = ceil(kernels / self.groups)
        self.depthwise_kernel = nn.Parameter(
            torch.FloatTensor(in_channels, self.groups, *self.kernel_size), requires_grad=True
        )
        self.pointwise_kernel = nn.Parameter(
            torch.FloatTensor(out_channels, in_channels, self.groups), requires_grad=True
        )
        if bias:
            self.depthwise_bias = torch.nn.Parameter(
                torch.FloatTensor(1, in_channels, 1, self.groups), requires_grad=True
            )
            self.pointwise_bias = torch.nn.Parameter(
                torch.FloatTensor(1, out_channels, 1, self.groups), requires_grad=True
            )
        if batch_norm:
            self.depthwise_norm = nn.BatchNorm2d(in_channels)
            self.pointwise_norm = nn.BatchNorm2d(out_channels)
        self.reset_parameters()
        self.batch_norm = batch_norm

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.depthwise_kernel, a=sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.depthwise_kernel)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.depthwise_bias, -bound, bound)

        nn.init.kaiming_uniform_(self.pointwise_kernel, a=sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pointwise_kernel)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.pointwise_bias, -bound, bound)

    def forward(self, x):
        assert x.ndim == 4
        x = torch.constant_pad_nd(
            x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        )
        x_stride = x.stride()
        x_shape = x.shape
        shape = (
            *x_shape[:-2],
            (x_shape[-2] - self.dilated_kernel_size[0]) // self.stride[0] + 1,
            (x_shape[-1] - self.dilated_kernel_size[1]) // self.stride[1] + 1,
            *self.kernel_size,
        )
        strides = (
            *x_stride[:-2],
            x_stride[-2] * self.stride[0],
            x_stride[-1] * self.stride[1],
            x_stride[-2] * self.dilation[0],
            x_stride[-1] * self.dilation[1],
        )
        x = torch.as_strided(x, size=shape, stride=strides)

        # x = torch.einsum('bitfmn,ifmn->bitf', x, self.depthwise_kernel)
        depthwise_kernel = self.depthwise_kernel.unsqueeze(dim=2).expand(
            (
                *self.depthwise_kernel.shape[:2],
                self.num_kernels_per_group,
                *self.depthwise_kernel.shape[-2:],
            )
        )
        depthwise_kernel = torch.reshape(
            depthwise_kernel, (depthwise_kernel.shape[0], -1, *depthwise_kernel.shape[-2:])
        )
        depthwise_kernel = depthwise_kernel[:, : self.kernels]
        x = torch.einsum('bitfmn,ifmn->bitf', x, depthwise_kernel)
        del depthwise_kernel

        if self.bias:
            depthwise_bias = self.depthwise_bias.unsqueeze(dim=-1).expand(
                (*self.depthwise_bias.shape, self.num_kernels_per_group)
            )
            depthwise_bias = torch.reshape(depthwise_bias, (*depthwise_bias.shape[:3], -1))
            depthwise_bias = depthwise_bias[..., : self.kernels]
            x += depthwise_bias
            # x += self.depthwise_bias
        del depthwise_bias

        if self.batch_norm:
            x = self.depthwise_norm(x)
        if self.act_depthwise is not None:
            x = self.act_depthwise(x)
        # x = F.leaky_relu(x)

        pointwise_kernel = self.pointwise_kernel.unsqueeze(dim=-1).expand(
            (*self.pointwise_kernel.shape, self.num_kernels_per_group)
        )
        pointwise_kernel = torch.reshape(pointwise_kernel, (*pointwise_kernel.shape[:2], -1))
        pointwise_kernel = pointwise_kernel[..., : self.kernels]
        x = torch.einsum('bitf,oif->botf', x, pointwise_kernel)
        del pointwise_kernel
        # x = torch.einsum('bitf,oif->botf', x, self.pointwise_kernel)

        if self.bias:
            pointwise_bias = self.pointwise_bias.unsqueeze(dim=-1).expand(
                (*self.pointwise_bias.shape, self.num_kernels_per_group)
            )
            pointwise_bias = torch.reshape(pointwise_bias, (*pointwise_bias.shape[:3], -1))
            pointwise_bias = pointwise_bias[..., : self.kernels]
            x += pointwise_bias
        del pointwise_bias
        # x += self.pointwise_bias

        if self.batch_norm:
            x = self.pointwise_norm(x)
        if self.act_pointwise is not None:
            x = self.act_pointwise(x)
        return x


class time_average(nn.Module):
    def _init_(self):
        super(time_average, self).__init__()

    def forward(self, x):
        # B, T , F => B, 1, F
        return x.mean(1, keepdim=True)


class multiplicative_adaptation(nn.Module):
    def _init_(self):
        super(multiplicative_adaptation, self).__init__()

    def forward(self, x, e):
        return x * e


class GroupGRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        group_num,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    ):
        super().__init__()
        self.rnns = nn.ModuleList()
        self.group_num = group_num
        for group in range(group_num):
            self.rnns.append(
                nn.GRU(
                    input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional
                )
            )

    def forward(self, x):
        # x: B, group_num, T, input_size
        # out: B, group_num, T, hidden_size
        features = [torch.jit.fork(self.rnns[i], x[:, i, :, :]) for i in range(self.group_num)]
        results = [torch.jit.wait(feat)[0] for feat in features]
        return torch.stack(results, dim=1)


class CausalDecayMA(nn.Module):
    def __init__(self, alpha=None, time_steps=100) -> None:
        super().__init__()
        self.time_steps = time_steps
        # self.weights = alpha**(torch.arange(start=time_steps-1,end=0,step=-1,dtype=torch.float32))*(1.-alpha)
        self.alpha = 10 ** (-2.8 / (time_steps - 1))
        self.weights = 10 ** torch.linspace(-2.8, 0, time_steps)
        self.weights = self.weights / torch.sum(self.weights)

    def forward(self, x):
        # B 1 T
        assert x.shape[1] == 1
        B, _, T = x.shape
        kernel = x.new_ones(1, 1, self.time_steps) * self.weights.to(x.device)
        x = torch.nn.functional.pad(x, [self.time_steps - 1, 0])
        x = torch.nn.functional.conv1d(x, weight=kernel)
        assert x.shape[-1] == T
        return x


class spatialbeam(nn.Module):
    def __init__(self, args):
        super().__init__()
        mic_num = args.mic_num
        out_channel1 = args.out_channel1
        out_channel2 = args.out_channel2

        self.spectral_cnn = nn.ModuleList(
            [
                SVMobileConv2d(
                    in_channels=mic_num * 2,
                    out_channels=mic_num * 2,
                    kernel_size=[1, 3],
                    kernels=64,
                    stride=[1, 2],
                    padding=[0, 0],
                    dilation=[1, 1],
                    batch_norm=True,
                    act_depthwise=nn.PReLU(),
                    act_pointwise=nn.PReLU(),
                ),
                SVMobileConv2d(
                    in_channels=mic_num * 2,
                    out_channels=mic_num * 2,
                    kernel_size=[1, 3],
                    kernels=32,
                    stride=[1, 2],
                    padding=[0, 1],
                    dilation=[1, 1],
                    batch_norm=True,
                    act_depthwise=nn.PReLU(),
                    act_pointwise=nn.PReLU(),
                ),
            ]
        )

        self.drop_out = nn.Dropout(0.2)
        self.combine_crn = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SVMobileConv2d(
                            in_channels=mic_num * 2,
                            out_channels=out_channel1,
                            kernel_size=[1, 3],
                            kernels=32,
                            stride=[1, 1],
                            padding=[0, 1],
                            dilation=[1, 1],
                            batch_norm=True,
                            act_depthwise=nn.PReLU(),
                            act_pointwise=nn.PReLU(),
                        ),
                        nn.GRU(32, 32, num_layers=1, batch_first=True),
                    ]
                ),
                nn.ModuleList(
                    [
                        SVMobileConv2d(
                            in_channels=out_channel1 * 2,
                            out_channels=out_channel1,
                            kernel_size=[1, 3],
                            kernels=32,
                            stride=[1, 1],
                            padding=[0, 2],
                            dilation=[1, 2],
                            batch_norm=True,
                            act_depthwise=nn.PReLU(),
                            act_pointwise=nn.PReLU(),
                        ),
                        nn.GRU(32, 32, num_layers=1, batch_first=True),
                    ]
                ),
                nn.ModuleList(
                    [
                        SVMobileConv2d(
                            in_channels=out_channel1 * 2,
                            out_channels=out_channel1,
                            kernel_size=[1, 3],
                            kernels=32,
                            stride=[1, 1],
                            padding=[0, 4],
                            dilation=[1, 4],
                            batch_norm=True,
                            act_depthwise=nn.PReLU(),
                            act_pointwise=nn.PReLU(),
                        ),
                        nn.GRU(32, 32, num_layers=1, batch_first=True),
                    ]
                ),
                nn.ModuleList(
                    [
                        SVMobileConv2d(
                            in_channels=out_channel1 * 2,
                            out_channels=out_channel2,
                            kernel_size=[1, 3],
                            kernels=32,
                            stride=[1, 1],
                            padding=[0, 8],
                            dilation=[1, 8],
                            batch_norm=True,
                            act_depthwise=nn.PReLU(),
                            act_pointwise=nn.PReLU(),
                        ),
                        nn.GRU(32, 32, num_layers=1, batch_first=True),
                    ]
                ),
            ]
        )
        self.sig_out = nn.Linear(32 * out_channel2, 129, bias=False)

    def forward(self, batch):
        x_spectra = batch['spectra_in']  # B nmic T F 2
        x_spectra = x_spectra.permute(0, 1, 4, 2, 3).contiguous()
        x_spectra = x_spectra.reshape(x_spectra.shape[0], -1, *x_spectra.shape[3:])

        for i in range(len(self.spectral_cnn)):
            x_spectra = self.spectral_cnn[i](x_spectra)

        for i in range(len(self.combine_crn)):
            if i == 0:
                x_combine = x_spectra
            else:
                x_combine = torch.cat([x_spectra, x_spatial], dim=1)
            x_spatial = self.combine_crn[i][0](x_combine)
            Batch = x_spatial.shape[0]
            Ch = x_spatial.shape[1]
            x_spectra = torch.reshape(x_spatial, (-1, x_spatial.shape[2], x_spatial.shape[3]))
            x_spectra = self.combine_crn[i][1](x_spectra)[0]
            x_spectra = self.drop_out(x_spectra)
            x_spectra = torch.reshape(x_spectra, (Batch, Ch, x_spectra.shape[1], -1))

        x_out = x_spectra.permute(0, 2, 1, 3)
        x_out = torch.reshape(x_out, (x_out.shape[0], x_out.shape[1], -1))
        mask_out = torch.sigmoid(self.sig_out(x_out))  # B T F
        pre_spectra = batch['base'] * mask_out.unsqueeze(-1)
        out = {
            'pre_mask': mask_out,
            'pre_spectra': pre_spectra,
        }
        return out
