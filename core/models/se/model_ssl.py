import sys
import torch
import numpy as np
from torch import nn
from math import sqrt, ceil
import torch.nn.functional as F


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
        spec_norm=False,
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
        if spec_norm:
            self.norm = nn.BatchNorm2d(kernels, affine=True, track_running_stats=True)
            self.spec_norm = True
        else:
            self.spec_norm = False
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
        if self.spec_norm:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1).contiguous()
        elif self.batch_norm:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GSVConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels,
        stride=1,
        padding=0,
        dilation=1,
        use_bias=True,
        spec_norm=False,
        batch_norm=False,
        act=nn.PReLU(),
        ch_groups=1,
        residual=False,
    ):
        super().__init__()
        assert in_channels % ch_groups == 0 and out_channels % ch_groups == 0
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
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.act = act
        self.ch_groups = ch_groups
        self.in_channels_per_group = in_channels // ch_groups
        self.out_channels_per_group = out_channels // ch_groups
        self.kernel = nn.Parameter(
            torch.FloatTensor(out_channels, self.in_channels_per_group, kernels, *self.kernel_size),
            requires_grad=True,
        )
        if use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(1, out_channels, 1, kernels), requires_grad=True
            )
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        if spec_norm:
            self.norm = nn.BatchNorm2d(kernels, affine=True, track_running_stats=True)
            self.spec_norm = True
        else:
            self.spec_norm = False
        self.reset_parameters()
        self.residual = residual

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, a=sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x_res = x
        assert x.ndim == 4
        x = torch.constant_pad_nd(
            x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        )
        x = torch.reshape(x, (x.shape[0], -1, self.ch_groups, *x.shape[-2:]))
        x = x.permute(0, 2, 1, 3, 4)
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
        x = torch.einsum(
            'bgitfmn,goifmn->bgotf', x, self.kernel.view(self.ch_groups, -1, *self.kernel.shape[1:])
        )
        x = torch.reshape(x, (x.shape[0], -1, *x.shape[-2:]))
        if self.use_bias:
            x += self.bias
        if self.residual:
            x += x_res
        if self.spec_norm:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1).contiguous()
        elif self.batch_norm:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GSVConv2d_noShuffle(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels,
        stride=1,
        padding=0,
        dilation=1,
        use_bias=True,
        spec_norm=False,
        batch_norm=False,
        act=nn.PReLU(),
        ch_groups=1,
        residual=False,
    ):
        super().__init__()
        assert in_channels % ch_groups == 0 and out_channels % ch_groups == 0
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
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.act = act
        self.ch_groups = ch_groups
        self.in_channels_per_group = in_channels // ch_groups
        self.out_channels_per_group = out_channels // ch_groups
        self.kernel = nn.Parameter(
            torch.FloatTensor(out_channels, self.in_channels_per_group, kernels, *self.kernel_size),
            requires_grad=True,
        )
        if use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(1, out_channels, 1, kernels), requires_grad=True
            )
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        if spec_norm:
            self.norm = nn.BatchNorm2d(kernels, affine=True, track_running_stats=True)
            self.spec_norm = True
        else:
            self.spec_norm = False
        self.reset_parameters()
        self.residual = residual

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, a=sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x_res = x
        assert x.ndim == 4
        x = torch.constant_pad_nd(
            x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        )
        x = torch.reshape(x, (x.shape[0], self.ch_groups, -1, *x.shape[-2:]))
        # x = x.permute(0, 2, 1, 3, 4)
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
        x = torch.einsum(
            'bgitfmn,goifmn->bgotf', x, self.kernel.view(self.ch_groups, -1, *self.kernel.shape[1:])
        )
        x = torch.reshape(x, (x.shape[0], -1, *x.shape[-2:]))
        if self.use_bias:
            x += self.bias
        if self.residual:
            x += x_res
        if self.spec_norm:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1).contiguous()
        elif self.batch_norm:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DFSMNLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        memory_size,
        out_size,
        left_kernel_size,
        right_kernel_size,
        dilation=1,
        dropout=0.0,
        groups=-1,
    ):
        super().__init__()
        self.h2p_proj = nn.Sequential(
            *[nn.Conv1d(hidden_size, memory_size, 1), nn.PReLU(), nn.Dropout(dropout)]
        )
        self.memory = nn.Conv1d(
            memory_size,
            memory_size,
            kernel_size=left_kernel_size + right_kernel_size + 1,
            padding=0,
            stride=1,
            dilation=dilation,
            groups=memory_size if groups < 0 else groups,
            bias=False,
        )
        self.p2h_proj = nn.Sequential(
            *[
                nn.Conv1d(memory_size, out_size, 1),
                nn.BatchNorm1d(out_size),
                nn.PReLU(),
                nn.Dropout(dropout),
            ]
        )
        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.memory_size = memory_size

    def forward(self, input_fea, skip_connection=None):
        p = self.h2p_proj(input_fea)
        pad_p = F.pad(
            p, (self.left_kernel_size * self.dilation, self.right_kernel_size * self.dilation, 0, 0)
        )  # (B,N,T+(l+r)*d)
        if skip_connection is None:
            memory_out = self.memory(pad_p) + p
        else:
            memory_out = self.memory(pad_p) + p + skip_connection
        output = self.p2h_proj(memory_out)
        return output, memory_out


class DFSMNModule(nn.Module):
    def __init__(
        self,
        layer_num,
        hidden_size,
        memory_size,
        out_size,
        kernel_size,
        dilation=1,
        groups=-1,
        dropout=0.0,
    ):
        super().__init__()
        self.module_list = nn.ModuleList()
        for i in range(layer_num):
            if i == 0:
                self.module_list.append(
                    DFSMNLayer(
                        hidden_size,
                        memory_size,
                        out_size,
                        kernel_size[0],
                        kernel_size[1],
                        dilation=dilation,
                        dropout=dropout,
                        groups=groups,
                    )
                )
            else:
                self.module_list.append(
                    DFSMNLayer(
                        out_size,
                        memory_size,
                        out_size,
                        kernel_size[0],
                        kernel_size[1],
                        dilation=dilation,
                        dropout=dropout,
                        groups=groups,
                    )
                )
        self.layer_num = layer_num

    def forward(self, input):
        dfsmn_out = input.permute(0, 2, 1)
        skip_connection = None
        for i in range(self.layer_num):
            dfsmn_out, skip_connection = self.module_list[i](dfsmn_out, skip_connection)
        return dfsmn_out.permute(0, 2, 1)


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
        act_depthwise=nn.PReLU(),
        act_pointwise=nn.PReLU(),
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


class nnlocation(nn.Module):
    def __init__(self, args):
        super().__init__()
        mic_num = args.mic_num
        zone_num = args.zone_num
        inter_channel1 = args.inter_channel1
        inter_channel2 = args.inter_channel2
        inter_channel3 = args.inter_channel3
        self.zone_num = zone_num
        if inter_channel3 is None:
            inter_channel3 = zone_num

        self.doa_conv_block1 = nn.Sequential(
            GSVConv2d_noShuffle(
                in_channels=mic_num * 2,
                out_channels=inter_channel1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                use_bias=False,
                kernels=129,
                batch_norm=False,
                spec_norm=False,
                act=None,
            ),
            nn.BatchNorm2d(inter_channel1),
            nn.ReLU(inplace=True),
            GSVConv2d_noShuffle(
                in_channels=inter_channel1,
                out_channels=inter_channel1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                use_bias=False,
                kernels=129,
                ch_groups=inter_channel1,
                batch_norm=False,
                spec_norm=False,
                act=None,
            ),
            nn.BatchNorm2d(inter_channel1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(
                in_channels=inter_channel1,
                out_channels=inter_channel1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channel1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.doa_conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channel1,
                out_channels=inter_channel1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                bias=False,
                groups=inter_channel1,
            ),
            nn.Conv2d(
                in_channels=inter_channel1,
                out_channels=inter_channel2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channel2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(
                in_channels=inter_channel2,
                out_channels=inter_channel2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channel2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.doa_conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channel2,
                out_channels=inter_channel3,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channel3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(
                in_channels=inter_channel3,
                out_channels=inter_channel3,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channel3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.doa_conv_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channel3,
                out_channels=inter_channel3,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channel3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(
                in_channels=inter_channel3,
                out_channels=inter_channel3,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(inter_channel3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.doa_history_track = DFSMNModule(
            layer_num=2,
            hidden_size=inter_channel3 * 4,
            memory_size=inter_channel3 * 2,
            out_size=inter_channel3 * 2,
            kernel_size=(4, 0),
        )
        self.doa_fc_out = nn.Linear(inter_channel3 * 2, zone_num, bias=True)

    def forward(self, batch):
        x = batch['block_input']
        x = x.contiguous()  # B block 4 T F
        batch_size, block_size = x.shape[0:2]
        x = x.reshape(-1, *x.shape[2:])
        x = self.doa_conv_block1(x)
        x = self.doa_conv_block2(x)
        x = self.doa_conv_block3(x)
        x = self.doa_conv_block4(x)
        x = x.reshape(batch_size, block_size, -1)
        x = self.doa_history_track(x)
        x = torch.sigmoid(self.doa_fc_out(x))
        return x
