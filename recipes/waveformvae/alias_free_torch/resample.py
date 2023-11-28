# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch

import torch.nn as nn
from torch.nn import functional as F
from .filter import LowPassFilter1d
from .filter import kaiser_sinc_filter1d
import copy
from recipes.waveformvae.utils.flops_utils import *
from typing import List

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]

        return x

    @torch.jit.export
    def flops(self, in_shape: List[int]):
        cur_shape = list(in_shape).copy()
        C = in_shape[1]
        cur_shape[2] = in_shape[2] * self.ratio
        filter_shape = self.filter.expand(C, -1, -1).shape
        flops = (2*calculate_product(filter_shape)-1) * calculate_product(cur_shape) / C + calculate_product(cur_shape) #groups=C
        return flops, cur_shape


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)

        return xx

    @torch.jit.export
    def flops(self, in_shape: List[int]):
        cur_shape = list(in_shape).copy()
        cur_shape[2] = in_shape[2] // self.ratio
        C = in_shape[1]
        filter_shape = self.lowpass.filter.expand(C, -1, -1).shape
        flops = (2*calculate_product(filter_shape)-1) * calculate_product(cur_shape) / C  #groups=C
        return flops, cur_shape
