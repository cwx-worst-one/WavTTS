# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch

import torch.nn as nn
from .resample import UpSample1d, DownSample1d
from typing import List

class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x

    @torch.jit.export
    def flops(self, in_shape: List[int]):
        flops_ = 0.0
        cur_shape = in_shape
        cur_flops, cur_shape = self.upsample.flops(cur_shape)
        flops_ += cur_flops
        cur_flops, cur_shape = self.act.flops(cur_shape)
        flops_ += cur_flops
        cur_flops, cur_shape = self.downsample.flops(cur_shape)
        flops_ += cur_flops
        return flops_, cur_shape
