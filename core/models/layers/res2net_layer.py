'''Res2Net implementation
https://github.com/Res2Net/Res2Net-PretrainedModels
'''
import math
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Bottle2neck(nn.Module):
    '''Bottle2neck
    The basic module of Res2Net
    '''

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=(3, 3),
        stride=(1, 1),
        downsample=None,
        base_width=26,
        scale=4,
        batchnorm_momentum=0.1,
        batchnorm_eps=1e-5,
        batchnorm_affine=True,
        stype='normal',
    ):
        """Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super().__init__()

        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            width * scale, momentum=batchnorm_momentum, eps=batchnorm_eps, affine=batchnorm_affine
        )
        self.nums = max(1, scale - 1)

        padding = [k // 2 for k in kernel_size]
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            bns.append(
                nn.BatchNorm2d(
                    width, momentum=batchnorm_momentum, eps=batchnorm_eps, affine=batchnorm_affine
                )
            )
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion,
            momentum=batchnorm_momentum,
            eps=batchnorm_eps,
            affine=batchnorm_affine,
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        '''forward'''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.stype == 'normal':
                spo.append(spx[-1])
            elif self.stype == 'stage':
                spo.append(self.pool(spx[-1]))
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Res2Block(nn.Module):
    '''The Res2Net Block'''

    def __init__(
        self,
        block_num,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        batchnorm_momentum=0.1,
        batchnorm_eps=1e-5,
        batchnorm_affine=True,
        base_width=26,
        scale=4,
    ):
        '''Initialize a Res2Block.'''
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.base_width = base_width
        self.scale = scale
        self.kernel_size = kernel_size
        self.stride = stride
        downsample = None
        use_stride = np.prod(stride) > 1
        if use_stride or self.in_planes != self.out_planes * Bottle2neck.expansion:
            downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'conv',
                            nn.Conv2d(
                                self.in_planes,
                                self.out_planes * Bottle2neck.expansion,
                                kernel_size=(1, 1),
                                stride=stride,
                                bias=False,
                            ),
                        ),
                        (
                            'norm',
                            nn.BatchNorm2d(
                                self.out_planes * Bottle2neck.expansion,
                                momentum=batchnorm_momentum,
                                eps=batchnorm_eps,
                                affine=batchnorm_affine,
                            ),
                        ),
                    ]
                )
            )
        self.layers = nn.ModuleList()
        self.layers.add_module(
            'block0',
            Bottle2neck(
                self.in_planes,
                self.out_planes,
                kernel_size=kernel_size,
                stride=stride,
                downsample=downsample,
                stype='stage',
                base_width=base_width,
                scale=scale,
                batchnorm_momentum=batchnorm_momentum,
                batchnorm_eps=batchnorm_eps,
                batchnorm_affine=batchnorm_affine,
            ),
        )
        self.in_planes = self.out_planes * Bottle2neck.expansion
        for i in range(1, block_num):
            self.layers.add_module(
                'block{}'.format(i),
                Bottle2neck(
                    self.in_planes,
                    self.out_planes,
                    kernel_size=kernel_size,
                    base_width=base_width,
                    scale=scale,
                    batchnorm_momentum=batchnorm_momentum,
                    batchnorm_eps=batchnorm_eps,
                    batchnorm_affine=batchnorm_affine,
                    stype='normal',
                ),
            )
        self._output_dim = self.in_planes

    def forward(self, input_feat, mask=None):
        '''forward
        Args:
            input_feat: the feature should be [B, C, D, T]
            mask: the mask should be [B, C, D, T] as well. Often [B, 1, 1, T]
        '''
        time_stride = self.stride[1]
        for i, layer in enumerate(self.layers):
            input_feat = layer(input_feat)
            if mask is not None:
                if i == 0 and time_stride > 1:
                    mask = F.pad(mask, (0, 2 * (self.kernel_size[1] // 2), 0, 0), mode='replicate')
                    mask = F.unfold(mask, (1, self.kernel_size[1]), 1, 0, stride=(1, time_stride))
                    mask = mask[:, :1, :].unsqueeze(2)
                input_feat = input_feat * mask
        return input_feat, mask

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim
