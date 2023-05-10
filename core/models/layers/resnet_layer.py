'''ResNet implementation
Conventional ResNet/ResNeXt has been implemented in:
  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

However, it only supported standard ResNet architectures, i.e. the input is first
downsampled by a 7x7 conv and 3x3 max-pooling. Also, in the conv block, only 3x3
and 1x1 conv are implemented. This configuration may be not suitable for speech
applications.

To fix this problem, this module takes the above code as reference, and re-implement
a more flexible ResNet in which we can specifier the kernel and other settings.
'''
from collections import OrderedDict
from typing import Type, Union, Callable, List, Optional
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np


def conv_block(
    in_planes: int,
    out_planes: int,
    kernel_size: List[int] = (1, 1),
    stride: tuple = (1, 1),
    groups: int = 1,
    dilation: List[int] = (1, 1),
) -> nn.Conv2d:
    '''NxN convolution with padding'''
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=[dilation[i] * (kernel_size[i] // 2) for i in range(len(kernel_size))],
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    '''Basic block in ResNet (2-layer conv block)'''

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: List[int] = (3, 3),
        stride: List[int] = (1, 1),
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: List[int] = (1, 1),
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        batchnorm_momentum: float = 0.1,
        batchnorm_eps: float = 1e-5,
        batchnorm_affine: bool = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # base_width is not used here
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock does not support group convolution.')
        for d in dilation:
            if d > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_block(inplanes, planes, kernel_size, stride=stride)
        self.bn1 = norm_layer(
            planes, momentum=batchnorm_momentum, eps=batchnorm_eps, affine=batchnorm_affine
        )
        self.conv2 = conv_block(planes, planes, kernel_size)
        self.bn2 = norm_layer(
            planes, momentum=batchnorm_momentum, eps=batchnorm_eps, affine=batchnorm_affine
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        '''forward'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    '''Bottleneck block in ResNet (3-layer block).'''

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: List[int] = (3, 3),
        stride: List[int] = (1, 1),
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: List[int] = (1, 1),
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        batchnorm_momentum: float = 0.1,
        batchnorm_eps: float = 1e-5,
        batchnorm_affine: bool = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # The first conv is a 1x1 conv.
        self.conv1 = conv_block(inplanes, width, kernel_size=(1, 1))
        self.bn1 = norm_layer(
            width, momentum=batchnorm_momentum, eps=batchnorm_eps, affine=batchnorm_affine
        )
        self.conv2 = conv_block(
            width, width, kernel_size=kernel_size, stride=stride, groups=groups, dilation=dilation
        )
        self.bn2 = norm_layer(
            width, momentum=batchnorm_momentum, eps=batchnorm_eps, affine=batchnorm_affine
        )
        self.conv3 = conv_block(width, planes * self.expansion, kernel_size=(1, 1))
        self.bn3 = norm_layer(
            planes * self.expansion,
            momentum=batchnorm_momentum,
            eps=batchnorm_eps,
            affine=batchnorm_affine,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        '''forward'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResBlock(nn.Module):
    '''The Residual Block
    The residual block is the block used in ResNet. Typically, a resnet consists
    of 4 residual block. You can use this block to build a customized ResNet.
    The kernel size, stride and dilation are tuple which is for the frequency and
    time axes, respectively.
    '''

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        block_num: int,
        in_planes: int,
        out_planes: int,
        kernel_size: List[int],
        stride: List[int] = (1, 1),
        dilation: List[int] = (1, 1),
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        batchnorm_momentum: float = 0.1,
        batchnorm_eps: float = 1e-5,
        batchnorm_affine: bool = True,
        groups: int = 1,
        base_width: int = 64,
    ):
        '''Initialize a ResBlock.
        Args:
            fbank_dim: The dimension of the feature
            block: BasicBlock or Bottleneck
            block_num: How many blocks in this block
            in_planes: The number of filters in the previous layer
            out_planes: The number of filters in this layer
            kernel_size: kernel size in the freq and time axes
            stride: stride in the freq and time axes
            dilation: dilation in the freq and time axes
            groups, base_width: used in ResNeXt
        '''
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.base_width = base_width
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        downsample = None
        use_stride = np.prod(stride) > 1
        if use_stride or self.in_planes != self.out_planes * block.expansion:
            downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'conv',
                            conv_block(
                                self.in_planes,
                                self.out_planes * block.expansion,
                                kernel_size=(1, 1),
                                stride=stride,
                            ),
                        ),
                        (
                            'bn',
                            norm_layer(
                                self.out_planes * block.expansion,
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
            block(
                self.in_planes,
                self.out_planes,
                kernel_size=kernel_size,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=dilation,
                norm_layer=norm_layer,
                batchnorm_momentum=batchnorm_momentum,
                batchnorm_eps=batchnorm_eps,
                batchnorm_affine=batchnorm_affine,
            ),
        )
        self.in_planes = self.out_planes * block.expansion
        for i in range(1, block_num):
            self.layers.add_module(
                'block{}'.format(i),
                block(
                    self.in_planes,
                    self.out_planes,
                    kernel_size=kernel_size,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    batchnorm_momentum=batchnorm_momentum,
                    batchnorm_eps=batchnorm_eps,
                    batchnorm_affine=batchnorm_affine,
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
                    mask = F.pad(
                        mask,
                        (0, 2 * self.dilation[1] * (self.kernel_size[1] // 2), 0, 0),
                        mode='replicate',
                    )
                    mask = F.unfold(
                        mask, (1, self.kernel_size[1]), (1, self.dilation[1]), 0, (1, time_stride)
                    )
                    mask = mask[:, :1, :].unsqueeze(2)
                input_feat = input_feat * mask
        return input_feat, mask

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim
