import torch
from typing import List

def flops_conv2d(module, in_shape):
    assert len(in_shape) == 4
    assert isinstance(module, torch.nn.Conv2d)
    b, ic, ih, iw = in_shape
    oh = (ih + 2*module.padding[0] -module.dilation[0]*(module.kernel_size[0]-1)-1) // module.stride[0] + 1
    ow = (iw + 2*module.padding[1] -module.dilation[1]*(module.kernel_size[1]-1)-1) // module.stride[1] + 1
    return (2*ic*module.kernel_size[0]*module.kernel_size[1]-1) * b * module.out_channels * oh * ow, [b, module.out_channels, oh, ow]

def flops_conv1d(module, in_shape):
    assert len(in_shape) == 3
    assert isinstance(module, torch.nn.Conv1d)
    b, ic, ih = in_shape
    oh = (ih + 2*module.padding[0] -module.dilation[0]*(module.kernel_size[0]-1)-1) // module.stride[0] + 1
    return (2*ic*module.kernel_size[0]-1)*module.out_channels*oh*b, [b, module.out_channels, int(oh)]

def flops_convtranspose1d(module, in_shape):
    assert len(in_shape) == 3
    assert isinstance(module, torch.nn.ConvTranspose1d)
    b, ic, ih = in_shape
    oh = (ih - 1) * module.stride[0] - 2*module.padding[0] + module.dilation[0]*(module.kernel_size[0]-1) + module.output_padding[0] + 1
    return (2*ic*module.kernel_size[0]-1)*module.out_channels*oh*b, [b, module.out_channels, int(oh)]

@torch.jit.script
def calculate_product(array: List[int]):
    product = 1
    for element in array:
        product *= element
    return product
