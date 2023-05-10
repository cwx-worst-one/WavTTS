'''
part of interface for complex computing
'''

import torch
from packaging import version


def complex_multiply(in1, in2):
    """complex multiply"""
    assert in1.ndim == in2.ndim
    if version.parse(torch.__version__) >= version.parse('1.7.0'):
        out_real = in1.real * in2.real - in1.imag * in2.imag
        out_imag = in1.real * in2.imag + in1.imag * in2.real
        return torch.complex(out_real, out_imag)

    in1_real = in1[..., 0]
    in1_imag = in1[..., 1]
    in2_real = in2[..., 0]
    in2_imag = in2[..., 1]
    out_real = in1_real * in2_real - in1_imag * in2_imag
    out_imag = in1_real * in2_imag + in1_imag * in2_real
    return torch.stack([out_real, out_imag], -1)


def to_old(input_tensor):
    '''
    complex tensor to real and imag tensor
    [N1, N2, N3, .. , Nn] ->[N1, N2, N3, .. , Nn, 2]
    '''
    return torch.stack([input_tensor.real, input_tensor.imag], -1)


def to_new(input_tensor):
    '''
    real and imag tensor to complex tensor
    [N1, N2, N3, .. , Nn, 2] -> [N1, N2, N3, .. , Nn]
    '''
    real = input_tensor.index_select(-1, torch.tensor(0))
    imag = input_tensor.index_select(-1, torch.tensor(1))
    return torch.complex(real, imag)
