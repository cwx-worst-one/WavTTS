# pylint: disable=protected-access
''' active_function '''

from typing import Callable, List
import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F


def get_activation_fn(activation, act_glu=False):
    '''get_activation_fn'''
    return ActiveFn(activation, act_glu)


class ActiveFn:
    '''ActiveFn'''

    def __init__(self, fn_name, act_glu=False):
        '''init.'''
        self.act_fn = name2fn_map(fn_name)
        self.act_glu = act_glu

    def __call__(self, x, **_kwargs):
        '''call'''
        if self.act_glu:
            x, v = x.chunk(2, dim=-1)
            return self.act_fn(x) * v
        return self.act_fn(x)


def name2fn_map(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    # pylint: disable=too-many-return-statements
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return gelu
    if activation == 'gelu_fast':
        warnings.warn('--activation-fn=gelu_fast has been renamed to gelu_accurate', stacklevel=3)
        return gelu_accurate
    if activation == 'gelu_accurate':
        return gelu_accurate
    if activation == 'tanh':
        return torch.tanh
    if activation == 'linear':
        return lambda x: x
    if activation == 'swish':
        return swish
    activation_fns = ','.join(get_available_activation_fns())
    raise RuntimeError("activation {} not supported in ({})".format(activation, activation_fns))


def get_available_activation_fns() -> List:
    '''get_available_activation_fns'''
    return [
        'relu',
        'gelu',
        'gelu_fast',  # deprecated
        'gelu_accurate',
        'swish',
        'tanh',
        'linear',
    ]


def gelu_accurate(x):
    '''gelu_accurate'''
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x: torch.Tensor) -> torch.Tensor:
    '''gelu'''
    return torch.nn.functional.gelu(x.float()).type_as(x)


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swich activation function."""
    x = x.float()
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    """Swish activation module"""

    def __init__(self, swish_beta: float = 1.0):
        """construct Swish module with beta"""
        super().__init__()
        self._beta = swish_beta

    def forward(self, x):
        """activation function"""
        return x * torch.sigmoid(self._beta * x)


class GELU(nn.Module):
    """GELU activation module"""

    # pylint: disable=no-self-use
    def forward(self, x):
        """activation function"""
        return torch.nn.functional.gelu(x.float()).type_as(x)


class ReLU(nn.ReLU):
    """ReLU activation module"""

    def __init__(self, relu_inplace=False):
        """construct ReLU with config"""
        super().__init__(inplace=relu_inplace)


activation_module = {
    "relu": ReLU,
    "gelu": GELU,
    "linear": nn.Identity,
    "swish": Swish,
    "tanh": nn.Tanh,
}


def get_activation_module(activation: str):
    """Returns the activation module corresponding to `activation`"""
    if activation in activation_module:
        return activation_module[activation]
    raise RuntimeError(
        "activation {} not supported in ({})".format(activation, activation_module.keys())
    )
