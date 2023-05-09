import math
from enum import Enum
from typing import Optional

import torch
from torch import nn


@torch.jit.script
def new_gelu(x):
    """Implementation of the GELU activation function currently in Google BERT
    repo (identical to OpenAI GPT).

    Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class Activation(str, Enum):
    GeLU = "gelu"
    SiLU = "silu"
    ReLU = "relu"
    LeakyReLU = "leaky_relu"


class NewGeLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return new_gelu(x)


def build_activation(activation: Optional[Activation]):
    if not activation:
        return nn.Identity()

    return {
        Activation.ReLU: nn.ReLU,
        Activation.SiLU: nn.SiLU,
        Activation.GeLU: NewGeLU,
        Activation.LeakyReLU: nn.LeakyReLU,
    }[activation]()
