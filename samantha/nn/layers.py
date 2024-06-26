import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


def WNLinear(*args, **kwargs):
    return weight_norm(nn.Linear(*args, **kwargs))


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
# TODO: memory usage?
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


try:
    snake_beta = torch.compile(snake_beta)
except RuntimeError:
    pass


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class SnakeBeta(nn.Module):
    """Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py"""

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x
