import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
import math
from typing import List
from typing import Union


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DecoderBlock(nn.Module):

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()

        kernel_size = 2 * stride if stride % 2 == 0 else 2 * stride + 1
        padding = (kernel_size - stride) // 2

        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Vocoder(nn.Module):

    def __init__(
        self,
        latent_freq: int = 16,
        sample_rate: int = 32000,
    ):
        super().__init__()

        decoder_dim = 1024
        hidden_dim = 256

        self.latent_freq = latent_freq
        self.sample_rate = sample_rate

        decoder_rates = [5, 5, 5, 4, 2, 2]
        latent_dim = 128
        self.frame_len = 2000

        self.post_quant_conv = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.decoder = Decoder(input_channel=hidden_dim,
                               channels=decoder_dim,
                               rates=decoder_rates,
                               d_out=1)

        self.sample_rate = sample_rate
        self.apply(init_weights)

    @torch.no_grad()
    def decode(self, z: torch.Tensor):

        z = self.post_quant_conv(z)

        x_hat = self.decoder(z)

        return x_hat

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = Vocoder(latent_freq=16, sample_rate=32000).to("cuda")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    x = torch.randn(1, 1, 192000 * 1).to("cuda")

    x_hat, kl_loss, mu, logvar = model(x)

    print(x_hat.shape, kl_loss, mu.shape, logvar.shape)
