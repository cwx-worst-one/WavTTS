from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def init_dac_weights(module: nn.Module):
    if isinstance(module, nn.Conv1d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def wn_conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def wn_conv_transpose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake_fn(x: torch.Tensor, alpha: torch.Tensor):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor):
        return snake_fn(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            wn_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            wn_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            wn_conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            wn_conv_transpose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class WavConvEncoder(nn.Module):
    def __init__(self, encoder_dim: int, encoder_rates: tuple[int, ...], latent_dim: int):
        super().__init__()
        layers = [wn_conv1d(1, encoder_dim, kernel_size=7, padding=3)]
        current_dim = encoder_dim
        for stride in encoder_rates:
            next_dim = current_dim * 2
            layers.append(EncoderBlock(next_dim, stride=stride))
            current_dim = next_dim
        layers.extend(
            [
                Snake1d(current_dim),
                wn_conv1d(current_dim, latent_dim, kernel_size=3, padding=1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class WavConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, decoder_dim: int, decoder_rates: tuple[int, ...]):
        super().__init__()
        layers = [wn_conv1d(latent_dim, decoder_dim, kernel_size=7, padding=3)]
        for i, stride in enumerate(decoder_rates):
            input_dim = decoder_dim // (2**i)
            output_dim = decoder_dim // (2 ** (i + 1))
            layers.append(DecoderBlock(input_dim, output_dim, stride=stride))
        layers.extend(
            [
                Snake1d(output_dim),
                wn_conv1d(output_dim, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class WavConvFrontendBackend(nn.Module):
    def __init__(
        self,
        model_dim: int,
        encoder_dim: int = 64,
        encoder_rates: tuple[int, ...] = (2, 4, 5, 4),
        latent_dim: int = 256,
        decoder_dim: int = 1024,
        decoder_rates: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if decoder_rates is None:
            decoder_rates = tuple(reversed(encoder_rates))

        self.encoder_rates = tuple(int(x) for x in encoder_rates)
        self.decoder_rates = tuple(int(x) for x in decoder_rates)
        self.hop_length = int(math.prod(self.encoder_rates))

        self.encoder = WavConvEncoder(
            encoder_dim=int(encoder_dim),
            encoder_rates=self.encoder_rates,
            latent_dim=int(latent_dim),
        )
        self.to_model = wn_conv1d(int(latent_dim), int(model_dim), kernel_size=1)   # TODO: ???

        self.from_model = wn_conv1d(int(model_dim), int(latent_dim), kernel_size=1)   # TODO: ???
        self.decoder = WavConvDecoder(
            latent_dim=int(latent_dim),
            decoder_dim=int(decoder_dim),
            decoder_rates=self.decoder_rates,
        )

        # Align with DAC-style conv initialization.
        self.apply(init_dac_weights)

    def encode(
        self,
        wav: torch.Tensor,
        mask: torch.Tensor | None = None,
        lens: torch.Tensor | None = None,
    ):
        assert wav.ndim == 2, f"Expected [B, N] wav input, got {tuple(wav.shape)}"
        bsz, num_samples = wav.shape
        pad_len = (self.hop_length - (num_samples % self.hop_length)) % self.hop_length

        wav_1d = wav.unsqueeze(1)
        if pad_len > 0:
            wav_1d = F.pad(wav_1d, (0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        tokens = self.to_model(self.encoder(wav_1d)).transpose(1, 2)
        token_count = tokens.shape[1]

        token_mask = None
        if mask is not None:
            pooled = F.max_pool1d(mask.float().unsqueeze(1), kernel_size=self.hop_length, stride=self.hop_length)
            token_mask = pooled.squeeze(1) > 0.0
            if token_mask.shape[1] > token_count:
                token_mask = token_mask[:, :token_count]
            elif token_mask.shape[1] < token_count:
                token_mask = F.pad(token_mask, (0, token_count - token_mask.shape[1]), value=False)

        token_lens = None
        if lens is not None:
            token_lens = (lens.to(dtype=torch.long, device=wav.device) + self.hop_length - 1) // self.hop_length
            token_lens = token_lens.clamp(max=token_count)

        return tokens, token_mask, token_lens

    def decode(self, tokens: torch.Tensor, target_num_samples: int):
        assert tokens.ndim == 3, f"Expected [B, T, D] tokens, got {tuple(tokens.shape)}"
        wav = self.decoder(self.from_model(tokens.transpose(1, 2))).squeeze(1)
        if wav.shape[1] < target_num_samples:
            wav = F.pad(wav, (0, target_num_samples - wav.shape[1]), value=0.0)
        return wav[:, :target_num_samples]
