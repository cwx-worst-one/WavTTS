import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .blocks import ConvNeXTDownBlock, ConvNeXTUpBlock


class Conv1d(nn.Conv1d):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class RFF_MLP_Block(nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList(
            [nn.Linear(64, 128), nn.Linear(128, 256), nn.Linear(256, 512)]
        )

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class DAGNext(nn.Module):  # pragma: no cover
    def __init__(
        self,
        num_signal_channels=1,
        downsampling_factors=[2, 4, 4],
        hidden_sizes=[64, 128, 256],
        middle_bypassed=[True, True, False],
        resampling_block_kernel_sizes=[15, 15, 15],
        resampling_block_depths=[2, 2, 2],
        text_cond_dim=128,
        text_cond_hidden_dim=128,
        text_cond_noise_coeff=0.001,
        input_norm=False,
        enable_text_cond_pos_encoding=False,
    ):
        super().__init__()
        self.num_signal_channels = num_signal_channels
        self.sigma_cond_channels = (
            512  # Hardcoded to match RFF embedding. Replace later
        )

        self.text_cond_mlp = nn.Sequential(
            nn.Linear(text_cond_dim, text_cond_hidden_dim),
            nn.GELU(),
            nn.Linear(text_cond_hidden_dim, text_cond_hidden_dim),
        )
        self.downsampling_factors = downsampling_factors
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes.insert(0, num_signal_channels)
        self.gru_bypassed = middle_bypassed
        self.resampling_block_kernel_sizes = resampling_block_kernel_sizes
        self.resampling_block_depths = resampling_block_depths
        self.input_norm = input_norm
        self.text_cond_noise_coeff = text_cond_noise_coeff

        self.block_size = int(
            torch.prod(torch.tensor(self.downsampling_factors)).item()
        )
        self.embedding = RFF_MLP_Block()

        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_signal_channels)
        self.downsample = nn.ModuleList([])
        self.middle = nn.ModuleList([])
        self.upsample = nn.ModuleList([])

        middle_layer_type = WrappedCrossAttn

        for index, factor in enumerate(self.downsampling_factors):
            self.downsample.append(
                ConvNeXTDownBlock(
                    self.hidden_sizes[index],
                    self.hidden_sizes[index + 1],
                    factor,
                    self.resampling_block_kernel_sizes[index],
                    depth=self.resampling_block_depths[index],
                    cond_channels=self.sigma_cond_channels,
                )
            )
            self.middle.append(
                middle_layer_type(
                    self.hidden_sizes[index + 1],
                    cond_dim=text_cond_hidden_dim,
                    dummy=self.gru_bypassed[index],
                    enable_pos_encoding=enable_text_cond_pos_encoding,
                )
            )
            self.upsample.append(
                ConvNeXTUpBlock(
                    self.hidden_sizes[index + 1],
                    self.hidden_sizes[index],
                    factor,
                    self.resampling_block_kernel_sizes[index],
                    depth=self.resampling_block_depths[index],
                    cond_channels=self.sigma_cond_channels,
                )
            )

    def forward(self, audio, sigma, text_cond=None):
        if audio.shape[-1] % self.block_size != 0:
            raise Exception(
                f"Incorrect input size! Please use a multiple of {self.block_size}"
            )
        if self.input_norm:
            x = self.norm(audio)
        else:
            x = audio
        downsampled = []
        sigma_encoding = self.embedding(sigma.squeeze(1))
        cond = sigma_encoding
        if text_cond is not None:
            if self.training:
                text_cond = text_cond + self.text_cond_noise_coeff * torch.randn_like(
                    text_cond
                )
            text_cond = self.text_cond_mlp(text_cond.permute(0, 2, 1))

        for layer in self.downsample:
            x = layer(x, cond)
            downsampled.append(x)

        for i in range(1, len(self.upsample) + 1):
            x = self.middle[-i](x, text_cond)
            x = self.upsample[-i](x, downsampled[-i], cond)

        return x


class WrappedGRU(nn.Module):  # pragma: no cover
    def __init__(self, channels=512, dummy=False):
        super().__init__()
        self.dummy = dummy
        if not self.dummy:
            self.gru = nn.GRU(channels, channels, batch_first=True, bidirectional=False)
            self.output_map = nn.Conv1d(2 * channels, channels, 1)

    def forward(self, x):
        if not self.dummy:
            gru_output, _ = self.gru(x.permute(0, 2, 1))
            combined = torch.cat([x, gru_output.permute(0, 2, 1)], dim=1)
            return self.output_map(combined)
        else:
            return x


class WrappedCrossAttn(nn.Module):  # pragma: no cover
    def __init__(
        self,
        channels=512,
        cond_dim=128,
        dummy=False,
        linear_attn=True,
        enable_pos_encoding=False,
    ):
        super().__init__()
        self.dummy = dummy
        if not self.dummy:
            self.key_map = nn.Linear(cond_dim, channels)
            self.query_map = nn.Linear(channels, channels)
            self.value_map = nn.Linear(cond_dim, channels)
            self.output_map = nn.Conv1d(2 * channels, channels, 1)
            self.num_heads = 4
            self.scale = (channels / self.num_heads) ** -0.5
            if enable_pos_encoding:
                self.input_pe = PositionalEncoding(channels)
                self.cond_pe = PositionalEncoding(cond_dim)
            self.enable_pos_encoding = enable_pos_encoding
            if linear_attn:
                self.attn = self.linear_attention
            else:
                self.attn = self.standard_attention

    def standard_attention(self, q, k, v, h):
        h = self.num_heads
        # Split heads
        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        # Compute similarity matrix and add eventual mask
        sim = torch.einsum("... n d, ... m d -> ... n m", q, k) * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = torch.einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out

    def linear_attention(self, q, k, v, h):
        # Split heads
        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        # Softmax rows and cols
        q = q.softmax(dim=-1) * self.scale
        k = k.softmax(dim=-2)
        # Attend on channel dim
        attn = torch.einsum("... n d, ... n c -> ... d c", k, v)
        out = torch.einsum("... n d, ... d c -> ... n c", q, attn)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out

    def forward(self, x, cond):
        if not self.dummy:
            x_permuted = x.permute(0, 2, 1)
            if self.enable_pos_encoding:
                x_permuted = self.input_pe(x_permuted)
                cond = self.cond_pe(cond)
            attn_output = self.attn(
                self.query_map(x_permuted),
                self.key_map(cond),
                self.value_map(cond),
                self.num_heads,
            )
            combined = torch.cat([x, attn_output.permute(0, 2, 1)], dim=1)
            return self.output_map(combined)
        else:
            return x


class PositionalEncoding(nn.Module):  # pragma: no cover
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        x = x.permute(1, 0, 2)
        return x
