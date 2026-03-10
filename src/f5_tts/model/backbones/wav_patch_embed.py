from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WavPatchEmbedV1FrontendBackend(nn.Module):
    """Lightweight waveform patch embed/de-embed with Conv1d and ConvTranspose1d."""

    def __init__(
        self,
        model_dim: int,
        out_dim: int | None = None,
        kernel_size: int = 400,
        stride: int = 160,
        padding: int | None = None,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.embed_dim = int(self.model_dim if out_dim is None else out_dim)
        self.kernel_size = int(kernel_size)
        self.hop_length = int(stride)
        if padding is None:
            padding = (self.kernel_size - self.hop_length) // 2
        self.padding = int(padding)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.embed_dim,
            kernel_size=self.kernel_size,
            stride=self.hop_length,
            padding=self.padding,
        )
        self.deconv = nn.ConvTranspose1d(
            in_channels=self.embed_dim,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=self.hop_length,
            padding=self.padding,
        )

        if self.embed_dim == self.model_dim:
            self.to_model = nn.Identity()
            self.from_model = nn.Identity()
        else:
            self.to_model = nn.Conv1d(self.embed_dim, self.model_dim, kernel_size=1)
            self.from_model = nn.Conv1d(self.model_dim, self.embed_dim, kernel_size=1)

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

        tokens = self.to_model(self.conv(wav_1d)).transpose(1, 2)
        token_count = tokens.shape[1]

        token_mask = None
        if mask is not None:
            pooled = F.max_pool1d(
                mask.float().unsqueeze(1),
                kernel_size=self.kernel_size,
                stride=self.hop_length,
                padding=self.padding,
            )
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
        wav = self.deconv(self.from_model(tokens.transpose(1, 2))).squeeze(1)
        if wav.shape[1] < target_num_samples:
            wav = F.pad(wav, (0, target_num_samples - wav.shape[1]), value=0.0)
        return wav[:, :target_num_samples]
