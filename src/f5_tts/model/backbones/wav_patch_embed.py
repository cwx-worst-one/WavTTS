from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _align_seq_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Align tensor on last dim to target length by crop/pad."""
    cur_len = x.shape[-1]
    if cur_len > target_len:
        return x[..., :target_len]
    if cur_len < target_len:
        return F.pad(x, (0, target_len - cur_len), value=0.0)
    return x


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


class WavPatchEmbedV2FrontendBackend(nn.Module):
    """Multi-scale waveform patch embed/de-embed with split + deconv + sum."""

    def __init__(
        self,
        model_dim: int,
        out_dim: int | None = None,
        kernels: list[int] | tuple[int, ...] = (160, 400, 800),
        stride: int = 160,
        paddings: list[int] | tuple[int, ...] | None = None,
        branch_dim: int | None = None,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.embed_dim = int(self.model_dim if out_dim is None else out_dim)
        self.kernels = [int(k) for k in kernels]
        if len(self.kernels) == 0:
            raise ValueError("kernels must be non-empty for embed_v2.")

        self.hop_length = int(stride)
        self.branch_dim = int(self.embed_dim // 2 if branch_dim is None else branch_dim)
        if self.branch_dim <= 0:
            raise ValueError(f"branch_dim must be positive, got {self.branch_dim}.")

        if paddings is None:
            self.paddings = [(k - self.hop_length) // 2 for k in self.kernels]
        else:
            self.paddings = [int(p) for p in paddings]
            if len(self.paddings) != len(self.kernels):
                raise ValueError(
                    f"Length mismatch: paddings({len(self.paddings)}) vs kernels({len(self.kernels)})."
                )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=self.branch_dim,
                    kernel_size=k,
                    stride=self.hop_length,
                    padding=p,
                )
                for k, p in zip(self.kernels, self.paddings, strict=False)
            ]
        )
        self.deconvs = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels=self.branch_dim,
                    out_channels=1,
                    kernel_size=k,
                    stride=self.hop_length,
                    padding=p,
                )
                for k, p in zip(self.kernels, self.paddings, strict=False)
            ]
        )

        cat_dim = self.branch_dim * len(self.kernels)
        self.proj = nn.Linear(cat_dim, self.embed_dim)
        self.proj_inv = nn.Linear(self.model_dim, cat_dim)

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

        branch_feats = [conv(wav_1d) for conv in self.convs]  # list of [B, Cb, T_i]
        token_count = branch_feats[0].shape[-1]
        branch_feats = [_align_seq_len(feat, token_count) for feat in branch_feats]
        cat = torch.cat(branch_feats, dim=1).transpose(1, 2)  # [B, T, Cb*num_kernels]
        tokens_embed = self.proj(cat)  # [B, T, embed_dim]
        tokens = self.to_model(tokens_embed.transpose(1, 2)).transpose(1, 2)  # [B, T, model_dim]

        token_mask = None
        if mask is not None:
            pooled_masks = []
            mask_1d = mask.float().unsqueeze(1)
            for k, p in zip(self.kernels, self.paddings, strict=False):
                pooled = F.max_pool1d(mask_1d, kernel_size=k, stride=self.hop_length, padding=p).squeeze(1) > 0.0
                pooled_masks.append(_align_seq_len(pooled, token_count))
            token_mask = pooled_masks[0]
            for pooled in pooled_masks[1:]:
                token_mask = token_mask | pooled

        token_lens = None
        if lens is not None:
            token_lens = (lens.to(dtype=torch.long, device=wav.device) + self.hop_length - 1) // self.hop_length
            token_lens = token_lens.clamp(max=token_count)

        return tokens, token_mask, token_lens

    def decode(self, tokens: torch.Tensor, target_num_samples: int):
        assert tokens.ndim == 3, f"Expected [B, T, D] tokens, got {tuple(tokens.shape)}"
        tokens_embed = self.from_model(tokens.transpose(1, 2)).transpose(1, 2)  # [B, T, embed_dim]
        cat = self.proj_inv(tokens_embed)  # [B, T, Cb*num_kernels]
        branches = cat.split(self.branch_dim, dim=-1)  # tuple of [B, T, Cb]

        branch_wavs = []
        for branch_tokens, deconv in zip(branches, self.deconvs, strict=False):
            wav_branch = deconv(branch_tokens.transpose(1, 2)).squeeze(1)  # [B, N_i]
            branch_wavs.append(wav_branch)

        wav_len = max(w.shape[1] for w in branch_wavs)
        wav_sum = None
        for wav_branch in branch_wavs:
            wav_branch = _align_seq_len(wav_branch, wav_len)
            wav_sum = wav_branch if wav_sum is None else wav_sum + wav_branch

        if wav_sum.shape[1] < target_num_samples:
            wav_sum = F.pad(wav_sum, (0, target_num_samples - wav_sum.shape[1]), value=0.0)
        return wav_sum[:, :target_num_samples]
