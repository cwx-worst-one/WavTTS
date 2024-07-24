import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.distributions.normal import Normal
from torchaudio.transforms import AmplitudeToDB

from samantha.models.base import LossDict
from samantha.transforms.audio import MelSpectrogram
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def linear_schedule(t, clip_min=1e-9) -> float:
    # A gamma function that simply is 1-t.
    return np.clip(1 - t, clip_min, 1.0)


def cosine_schedule(
    t, start: float, end: float, tau: float = 1, clip_min: float = 1e-9
) -> float:
    # Algorithm #1 from https://arxiv.org/pdf/2301.10972.pdf
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)


def get_steps_from_schedule(
    n_steps: int,
    schedule: str,
    start: float = 0.0,
    end: float = 1.0,
    tau: float = 1.0,
    clip_min: float = 1e-9,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    ts = torch.linspace(start, end, n_steps)
    if schedule == "linear":
        schedule = [linear_schedule(t, clip_min=clip_min) for t in ts]
    elif schedule == "cosine":
        schedule = [
            cosine_schedule(t, start=start, end=end, tau=tau, clip_min=clip_min)
            for t in ts
        ]
    return torch.tensor(schedule, dtype=torch.float32, device=device)


def get_alpha_beta(sigmas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get alpha/beta values from sampled sigmas (timesteps / noise levels)

    Args:
        sigmas (torch.Tensor): Scalar value representing the noise level

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Alpha/beta values used to determine velocity
    """
    angle = sigmas * math.pi / 2
    alpha, beta = torch.cos(angle), torch.sin(angle)
    return alpha, beta


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def normal_init(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def xavier_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class MelTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        f_min: int,
        f_max: int,
        power: int = 2,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

        if power == 2:
            stype = "power"
        elif power == 1:
            stype = "magnitude"
        else:
            raise NotImplementedError("Choose between [power, magnitude]")

        self.amplitude_to_db = AmplitudeToDB(stype=stype)

    def verify_input_waveform(self, waveform: torch.Tensor) -> None:
        if waveform.ndim != 3:
            raise Exception("input tensor should be [batch_size, channels, time]")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        self.verify_input_waveform(waveform)
        mel = self.mel_transform(waveform)
        mel = mel[..., :-1]
        mel = self.amplitude_to_db(mel)  # alternatively, safe_log(mel)
        return rearrange(mel, "b n_channels n_mels time -> b n_channels time n_mels")


class MelModel(nn.Module):
    """This class represents an 'ideal' audio feature model.
    Ideally, autoencoders compress and reconstruct well enough so that our
    latent diffusion model has enough guidance. This 'model' can be used to
    mimic an ideal autoencoder.
    """

    def __init__(self, sample_rate: int, n_mels: int, frame_rate: int, f_max: int):
        self.mel_transform = MelTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            win_length=2048,
            hop_length=(config.sample_rate // frame_rate),
            f_min=0,
            f_max=f_max,
        )
        self.n_embd = n_mels

    def forward(self, audio: torch.Tensor):
        return self.mel_transform(audio)


class UniformDistribution:
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin

class StandardNormalDistribution:
    
    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        return torch.randn(num_samples, device=device)


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features)

    def to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return self.to_out(fouriered)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class TimeEmbedding(nn.Module):
    def __init__(
        self, modulation_features: int, num_layers: int = 2, bias: bool = False
    ):
        super().__init__()
        self.embedding = NumberEmbedder(features=modulation_features)

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(modulation_features, modulation_features, bias=bias),
                    nn.GELU(),
                )
                for _ in range(num_layers)
            ]
        )

    def init_weights(self):
        logger.info("Initializing TimeEmbedding weights...")
        self.embedding.apply(normal_init)  # TODO: verify this
        self.mlp.apply(normal_init)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # Process time to time_features
        time_features = F.gelu(
            self.embedding(time), approximate="none"
        )  # TODO: tanh to speed up

        for layer in self.mlp:
            time_features = layer(time_features)

        # Overlap features if more than one per batch
        if time_features.ndim == 3:
            time_features = reduce(time_features, "b n d -> b d", "sum")

        return time_features


def compound_noise_schedule(
    x: torch.Tensor,
    eps: torch.Tensor,
    sigma: torch.Tensor,
    scale: float,
    normalize: bool = False,
) -> torch.Tensor:
    """Proposed in https://arxiv.org/pdf/2301.10972.pdf
    As we reduce the scaling factor, it increases the noise levels, as demonstrated in Figure 4.

    NOTE: If normalize=True, it assumes that the data `x` is already zero-mean unit-variance
    """

    gamma = lambda t: 1 - t

    x_t = torch.sqrt(gamma(sigma)) * scale * x + torch.sqrt(1 - gamma(sigma)) * eps

    if normalize:
        x_t = x_t / x_t.std(dim=(1, 2), keepdims=True)
    return x_t


def sequence_mask(length: torch.Tensor, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class WeightedMSELoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.weight * F.mse_loss(x, y, reduction="none")  # [B, _, _]
        loss = loss.mean(dim=list(range(1, len(x.shape))))  # [B]
        return loss


class WeightedMAELoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.weight * F.l1_loss(x, y, reduction="none")  # [B, _, _]
        loss = loss.mean(dim=list(range(1, len(x.shape))))  # [B]
        return loss


class NullEmb(nn.Module):

    def __init__(self, n_embd: int, cfg_prob: float, learnable: bool = True):
        super().__init__()

        if learnable:
            self.null_emb = nn.Parameter(torch.randn(n_embd), requires_grad=True)
        else:
            self.register_buffer("null_emb", torch.zeros(n_embd, requires_grad=False))

        self.cfg_prob = cfg_prob

    def forward_cfg(self, x: torch.Tensor) -> torch.Tensor:
        return repeat(self.null_emb, "d -> b t d", b=x.shape[0], t=x.shape[1])

    def forward(
        self, x: torch.Tensor, keep_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        if keep_mask is None:
            keep_mask = prob_mask_like(
                (batch_size, 1, 1), 1.0 - self.cfg_prob, device=x.device
            )
        x_null = repeat(self.null_emb, "d -> b t d", b=batch_size, t=x.shape[1])
        return torch.where(keep_mask, x, x_null)


class SemanticPreNet(nn.Module):

    def __init__(
        self, input_dim: int, n_embd: int, upscale: List[int], downscale: List[int]
    ):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv1d(input_dim, n_embd, kernel_size=1)])
        self.act_fn = nn.GELU()

        for scale in upscale:
            self.blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=scale, mode="nearest"),
                    nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1),
                    self.act_fn,
                    # RMSNorm(n_embd, dim=1),
                )
            )

        for scale in downscale:
            if scale == 1:
                conv = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1)
            else:
                conv = nn.Conv1d(
                    n_embd,
                    n_embd,
                    kernel_size=scale * 2,
                    stride=scale,
                    padding=scale // 2 + scale % 2,
                )

            self.blocks.append(
                nn.Sequential(
                    conv,
                    self.act_fn,
                    # RMSNorm(n_embd, dim=1)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prenet for semantic embedding <-> audio latent alignment

        Args:
            x (torch.Tensor): Input of size [B, D, T]

        Returns:
            torch.Tensor: Output of size [B, D, T]
        """
        for layer in self.blocks:
            x = layer(x)
        return x


@dataclass
class DiffConfig:
    sample_rate: int = 44100
    max_duration: int = 30

    n_embd: int = 1536
    n_layer: int = 24
    n_head: int = 24
    time_n_embd: int = 1024
    cond_n_embd: int = 768

    # diffusion
    cfg_dropout: float = 0.1
    min_t: float = 0.0
    max_t: float = 1.0

    # noise scale
    noise_scale_factor: float = 1.0
    snr_sampler: str = "cosine"

    # Optimizer
    learning_rate: float = 5.0e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.001
    warmup_steps: int = 8000
    hold_steps: int = 200000
    decay_steps: int = 400000
    cycle_steps: int = 1000000
    min_lr: float = learning_rate * 0.1


@dataclass
class DiffResult:
    audio_latents: torch.Tensor
    global_cond: torch.Tensor
    prefix_cond: torch.Tensor
    cross_attn_cond: torch.Tensor
    lyrics_emb: Optional[torch.Tensor] = None
    lyrics_length: Optional[torch.Tensor] = None
    lyrics_mask: Optional[torch.Tensor] = None
    pred_noise: Optional[torch.Tensor] = None
    loss: Optional[LossDict] = None
