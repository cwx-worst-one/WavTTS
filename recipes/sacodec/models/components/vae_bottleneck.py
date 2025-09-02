import torch
import torchaudio
from torch import nn
from torch import Tensor
from typing import List, Dict, Tuple

class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

## VAEBottleneck
@torch.cuda.amp.autocast(enabled=False)
def sample_vae(
    mean: Tensor, scale: Tensor, epsilon: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    stdev = nn.functional.softplus(scale) + 1e-4  # NOTE: tie to beta?
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    # NOTE: the batch-wise sum is done with dim=[1, 2], which is mathematically correct.
    # kl = (mean * mean + var - logvar - 1).sum(dim=[1, 2]).mean()

    # NOTE: switching to batch+length mean to account for variable length
    kl = (mean * mean + var - logvar - 1).sum(dim=[1]).mean() # B x C x L

    return latents, kl
    

class VAEBottleneck(nn.Module):
    "From Stable Audio"

    def __init__(self, in_channels: int, latent_dim: int, beta: float):
        super().__init__()
        self.bottleneck = nn.Linear(in_channels, latent_dim * 2)
        self.beta = beta

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # in: b x c x l
        x = x.float()
        x = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        mu, logvar = x.chunk(2, dim=1)

        z, kl = sample_vae(mu, logvar)

        kl = self.beta * kl
        return z, kl

class VAEBottleneckV2(nn.Module):
    "From Soundstream"

    def __init__(self, in_channels: int, latent_dim: int, beta: float):
        super().__init__()
        self.bottleneck = nn.Linear(in_channels, latent_dim * 2)
        self.beta = beta

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor, deterministic=False) -> Dict[str, Tensor]:
        # in: b x c x l
        x = x.float()
        x = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        mean, logvar = x.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        if deterministic:
            kl_loss = torch.FloatTensor([0.0]).to(x.device)
            sample = mean
        else:
            # NOTE: switching to batch+length mean to account for variable length
            kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1])
            # kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2])
            sample = mean + std * torch.randn_like(mean).to(device=x.device)
        sample = sample.clamp(-3, 3) / 3
        return sample, kl_loss.mean() * self.beta

class VAEBottleneckV3(nn.Module):
    "From Music2Latent. Tanh activation"

    def __init__(self, in_channels: int, latent_dim: int, beta: float):
        super().__init__()
        self.bottleneck = nn.Linear(in_channels, latent_dim)
        self.activation_bottleneck = nn.Tanh()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor, deterministic=False) -> Dict[str, Tensor]:
        # in: b x c x l
        x = x.float()
        x = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        return self.activation_bottleneck(x), 0


class VAEBottleneckV4(nn.Module):
    "sigma-vae variant from latentLM"
    "https://arxiv.org/pdf/2412.08635"

    def __init__(self, in_channels: int, latent_dim: int, beta: float, std: float=0.75):
        super().__init__()
        self.bottleneck = nn.Linear(in_channels, latent_dim)
        self.beta = beta
        self.std = std

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # in: b x c x l
        x = x.float()
        mean = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        value = self.std / 0.8
        
        batch_size = mean.shape[0]
        std = torch.randn([batch_size,1,1]).to(mean.device) * value
        std = nn.functional.softplus(std) + 1e-4  # NOTE: tie to beta?
        kl = mean.norm(p=2, dim=-1)
        z = mean + std * torch.randn_like(mean).to(device=x.device)
        return z, kl.mean()


class VAEBottleneckV5(nn.Module):
    "VAE with high-res and low-res latents"

    def __init__(self, in_channels: int, latent_dim: int, beta: float, latent_dim_lores: int):
        super().__init__()
        self.bottleneck = nn.Linear(in_channels, latent_dim * 2)
        self.linear_fine2corse = nn.Linear(latent_dim, latent_dim_lores)
        self.linear_corse_to_fine = nn.Linear(latent_dim_lores, latent_dim)
        self.beta = beta

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # in: b x c x l
        x = x.float()

        # high_res residual
        x_residual = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        mu, logvar = x_residual.chunk(2, dim=1)
        z, kl = sample_vae(mu, logvar)
        z = z.transpose(1, 2)
        kl = self.beta * kl

        # low_res
        z_low_res = self.linear_fine2corse(z)
        # shape low-res, add residual, and concat two part
        z_low_res_reshape = self.linear_corse_to_fine(z_low_res)
        z_residual = z - z_low_res_reshape
        merged_z = torch.cat([z_residual, z_low_res_reshape], dim=-1)
        merged_z = merged_z.transpose(1, 2)
        return merged_z, kl

class VAEBottleneckConstSigma(nn.Module):
    "sigma-vae variant with constant sigma"

    def __init__(self, in_channels: int, latent_dim: int, beta: float, std: float=0.2):
        super().__init__()
        self.bottleneck = nn.Linear(in_channels, latent_dim)
        self.beta = beta
        self.std = std

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # in: b x c x l
        x = x.float()
        mean = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        kl = mean.norm(p=2, dim=-1)
        z = mean + self.std * torch.randn_like(mean).to(device=x.device)
        return z, kl.mean()
