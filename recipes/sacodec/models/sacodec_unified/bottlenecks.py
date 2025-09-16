import torch
from torch import nn
from torch import Tensor
from typing import List, Dict, Tuple
import random
from torch.nn.utils.parametrizations import weight_norm

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
        x = x.float() # TODO: not sure if fp32 needed here
        x = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        mu, logvar = x.chunk(2, dim=1)

        z, kl = sample_vae(mu, logvar)

        kl = self.beta * kl
        return {
            "latents": z,
            "kl_loss": kl,
        }

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

        kl = kl_loss.mean() * self.beta
        return {
            "latents": sample,
            "kl_loss": kl,
        }

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

        sample, kl = self.activation_bottleneck(x), 0
        return {
            "latents": sample,
            "kl_loss": kl,
        }

### Hierarchical

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def upsample_head(input_dim, output_dim):
    return nn.Sequential(
        WNConv1d(input_dim, output_dim*2, kernel_size=1),
        nn.GELU(),
        WNConv1d(output_dim*2, output_dim, kernel_size=1),
    )

class VAEVectorizer(nn.Module):
    def __init__(self, decoder_dim: int, sub_dim: int, beta=1e-5, bottleneck_type="v1"):
        super().__init__()

        if decoder_dim == sub_dim:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            # self.in_proj = upsample_head(decoder_dim, sub_dim)
            self.out_proj = upsample_head(sub_dim, decoder_dim)

        bottlenck_cls = {
            "v1": VAEBottleneck,
            "v2": VAEBottleneckV2,
            "v3": VAEBottleneckV3,
        }[bottleneck_type]
        self.bottleneck: VAEBottleneck = bottlenck_cls(in_channels=decoder_dim, latent_dim=sub_dim, beta=beta)
        
    def forward(self, z):
        # z_emb = self.in_proj(z)  # z_e : (B x D x T)
        z_emb = z
        # TODO: should we use F.normalize before sending to bottleneck? See: ResidualVectorQuantize

        bottleneck_res = self.bottleneck(z_emb)
        vae_latent, kl_loss = bottleneck_res["latents"], bottleneck_res["kl_loss"]
        # TODO: should we use mu instead of z_latent?

        # # TODO: figure out what this is for.
        # z_latent = (
        #     z_emb + (z_latent - z_emb).detach()
        # )  # noop in forward pass, straight-through gradient estimator in backward pass
        expand_latent = self.out_proj(vae_latent)

        return expand_latent, kl_loss, vae_latent

    def features_to_decoder_latents(self, z):
        return self.out_proj(z)
    


class TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)

class ResidualVAEVectorizer(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, sub_dims: list[int], beta=1e-5, vector_dropout=False, skip_semantic_residual=False, bottleneck_type="v1"):
        super().__init__()
        self.sub_dims = sub_dims
        self.bottleneck = nn.Sequential(
            TransposeLast(),
            nn.Linear(in_channels, latent_dim),
            TransposeLast()
        )
        self.vectorizers = nn.ModuleList(
            [
                VAEVectorizer(latent_dim, sub_dim, beta, bottleneck_type=bottleneck_type)
                for sub_dim in sub_dims
            ]
        )
        self.vector_dropout = vector_dropout
        self.skip_semantic_residual = skip_semantic_residual

    def split_features(self, features, dim=1):
        """Splits features into residual sub dimensions"""
        return torch.split(features, self.sub_dims, dim=dim)
    
    def forward(self, audio_features):
        # input: B C T
        residual = self.bottleneck(audio_features)
        kl_loss = 0
        hierarchical_latents = []
        decoder_latents = 0

        if self.training and self.vector_dropout:
            n_sub_dims = len(self.sub_dims)
            n_dropout = random.randint(self.vector_dropout, n_sub_dims+3)
        else:
            n_dropout = len(self.sub_dims) + 1

        for idx, vectorizer in enumerate(self.vectorizers):
            decoder_latent, kl, diffusion_latent = vectorizer(residual)
            kl_loss += kl
            if idx < n_dropout:
                decoder_latents += decoder_latent
            hierarchical_latents.append(diffusion_latent)
            if idx == 0 and self.skip_semantic_residual:
                pass
            else:
                residual = residual - decoder_latent

        hierarchical_latents_concat = torch.cat(hierarchical_latents, dim=1)
        return {
            "latents": decoder_latents, # for decoder reconstruction
            "hierarchical_latents": hierarchical_latents_concat,
            "hierarchical_latents_list": hierarchical_latents,
            "kl_loss": kl,
            "n_vector_dropout": n_dropout
        }
