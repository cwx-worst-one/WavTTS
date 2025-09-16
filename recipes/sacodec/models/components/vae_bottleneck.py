import torch
import torch.nn.functional as F
import torchaudio
import random
import numpy as np
from torch import nn
from torch import Tensor
from typing import List, Dict, Tuple, Optional

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
        return z, kl.mean() * self.beta


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
        return z, kl.mean() * self.beta

class VAEBottleneckResidual(nn.Module):
    "Enforce lower channel encode major information, and higher channel encode residual"
    def __init__(self, in_channels: int, latent_dim: int, beta: float, std: float=0.2):
        super().__init__()
        self.std = std
        self.beta = beta
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.dropout_prob_base = in_channels
        
        # group-wise linear transform, weight shared within each channel
        self.channel_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * latent_dim,
            kernel_size=1,
            groups=in_channels,
            bias=True
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # in: [B, C, L]
        x = x.float()
        B, C, L = x.shape
        base = self.dropout_prob_base

        if self.training:
            mask = torch.ones_like(x)
            for i in range(0, self.in_channels):
                # high channel has larger dropout probability, encourage model to record major info in lower channel
                dorp_prob = torch.log(torch.tensor(i+1)) / torch.log(torch.tensor(base)) * 0.5
                mask[:, i, :] = nn.functional.dropout(mask[:, i, :], p=dorp_prob, training=self.training)
            x = x * mask

        # [B, C, L] -> [B, C*D, L] (D=latent_dim)
        conv_output = self.channel_conv(x)
        # reshape to [B, C, D, L] and sum over channel dimension
        conv_output = conv_output.view(B, C, self.latent_dim, L)
        mean = conv_output.sum(dim=1)  # [B, D, L]

        kl = mean.norm(p=2, dim=-1)
        # some channels will be dropped out, to avoid all-zero feature, the std must be fixed
        z = mean + self.std * torch.randn_like(mean).to(device=x.device)
        return z, kl.mean() * self.beta

# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(0, seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class DeTokBottleneck(nn.Module):
    '''
    Basically a MAE with not only spatial masking but also latent noising, inspried by
    `Latent Denoising Makes Good Visual Tokenizers`
    https://arxiv.org/pdf/2507.15856
    '''

    def __init__(self, in_channels: int, latent_dim: int, beta: float, std: float=1, gamma: float=3, pos_embedding_type: str='absolute'):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, latent_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=latent_dim//4,
                dim_feedforward=latent_dim,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=latent_dim//4,
                dim_feedforward=latent_dim,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.beta = beta
        self.std = std
        self.gamma = gamma
        if pos_embedding_type == 'rotary':
            from apps.mariana.mariana.models.audio.positional_encoding import RotaryPositionalEmbedding
            self.positional_embedding = RotaryPositionalEmbedding(head_dim=latent_dim, max_seq_len=50*60*10)
        elif pos_embedding_type == 'absolute':
            self.positional_embedding = nn.Embedding(50*60*10, latent_dim)
        elif pos_embedding_type == 'sinusoidal':
            self.positional_embedding = RoFormerSinusoidalPositionalEmbedding(num_positions=50*60*10, embedding_dim=latent_dim)
        else:
            raise ValueError(f"pos_embedding_type {pos_embedding_type} not supported")
        self.positional_embedding_type = pos_embedding_type

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.prepare(x) # b x l x c

        if self.training:
            # temporal random mask alone spatial (Eq. 5)
            mask_ratio = max(0.0, random.uniform(-0.1, 0.9))
            B, L, _C = x.shape
            mask = torch.rand(1, L, 1).to(x.device) > mask_ratio
            if mask.sum() != 0:
                unmask_x = torch.randn_like(x)
                x = torch.masked_select(x, mask).reshape(B, -1, _C)

        # encode
        latent = self.encoder(x) # b x l x c

        # latent regularization
        kl = latent.norm(p=2, dim=-1)
        latent = latent + self.std * torch.randn_like(latent).to(device=x.device)

        if self.training:
            # latent noise (Eq. 4)
            noise = torch.randn_like(latent) * self.gamma
            mix_ratio = torch.rand(latent.shape[0], 1, 1)
            mix_ratio = mix_ratio.to(x.device)
            latent = latent * mix_ratio + noise * (1 - mix_ratio)
        
        # decode
        if self.training and mask.sum() != 0:
            latent = torch.masked_scatter(unmask_x, mask, latent)
        x = self.decode(latent)
        return x, kl.mean() * self.beta

    def prepare(self, x: Tensor) -> Tensor:
        # in: b x c x l
        x = x.float().transpose(1, 2) # b x l x c
        x = self.proj_in(x)
        _B, L, _C = x.shape
        if self.positional_embedding_type == 'rotary':
            pos_emb = self.positional_embedding(L)
        elif self.positional_embedding_type == 'absolute':
            pos_emb = self.positional_embedding(torch.arange(L, device=x.device)).unsqueeze(0)
        elif self.positional_embedding_type == 'sinusoidal':
            pos_emb = self.positional_embedding(L).unsqueeze(0)
        x = x + pos_emb
        # out: b x l x c
        return x

    def encode(self, x: Tensor) -> Tensor:
        # in: b x c x l
        x = self.prepare(x) # b x l x c
        # encode clean latent for downsteam tasks
        latent = self.encoder(x)
        return latent # b x l x c

    def decode(self, latent: Tensor) -> Tensor:
        # in: b x l x c
        x = self.decoder(latent)
        x = x.transpose(1, 2) # b x c x l
        return x

def test_DeTokBottleneck():
    torch.manual_seed(0)
    in_channels = 4
    latent_dim = 8
    T = 10
    B = 2
    x = torch.randn(B, in_channels, T)
    print("input", x.shape)
    vae = DeTokBottleneck(in_channels, latent_dim, beta=0.0001, pos_embedding_type='sinusoidal')
    for i in range(10):
        x_, kl = vae(x)
        print("train x_", x_.shape)
        print("kl", kl)
        assert(not torch.isnan(kl.sum()))
        assert(x_.shape == (B, latent_dim, T))

        z = vae.encode(x)
        print("infer z", z.shape)
        assert(z.shape == (B, T, latent_dim))

        x_ = vae.decode(z)
        print("infer x_", x_.shape)
        assert(x_.shape == (B, latent_dim, T))

if __name__ == "__main__":
    test_DeTokBottleneck()