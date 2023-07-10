import torch
import torch.nn as nn

from recipes.semantic_coder.models.decoder import Decoder
from recipes.semantic_coder.models.encoder import Encoder


class VAE(nn.Module):
    def __init__(
        self,
        feature_dim,
        latent_dim,
        downsample_rates,
        upsample_rates,
        encoder_base_dim,
        decoder_base_dim,
    ):
        super().__init__()
        self.encoder = Encoder(
            feature_dim=feature_dim,
            model_dim=encoder_base_dim,
            strides=downsample_rates,
        )

        self.mean_logvar_conv = nn.Conv1d(self.encoder.enc_dim, latent_dim * 2, 1)

        self.decoder = Decoder(
            input_channel=latent_dim,
            channels=decoder_base_dim,
            rates=upsample_rates,
            feature_dim=feature_dim,
        )

    def forward(self, x, deterministic=False):
        encoder_out = self.encode(x)
        sample, kl_loss, std_mean = self.sample(
            encoder_out, deterministic=deterministic
        )
        decoder_out = self.decode(sample)
        return decoder_out, kl_loss, std_mean

    def encode(self, x):
        encoder_out = self.encoder(x)
        return encoder_out

    def sample(self, x, deterministic=False):
        mean_logvar = self.mean_logvar_conv(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        if deterministic:
            kl_loss = torch.FloatTensor([0.0]).to(x.device)
            sample = mean
        else:
            kl_loss = 0.5 * torch.sum(
                torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2]
            )
            sample = mean + std * torch.randn_like(mean).to(device=x.device)
        sample = sample.clamp(-3, 3) / 3
        return sample, kl_loss.mean(), std.mean()

    def decode(self, x):
        decoder_out = self.decoder(x)
        return decoder_out
