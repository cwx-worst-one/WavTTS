import torch
import torch.nn as nn

from .encoder import Encoder
from .generator import Generator


class AutoencoderKL(nn.Module):

    def __init__(self, hp, device='cpu', stage='enc'):
        super().__init__()
        self.hp = hp
        self.device = device
        assert stage in {'dec', 'enc'}
        self.stage = stage
        self.encoder = Encoder(hp)
        self.mean_logvar_conv = nn.Conv1d(hp.quant_token_dim, hp.latent_dim * 2, 1)
        self.decoder = Generator(hp)

    def forward(self, xs, x_lens):
        zs, z_lens = [], []
        max_len = 0
        for i, x in enumerate(xs):
            if x.ndim == 1:
                x = x[:x_lens[i]].to(x.device)[None, None]
            if self.stage == 'enc':
                zs.append(self.encode(x))
            elif self.stage == 'dec':
                zs.append(self.decode(x))
            z_lens.append(zs[-1].shape[-1])
            max_len = max(max_len, z_lens[-1])
        for i, z in enumerate(zs):
            if z.shape[-1] < max_len:
                zs[i] = torch.cat([z, torch.zeros(1, z.size(1), max_len-z.size(-1), device=z.device)], -1)
        return torch.cat(zs, 0), torch.tensor(z_lens).long()

    def encode(self, x, deterministic=False):
        x = self.encoder(x)
        x = self.sample(x, deterministic)
        return x

    def sample(self, x, deterministic=False):
        mean_logvar = self.mean_logvar_conv(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        if deterministic:
            sample = mean
        else:
            sample = mean + std * torch.randn_like(mean).to(device=x.device)
        sample = sample.clamp(-3, 3) / 3
        if self.training:
            if deterministic:
                kl_loss = torch.FloatTensor([0.0]).to(x.device)
            else:
                kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2])
            return sample, kl_loss, std.mean()
        return sample
        
    def decode(self, x):
        if self.training:
            x, kl_loss, std_mean = x
        decoder_out = self.decoder(x.clamp(-1, 1))
        if self.training:
            return decoder_out, kl_loss, std_mean
        return decoder_out

    def get_quant_output_from_index(self, index):
        # index: should be list/tuple or tensor
        # if index is a list/tuple, each tensor's shape must be [b, t]
        # if index is a tensor, the tensor's shape must be [b, n_codebook, t]
        quant_outs = []
        if isinstance(index, (tuple, list)):
            for i, ind in enumerate(index):
                embedding = self.quant_vaes[i].embedding(ind) # [b, t, d]
                quant_outs.append(embedding.transpose(1, 2)) # [b, d, t]
        elif isinstance(index, torch.Tensor):
            for i in range(index.size(1)):
                ind = index[:, i, :]
                embedding = self.quant_vaes[i].embedding(ind) # [b, t, d]
                quant_outs.append(embedding.transpose(1, 2)) # [b, d, t]
        else:
            print("Unexpected index type: {}".format(type(index)))
            raise Exception

        quant_out = sum(quant_outs)
        return quant_out

    def convert_sync_batchnorm(self, rank, num_gpus):
        for quant_vae in self.quant_vaes:
            quant_vae.convert_sync_batchnorm(rank, num_gpus)
