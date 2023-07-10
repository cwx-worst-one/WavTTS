import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from recipes.semantic_coder.models.blocks import (
    CausalConv1d,
    CausalWNResBlock,
    WNResBlock,
)
from recipes.semantic_coder.models.utils import init_weights


class EMAEmbedding(nn.Module):
    def __init__(self, n_e, e_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps

        weight = torch.randn(n_e, e_dim)
        weight[0] = 0.0
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(n_e) + 8, requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)
        # make sure the greedy algorithm to get best estimation
        self.weight[0] = 0.0
        self.embed_avg[0] = 0.0

    @torch.no_grad()
    def entropy(self):
        p = self.cluster_size / self.cluster_size.sum()
        entropy = (-p * p.log()).sum()
        return entropy


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        quant_token_num,
        quant_token_dim,
        same_index_shape=True,
        decay=0.99,
        dist=True,
    ):
        super().__init__()
        self.n_e = quant_token_num
        self.e_dim = quant_token_dim
        self.decay = decay

        # ema for dist
        self.dist = dist

        self.embedding = EMAEmbedding(self.n_e, self.e_dim, decay=decay)
        self.same_index_shape = same_index_shape

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c h -> b h c").contiguous()  # [b, h, c]
        z_flattened = z.view(-1, self.e_dim)  # [b*h, c]

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # [b*h]
        z_q = self.embedding(min_encoding_indices).view(
            z.shape
        )  # [b*h, c] -> [b, h, c]

        # EMA updating, use for
        if self.training and self.embedding.update:
            one_hot = F.one_hot(min_encoding_indices, self.n_e).type(
                z.dtype
            )  # [b*h, k]
            # EMA cluster size
            one_hot_sum = one_hot.sum(0)  # [k]
            if self.dist:
                torch.distributed.all_reduce(one_hot_sum)
            self.embedding.cluster_size_ema_update(one_hot_sum)
            # EMA embedding average
            embed_sum = (
                one_hot.transpose(0, 1) @ z_flattened
            )  # [k, b*h] * [b*h, c] = [k, c]
            if self.dist:
                torch.distributed.all_reduce(embed_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.n_e)

        loss = torch.mean((z_q.detach() - z) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, "b h c -> b c h").contiguous()

        if self.same_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2]
            )

        return z_q, loss, min_encoding_indices


class Encoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, quant_token_dim=16):
        super().__init__()
        self.conv_pre = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            WNResBlock(channels=hidden_dim, kernel_size=3, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=3, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=1, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=3, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=1, dilation=(1, 1, 1)),
        )
        self.encoder_bn = nn.BatchNorm1d(quant_token_dim, affine=False, momentum=0.05)

        self.to_hidden = CausalConv1d(
            hidden_dim, quant_token_dim, kernel_size=3, padding=1
        )
        self.register_buffer("cnt", torch.FloatTensor([0]))

        self.conv_pre.apply(init_weights)
        self.to_hidden.conv.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.resblocks(x) / 5
        x = self.to_hidden(x)
        x = self.encoder_bn(x)
        if self.training:
            x = x + torch.randn_like(x) * ((1e4 - self.cnt).clamp(0) / 1e4)
            self.cnt.add_(1)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, quant_token_dim=16):
        super().__init__()
        self.conv_pre = nn.Conv1d(quant_token_dim, hidden_dim, kernel_size=5, padding=2)
        self.resblocks = nn.Sequential(
            CausalWNResBlock(channels=hidden_dim, kernel_size=1, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=3, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=1, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=3, dilation=(1, 1, 1)),
            CausalWNResBlock(channels=hidden_dim, kernel_size=1, dilation=(1, 1, 1)),
        )
        self.to_output = CausalConv1d(hidden_dim, input_dim, kernel_size=5, padding=2)

        self.conv_pre.apply(init_weights)
        self.to_output.conv.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        # x = self.bn(x)
        x = self.resblocks(x) / 5
        x = self.to_output(x)
        return x


class VQVAE(nn.Module):
    def __init__(
        self, feature_dim=1024, hidden_dim=512, quant_token_num=16, quant_token_dim=512
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.quant_token_num = quant_token_num
        self.quant_token_dim = quant_token_dim
        self.encoder = Encoder(self.feature_dim, self.hidden_dim, self.quant_token_dim)
        self.vq = EMAVectorQuantizer(
            quant_token_num=self.quant_token_num, quant_token_dim=self.quant_token_dim
        )
        self.decoder = Decoder(self.feature_dim, self.hidden_dim, self.quant_token_dim)

    def forward(self, x):
        encoder_out = self.encoder(x)
        quant_out, quant_loss, quant_index = self.quant(encoder_out)
        decoder_out = self.decoder(quant_out)

        return decoder_out, quant_loss, quant_index

    def quant(self, x):
        quant_out, quant_loss, quant_index = self.vq(x)
        return quant_out, quant_loss, quant_index
