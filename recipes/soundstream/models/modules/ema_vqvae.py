import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


class EmbeddingEMA(nn.Module):
    def __init__(self, n_e, e_dim, decay=0.99, eps=1e-5, init_cluster_size=1):
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
        quant_beta,
        same_index_shape=True,
        decay=0.99,
        init_cluster_size=1,
        dist=True,
    ):
        super().__init__()
        self.n_e = quant_token_num
        self.e_dim = quant_token_dim
        self.beta = quant_beta
        self.decay = decay

        # ema for dist
        self.dist = dist

        self.embedding = EmbeddingEMA(
            self.n_e, self.e_dim, decay=decay, init_cluster_size=init_cluster_size
        )
        self.same_index_shape = same_index_shape

    def forward(self, z, warmup=False):
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
