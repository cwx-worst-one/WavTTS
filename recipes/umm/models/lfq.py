import torch
import numpy as np
import math

from torch import nn
from torch.nn import functional as F


def log(t, eps=1e-10):
    return t.clamp(min=eps).log()


def binary_entropy(prob):
    return -prob * log(prob) - (1 - prob) * log(1 - prob)


class LookupFreeQuantizer(nn.Module):
    def __init__(
        self, codebook_size, codebook_dim, same_index_shape=True, decay=0.99, dist=True, loss_type='entropy'
    ):
        super().__init__()
        assert loss_type.lower() in ['l2', 'entropy']
        self.bin_dim = int(math.ceil(math.log2(codebook_size)))
        self.codebook_size = 2 ** self.bin_dim
        self.codebook_dim = codebook_dim
        self.register_buffer("ranges", torch.LongTensor([2 ** i for i in range(self.bin_dim)]))
        self.same_index_shape = same_index_shape
        self.decay = decay
        self.dist = dist
        self.loss_type = loss_type

        self.proj_in = nn.Linear(codebook_dim, self.bin_dim, bias=False)
        self.proj_out = nn.Linear(self.bin_dim, codebook_dim * self.bin_dim, bias=False)
        self.embedding = EMAEmbedding(codebook_size, codebook_dim) # only for entropy

    def forward(self, z):
        _b, _t, _d = z.shape
        proj_z = self.proj_in(z).tanh()
        z_q = ((proj_z > 0).float() * 2 - 1).float()
        if self.loss_type == 'l2':
            # l2 loss
            loss = ((proj_z - z_q) ** 2).mean() * 0.01
        else:
            # entropy loss
            prob = proj_z.sigmoid() # [b, t, d]
            bit_entropy = binary_entropy(prob).mean()
            avg_prob = prob.mean(dim=1) # [b, t, d] -> [b, d]
            codebook_entropy = binary_entropy(avg_prob).mean()
            loss = 0.1 * (bit_entropy - 2.5 * codebook_entropy) + ((proj_z - z_q) ** 2).mean() * 0.01

        min_encoding_indices = self.get_index(z_q)
        if self.training:
            one_hot = F.one_hot(min_encoding_indices.reshape(-1), self.codebook_size).type(z.dtype)
            one_hot_sum = one_hot.sum(0)
            if self.dist:
                torch.distributed.all_reduce(one_hot_sum)
            self.embedding.cluster_size_ema_update(one_hot_sum)
        # preserve gradients
        z_q = proj_z + (z_q - proj_z).detach()
        z_q = self.proj_out(z_q).reshape(_b, _t, self.bin_dim, self.codebook_dim).sum(dim=-2)
        return z_q, min_encoding_indices, loss

    @torch.no_grad()
    def get_index(self, z_q):
        point = (z_q == 1).long()
        index = (point * self.ranges).sum(dim=-1)
        return index

    @torch.no_grad()
    def entropy(self):
        return self.embedding.entropy()


class EMAEmbedding(nn.Module):
    def __init__(self, codebook_size, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("cluster_size", torch.zeros(codebook_size) + 8)

    def forward(self, embed_id):
        return self.entropy()

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size.data, alpha=1 - self.decay
        )

    @torch.no_grad()
    def entropy(self):
        p = self.cluster_size / self.cluster_size.sum()
        p = p.clamp(1e-9)
        entropy = (-p * p.log()).sum()
        return entropy
