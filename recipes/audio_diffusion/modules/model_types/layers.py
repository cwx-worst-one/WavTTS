import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmaEmbedding(nn.Module):  # pragma: no cover
    def __init__(self, embedding_dim=64, output_dim=64):
        super().__init__()
        self.input_mapping = nn.Linear(1, embedding_dim)
        self.hidden = nn.Linear(2 * embedding_dim, output_dim)
        self.output_mapping = nn.Linear(output_dim, output_dim)
        freqs = torch.cat(
            [torch.arange(embedding_dim), torch.arange(embedding_dim)], dim=0
        ) / (embedding_dim)
        self.register_buffer("freq_mults", freqs.view(1, -1))
        phases = (
            torch.cat([torch.zeros(embedding_dim), torch.ones(embedding_dim)], dim=0)
            * 0.5
            * math.pi
        )
        self.register_buffer("phase_offsets", phases.view(1, -1))

    def forward(self, inp):
        inp = inp.view(inp.shape[0], 1)
        expanded = self.input_mapping(inp)
        stacked = torch.cat([expanded, expanded], dim=-1)
        sinusoids = torch.sin(
            self.freq_mults * 2 * math.pi * stacked + self.phase_offsets
        )
        sinusoids = F.gelu(self.hidden(sinusoids))
        out = self.output_mapping(sinusoids)
        return out


class FilmLayer(nn.Module):  # pragma: no cover
    def __init__(self, input_dim, conditioning_dim=512, activation=F.gelu):
        super().__init__()
        self.conditioning_map = nn.Linear(conditioning_dim, conditioning_dim)
        self.gamma_beta_layer = nn.Linear(conditioning_dim, 2 * input_dim)
        self.activation = activation

    def forward(self, x, conditioning):
        conditioning = self.activation(self.conditioning_map(conditioning))
        gamma_beta = self.gamma_beta_layer(conditioning)
        gamma_beta = gamma_beta.unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        x = (1 + gamma) * x + beta
        return x


class GRN(nn.Module):  # pragma: no cover
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
