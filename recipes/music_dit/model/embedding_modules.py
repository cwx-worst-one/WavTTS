import torch
import torch.nn as nn
from torch import Tensor, einsum
import typing as tp
from math import ceil, floor, log, pi, log2
from einops import rearrange, reduce, repeat
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from recipes.bigmusic.lightning.embedding_modules import ContinuousEmbedder

class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, 1]
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class NumberEmbedder(ContinuousEmbedder):
    def __init__(self, embedding_dim, input_dim=None, add_sos=False, add_eos=False, add_none=False, min_val=0, max_val=1024):
        if input_dim is None:
            input_dim = embedding_dim
        super().__init__(input_dim, embedding_dim, add_sos, add_eos, add_none)
        self.min_val = min_val
        self.max_val = max_val

        self.embedding = nn.Sequential(
            LearnedPositionalEmbedding(input_dim),
            nn.Linear(in_features=input_dim + 1, out_features=input_dim),
        )

    def get_embeds(self, requires, batch):
        # Cast the inputs to floats
        batch = batch.float().clamp(self.min_val, self.max_val)
        normalized_floats = (batch - self.min_val) / (self.max_val - self.min_val)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        # Get embedding
        float_embeds = self.embedding(normalized_floats).unsqueeze(1)
        return float_embeds

def test_number_embedder():
    batch = {"duration": torch.rand([7, 1])}
    embedder = NumberEmbedder(
        input_dim=512, 
        embedding_dim=1024, 
        add_sos=True,
        add_eos=True,
    )
    embeds = embedder.embed("A", batch['duration'], with_sos=True)
    print(embeds.shape)

if __name__ == "__main__":
    test_number_embedder()

