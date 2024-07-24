from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from recipes.research.diff.diff import NumberEmbedder
from samantha.transforms.chroma import ChromaSpectrogram


class IntConditioner(nn.Module):

    def __init__(self, n_embd: int, min_val: int = 0, max_val: int = 512):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, n_embd).requires_grad_(
            True
        )

    def forward(self, ints: List[int], device=None) -> Dict[str, torch.Tensor]:
        ints = torch.tensor(ints, dtype=torch.long, device=device)
        ints = ints.clamp(self.min_val, self.max_val)
        int_embeds = self.int_embedder(ints).unsqueeze(1)
        return {
            "embeds": int_embeds,
            "attention_mask": torch.ones(
                int_embeds.shape[0], int_embeds.shape[1], device=int_embeds.device
            ),
            "ints": ints,
        }


class NumberConditioner(nn.Module):
    """Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings."""

    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: List[float], device=None) -> Dict[str, torch.Tensor]:
        # Cast the inputs to floats
        floats = [float(x) for x in floats]

        floats = torch.tensor(floats).to(device)

        floats = floats.clamp(self.min_val, self.max_val)

        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        float_embeds = self.embedder(normalized_floats).unsqueeze(1)
        return {
            "embeds": float_embeds,
            "attention_mask": torch.ones(
                float_embeds.shape[0], 1, device=float_embeds.device
            ),
            "floats": floats,
        }


class ChromaConditioner(nn.Module):

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_chroma: int,
        n_embd: int,
    ):
        super().__init__()
        self.transform = ChromaSpectrogram(
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            normalized=False,
            n_chroma=n_chroma,
        )
        self.chroma_embedder = nn.Linear(n_chroma, n_embd, bias=False)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.mean(dim=1, keepdim=True)

        x = x.squeeze(dim=1)
        chroma = self.transform(x)[:, :, :-1].transpose(1, 2)
        chroma = F.normalize(chroma, p=2, dim=-1)

        embeds = self.chroma_embedder(chroma)
        return {
            "embeds": embeds,
            "mask": torch.ones(embeds.shape[0], embeds.shape[1], device=embeds.device),
            "chroma": chroma,
        }
