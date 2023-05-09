import torch.nn as nn
from torch import Tensor
from torch.nn.functional import normalize


class NormalizeMTR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return normalize(x, dim=0)


class EmbeddingExtractor(nn.Module):
    def __init__(self, sample_rate: int) -> None:
        """
        sample_rate: how many samples per embedding
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.log_count = 0

    def forward(self, x: Tensor, st: int, en: int) -> Tensor:
        """
        Inputs:
            x: embedding tensor, (num_segments, embed_dim)
            st: start sample
            en: end sample

        Returns:
            1D averaged embedding vector, (embed_dim, 1)
        """
        st_idx = min(x.shape[0] - 1, int(st / self.sample_rate))
        en_idx = min(x.shape[0] - 1, int(en / self.sample_rate)) + 1
        en_idx = max(st_idx + 1, en_idx)
        return x[st_idx:en_idx].mean(dim=0)
