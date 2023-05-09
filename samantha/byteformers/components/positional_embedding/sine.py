import math

import torch
import torch.nn as nn


class SinePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        """Sinusoidal positional embedding as described in the Attention Is All
        You Need (A. Vaswani, 2017) paper.

        - register_buffer is used so that it is not registered as a parameter,
        but should be part of the modules state. It is used for tensors that
        need to be on the same device as the module.

        - persistent=False tells PyTorch to not add the buffer to the state dict
        (e.g. when we save the model)

        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length the module will ever encounter
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos = torch.arange(0, max_len)[:, None]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)])
