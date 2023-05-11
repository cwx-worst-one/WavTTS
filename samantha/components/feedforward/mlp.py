import torch
import torch.nn as nn

from ...components.activations import Activation, build_activation


class MLP(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_inner: int,
        bias: bool = True,
        activation_fn: Activation = Activation.GeLU,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, n_inner, bias=bias)
        self.c_proj = nn.Linear(n_inner, n_embd, bias=bias)
        self.act_fn = build_activation(activation_fn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act_fn(x)
        x = self.c_proj(x)
        return self.dropout(x)


class GatedMLP(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_inner: int,
        bias: bool = True,
        activation_fn: Activation = Activation.GeLU,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.c_fc1 = nn.Linear(n_embd, n_inner, bias=bias)
        self.c_fc2 = nn.Linear(n_embd, n_inner, bias=bias)
        self.c_proj = nn.Linear(n_inner, n_embd, bias=bias)
        self.act_fn = build_activation(activation_fn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return self.dropout(x)
