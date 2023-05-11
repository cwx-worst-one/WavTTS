import torch
import torch.nn as nn
from torch.nn.functional import layer_norm


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias.

    Bias-less layer norm is being used in more recent T5s and PaLM
    models, and yield greater stability. PyTorch doesn't support
    bias=False
    """

    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.register_buffer("bias", torch.zeros(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(
            x, self.weight.shape, self.weight, bias=self.bias, eps=self.eps
        )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # TODO: Explicit full/mixed precision?
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
