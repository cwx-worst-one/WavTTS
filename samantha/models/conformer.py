from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from samantha.components.activations import Activation, build_activation
from samantha.components.attention import MultiHeadAttention
from samantha.components.feedforward.mlp import MLP
from samantha.models.base import BaseModel


@dataclass
class ConformerConfig:
    n_layer: int
    n_head: int
    n_embd: int
    mlp_dropout: float = 0.0
    attn_dropout: float = 0.0
    rms_norm_epsilon: float = 1e-6
    activation_fn: Activation = Activation.GLU
    attention_kwargs: Optional[dict] = field(default_factory=dict)
    use_rotary_embeddings: bool = True
    conv_expansion_factor: int = 2
    conv_kernel_size: int = 5
    conv_dropout: float = 0.0
    causal: bool = False
    conv_causal: bool = False


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        n_embd: int,
        activation_fn: Activation,
        causal: bool,
        expansion_factor: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        inner_dim = n_embd * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(n_embd),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(n_embd, inner_dim * 2, 1),
            build_activation(activation_fn),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, n_embd, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        n_inner = self.get_n_inner(config.n_embd)
        self.mlp_1 = MLP(config.n_embd, n_inner=n_inner, dropout=config.mlp_dropout)

        self.attn_norm_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(
            config.n_embd,
            config.n_head,
            dropout=config.attn_dropout,
            causal=config.causal,
            use_rotary_embeddings=config.use_rotary_embeddings,
            **config.attention_kwargs,
        )
        self.conv = ConformerConvModule(
            n_embd=config.n_embd,
            activation_fn=config.activation_fn,
            causal=config.conv_causal,
            expansion_factor=config.conv_expansion_factor,
            kernel_size=config.conv_kernel_size,
            dropout=config.conv_dropout,
        )
        self.mlp_2 = MLP(config.n_embd, n_inner=n_inner, dropout=config.mlp_dropout)

        self.mlp_norm_1 = nn.LayerNorm(config.n_embd)
        self.mlp_norm_2 = nn.LayerNorm(config.n_embd)
        self.post_norm = nn.LayerNorm(config.n_embd)
        self.mlp_scale = 0.5

    def get_n_inner(self, n_embd: int) -> int:
        return 4 * n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_scale * self.mlp_norm_1(self.mlp_1(x)) + x
        x = self.attn_norm_1(self.attn(x)) + x
        x = self.conv(x) + x
        x = self.mlp_scale * self.mlp_norm_2(self.mlp_2(x)) + x
        return self.post_norm(x)


class Conformer(BaseModel):
    """Conformer model as proposed in the paper from Anmol G. et al.:
    "Conformer: Convolution-augmented Transformer for Speech Recognition"

    This paper studied how to combine convolution neural networks and
    transformers to model both local and global dependencies of an audio
    sequence in a parameter-efficient way. To this regard, we propose the
    convolution-augmented transformer for speech recognition, named Conformer.
    Conformer significantly outperforms the previous Transformer and CNN based
    models achieving state-of-the-art accuracies
    """

    def __init__(self, config: ConformerConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ConformerBlock(config) for _ in range(config.n_layer)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        return x
