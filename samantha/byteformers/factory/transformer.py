from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from ..components.attention import MultiHeadAttention
from ..components.feedforward import MLP
from .utils import get_clones


@dataclass
class TransformerConfig:
    vocab_size: int
    max_position_embeddings: int
    n_layer: int
    n_head: int
    n_embd: int
    n_inner: int
    dropout: float
    bias: bool = False
    attention_kwargs: Optional[dict] = None


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        dropout: float = 0.1,
        attention_kwargs: dict = {},
        norm_first: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=n_embd, n_heads=n_head, dropout=dropout, **attention_kwargs
        )
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, n_inner=n_inner, dropout=dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prefix_context: Optional[torch.Tensor] = None,
        mask: Any = None,
        attn_bias: Any = None,
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn(self.ln_1(x), context, prefix_context, mask, attn_bias)
            x = x + self.mlp(self.ln_2(x))
        else:
            x = self.ln_1(x + self.attn(x, context, prefix_context, mask, attn_bias))
            x = self.ln_2(x + self.mlp(x))
        return x


class TransformerLayers(nn.Module):
    def __init__(self, block: TransformerBlock, num_layers: int):
        super().__init__()
        self.layers = get_clones(block, num_layers)
        self.n_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prefix_context: Optional[torch.Tensor] = None,
        mask: Any = None,
        attn_bias: Any = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                prefix_context=prefix_context,
                mask=mask,
                attn_bias=attn_bias,
            )
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        layer = TransformerBlock(
            config.n_embd,
            config.n_head,
            config.n_inner,
            config.dropout,
            attention_kwargs=config.attention_kwargs,
        )
        self.transformer = TransformerLayers(layer, config.n_layer)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prefix_context: Optional[torch.Tensor] = None,
        mask: Any = None,
        attn_bias: Any = None,
    ) -> torch.Tensor:
        return self.transformer(
            x,
            context=context,
            prefix_context=prefix_context,
            mask=mask,
            attn_bias=attn_bias,
        )
