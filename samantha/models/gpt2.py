import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from samantha.components.attention import _is_blocksparse_available
from samantha.components.attention.base import MultiHeadAttention
from samantha.components.feedforward import MLP
from samantha.components.normalization import LayerNorm
from samantha.models.utils import get_clones

from .base import BaseModel


@dataclass
class GPT2Config:
    """Bias is True for Linears and LayerNorms in the original GPT-2.

    Setting to False is a bit better and faster (A. Karpathy)
    """

    vocab_size: int
    n_positions: int
    n_layer: int
    n_head: int
    n_embd: int
    n_inner: int
    resid_pdrop: float
    embd_pdrop: float
    attn_pdrop: float
    mlp_bias: bool = False
    layer_norm_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    attention_kwargs: Optional[dict] = field(default_factory=dict)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = LayerNorm(
            config.n_embd, bias=config.layer_norm_bias, eps=config.layer_norm_epsilon
        )
        self.attn = MultiHeadAttention(
            d_model=config.n_embd,
            n_heads=config.n_head,
            causal=True,
            dropout=config.attn_pdrop,
            **config.attention_kwargs,
        )
        self.ln_2 = LayerNorm(
            config.n_embd, bias=config.layer_norm_bias, eps=config.layer_norm_epsilon
        )
        self.mlp = MLP(
            n_embd=config.n_embd,
            n_inner=config.n_inner,
            dropout=config.resid_pdrop,
            bias=config.mlp_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if _is_blocksparse_available:
    from samantha.components.attention import BlockSparseAttention

    class GPT2SparseBlock(GPT2Block):
        def __init__(self, config: GPT2Config):
            super().__init__(config)
            block_size = 32
            layout = torch.ones(
                config.n_positions // block_size,
                config.n_positions // block_size,
                dtype=torch.long,
            ).tril()

            self.attn = BlockSparseAttention(
                layout=layout,
                block_size=block_size,
                d_model=config.n_embd,
                n_heads=config.n_head,
                causal=True,
                **config.attention_kwargs,
            )


class GPT2Model(BaseModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config=config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)

        layer = GPT2Block(config)
        self.h = get_clones(layer, config.n_layer)
        self.ln_f = LayerNorm(
            config.n_embd, bias=config.layer_norm_bias, eps=config.layer_norm_epsilon
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=self.config.initializer_range
                    / math.sqrt(2 * self.config.n_layer),
                )

    def _init_weights(self, module) -> None:
        """[1] Reinitialize selected weights subject to the OpenAI GPT-2 Paper
        Scheme: A modified initialization which accounts for the accumulation
        on the residual path with model depth. Scale the weights of residual
        layers at initialization by a factor of 1/√N where N is the # of
        residual layers.

        Source:
        https://openai.com/blog/better-language-models/

        Reference (Megatron-LM):
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py

        Args:
            module (_type_): _description_
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[1]
        pos = rearrange(
            torch.arange(0, s, dtype=torch.long, device=x.device), "s -> 1 s"
        )
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class GPT2(BaseModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config=config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.transformer.apply(self.transformer._init_weights)

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get
        subtracted. The token embeddings would too, except due to the
        parameter sharing these params are actually used as weights in
        the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        return self.lm_head(x)
