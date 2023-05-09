import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..components.activations import Activation
from ..components.attention import SeerAttention
from ..components.feedforward import GatedMLP
from ..components.normalization import RMSNorm
from ..factory.utils import get_clones
from .base import BaseModel


@dataclass
class SeerConfig:
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    logit_num: int
    n_priors: int
    n_seers: int
    seq_len: int = 2000
    attn_bias: bool = False
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    rms_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    activation_fn: Activation = Activation.SiLU
    attention_kwargs: Optional[dict] = field(default_factory=dict)


class SeerBlock(nn.Module):
    def __init__(self, config: SeerConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = SeerAttention(
            d_model=config.n_embd,
            n_heads=config.n_head,
            n_priors=config.n_priors,
            n_seers=config.n_seers,
            seq_len=config.seq_len,
            bias=config.attn_bias,
            **config.attention_kwargs,
        )
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        n_inner = self.get_n_inner_dim(config.n_embd)
        config.n_inner = n_inner

        config.n_inner = n_inner
        self.mlp = GatedMLP(
            n_embd=config.n_embd,
            n_inner=config.n_inner,
            bias=config.mlp_bias,
            activation_fn=config.activation_fn,
            dropout=config.mlp_dropout,
        )

    def get_n_inner_dim(self, n_embd: int):
        n_inner = 4 * n_embd
        n_inner = int(2 * n_inner / 3)
        N = 256
        return ((n_inner - 1) // N) * N + N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SeerModel(BaseModel):
    def __init__(self, config: SeerConfig):
        super().__init__(config=config)
        assert config.vocab_size is not None
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        layer = SeerBlock(config)
        self.h = get_clones(layer, config.n_layer)
        self.ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

    def _init_weights(self, module: nn.Module) -> None:
        """Reinitialize selected weights subject to the OpenAI GPT-2 Paper
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
                module.weight,
                mean=0.0,
                std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer),
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class Seer(nn.Module):
    def __init__(self, config: SeerConfig):
        super().__init__()
        self.config = config
        self.transformer = SeerModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.logit_num, bias=False)

    def _init_weights(self, module: nn.Module) -> None:
        self.transformer._init_weights(module)

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
