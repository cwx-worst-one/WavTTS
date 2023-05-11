import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from typing_extensions import Self

from ..components.activations import Activation
from ..components.attention import MultiHeadAttention
from ..components.feedforward import GatedMLP
from ..components.normalization import RMSNorm
from ..factory.utils import get_clones
from .base import BaseModel


@dataclass
class LlamaConfig:
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: Optional[int] = None
    logit_num: Optional[int] = None
    attn_bias: bool = False
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    rms_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    activation_fn: Activation = Activation.SiLU
    attention_kwargs: Optional[dict] = field(default_factory=dict)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return llama_configs[name]


llama_configs = {
    "300M": LlamaConfig(n_layer=24, n_head=16, n_embd=1024),
    "7B": LlamaConfig(n_layer=32, n_head=32, n_embd=4096),
    "13B": LlamaConfig(n_layer=40, n_head=40, n_embd=5120),
    "30B": LlamaConfig(n_layer=60, n_head=52, n_embd=6656),
    "65B": LlamaConfig(n_layer=80, n_head=64, n_embd=8192),
}


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = MultiHeadAttention(
            d_model=config.n_embd,
            n_heads=config.n_head,
            bias=config.attn_bias,
            causal=True,
            use_rotary_embeddings=True,
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

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class LlamaModel(BaseModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config=config)
        assert config.vocab_size is not None
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        layer = LlamaBlock(config)
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

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        x = self.wte(x)
        for block in self.h:
            x = block(x, kv_cache=kv_cache)
        return self.ln_f(x)


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.transformer = LlamaModel(config)
        output_dim = config.logit_num
        if output_dim is None:
            output_dim = config.vocab_size
        self.lm_head = nn.Linear(config.n_embd, output_dim, bias=False)

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

    def forward(self, x: torch.Tensor, kv_cache=None, last_logit_only: bool = False):
        x = self.transformer(x, kv_cache=kv_cache)
        if last_logit_only:
            x = x[:, -1]
        return self.lm_head(x)
