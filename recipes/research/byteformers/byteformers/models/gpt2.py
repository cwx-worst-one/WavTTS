import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
# from byteformers.components.attention import _is_blocksparse_available
# from byteformers.components.attention.base import MultiHeadAttention
from samantha.components.ctiga.mha import MHA as MultiHeadAttention
from byteformers.components.feedforward import MLP
from byteformers.components.normalization import LayerNorm
from byteformers.models.utils import get_clones
from einops import rearrange

from .base import BaseModel


@dataclass
class GPT2Config:
    """Bias is True for Linears and LayerNorms in the original GPT-2.

    Setting to False is a bit better and faster (A. Karpathy)
    """

    n_layer: int
    n_head: int
    n_embd: int
    n_inner: int
    resid_pdrop: float
    embd_pdrop: float
    attn_pdrop: float
    vocab_size: Optional[int] = None
    logit_num: Optional[int] = None
    max_seq_len: Optional[int] = None
    mlp_bias: bool = False
    layer_norm_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    attention_kwargs: Optional[dict] = field(default_factory=dict)
    is_causal: bool = True
    use_rotary_embeddings: bool = True


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.ln_1 = LayerNorm(
            config.n_embd, bias=config.layer_norm_bias, eps=config.layer_norm_epsilon
        )
        # FA 1
        # self.attn = MultiHeadAttention(
        #     d_model=config.n_embd,
        #     n_heads=config.n_head,
        #     dropout=config.attn_pdrop,
        #     is_causal=config.is_causal,
        #     use_rotary_embeddings=config.use_rotary_embeddings,
        #     **config.attention_kwargs,
        # )

        # FA 2
        d_k = config.n_embd // config.n_head
        rotary_emb_dim = d_k if config.use_rotary_embeddings else 0
        self.attn = MultiHeadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            cross_attn=False,
            qkv_proj_bias=True,
            out_proj_bias=True,
            dropout=0.0,
            causal=config.is_causal,
            layer_idx=layer_idx,
            rotary_emb_dim=rotary_emb_dim,
            fused_bias_fc=True,
            return_residual=True,
            use_flash_attn=True,
            version="2.3",
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

    def forward(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
        x, residual = self.attn(self.ln_1(x))
        x = x + residual
        x = x + self.mlp(self.ln_2(x))
        return x


# if _is_blocksparse_available:
#     from byteformers.components.attention import BlockSparseAttention

#     class GPT2SparseBlock(GPT2Block):
#         def __init__(self, config: GPT2Config):
#             super().__init__(config)
#             block_size = 32
#             layout = torch.ones(
#                 config.max_seq_len // block_size,
#                 config.max_seq_len // block_size,
#                 dtype=torch.long,
#             ).tril()

#             self.attn = BlockSparseAttention(
#                 layout=layout,
#                 block_size=block_size,
#                 d_model=config.n_embd,
#                 n_heads=config.n_head,
#                 is_causal=config.is_causal,
#                 **config.attention_kwargs,
#             )


class GPT2Model(BaseModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config=config)
        if config.vocab_size:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        if self.use_default_positional_encoding:
            self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)

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

    @property
    def use_default_positional_encoding(self) -> bool:
        return not self.config.use_rotary_embeddings

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
        # elif isinstance(module, nn.LayerNorm):
        #     torch.nn.init.zeros_(module.bias)
        #     torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
        if self.config.vocab_size:
            x = self.wte(x)

        if self.use_default_positional_encoding:
            s = x.shape[1]
            pos = torch.arange(0, s, dtype=torch.long, device=x.device)
            pos_emb = self.wpe(pos)
            x = x + pos_emb

        for block in self.h:
            x = block(x, kv_cache=kv_cache)
        return self.ln_f(x)


class GPT2(BaseModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config=config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, self.output_dim, bias=False)

        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    @property
    def output_dim(self):
        if self.config.logit_num is not None:
            return self.config.logit_num
        elif self.config.vocab_size is not None:
            return self.config.vocab_size
        else:
            raise Exception(f"Either `logit_num` or `vocab_size` must be set`")

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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[dict] = None,
        last_logit_only: bool = False,
    ) -> torch.Tensor:
        x = self.transformer(x, kv_cache=kv_cache)
        if last_logit_only:
            x = x[:, -1]
        return self.lm_head(x)
