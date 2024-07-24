


from curses import window
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# from byteformers.components.normalization import RMSNorm
# from flash_attn.modules.mha import MHA as MultiHeadAttention
# from flash_attn.ops.triton.layer_norm import RMSNorm
from samantha.utils.ctiga.padding import pad_input, unpad_input
from samantha.components.ctiga.ops.rms_norm import RMSNorm  # NOTE: is this RMSNorm?
from typing_extensions import Self

from byteformers.components.activations import Activation
from byteformers.components.feedforward import GatedMLP
from byteformers.models.base import BaseModel
from byteformers.models.utils import checkpoint

@dataclass
class LlamaConfig:
    n_layer: int
    n_head: int
    n_embd: int
    is_causal: bool
    n_inner: Optional[int] = None
    vocab_size: Optional[int] = None
    logit_num: Optional[int] = None
    attn_bias: bool = False
    attn_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    rms_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    activation_fn: Activation = Activation.SiLU
    attention_kwargs: Optional[dict] = field(default_factory=dict)
    use_rotary_embeddings: bool = True
    interleave_rotary_embeddings: bool = False  # True = GPT-J, False = GPT-Neo

    # FlashAttention 2.0 parameters:
    version: str = "2.3"
    use_window_mask: bool = False
    window_type: int = 0 # [0: elemwise, 1: blockwise]
    window_size: Tuple[int, int] = (-1, -1)
    blocksparse: bool = False
    gradient_checkpointing: bool = False

    @classmethod
    def from_name(cls, name: str) -> Self:
        return llama_configs[name]


llama_configs = {
    "100M": LlamaConfig(n_layer=12, n_head=12, n_embd=768, is_causal=True),
    "300M": LlamaConfig(n_layer=24, n_head=16, n_embd=1024, is_causal=True),
    "2B": LlamaConfig(n_layer=32, n_head=16, n_embd=2048, is_causal=True),
    "7B": LlamaConfig(n_layer=32, n_head=32, n_embd=4096, is_causal=True),
    "13B": LlamaConfig(n_layer=40, n_head=40, n_embd=5120, is_causal=True),
    "30B": LlamaConfig(n_layer=60, n_head=52, n_embd=6656, is_causal=True),
    "65B": LlamaConfig(n_layer=80, n_head=64, n_embd=8192, is_causal=True),
}


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.version = config.version
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        if self.version == 1:
            from byteformers.components.attention import MultiHeadAttention
            self.attn = MultiHeadAttention(
                d_model=config.n_embd,
                n_heads=config.n_head,
                dropout=config.attn_dropout,
                bias=config.attn_bias,
                is_causal=config.is_causal,
                use_rotary_embeddings=config.use_rotary_embeddings,
                **config.attention_kwargs,
            )
        else:
            # FA2
            from samantha.components.ctiga.mha import MHA as MultiHeadAttention
            d_k = config.n_embd // config.n_head
            rotary_emb_dim = d_k if config.use_rotary_embeddings else 0
            self.use_window_mask = config.use_window_mask
            self.attn = MultiHeadAttention(
                embed_dim=config.n_embd,
                num_heads=config.n_head,
                cross_attn=False,
                qkv_proj_bias=config.attn_bias,
                out_proj_bias=config.attn_bias,
                dropout=config.attn_dropout,
                causal=config.is_causal,
                layer_idx=layer_idx,
                fused_bias_fc=True,
                return_residual=True,
                use_flash_attn=True,
                version=config.version,
                window_type=config.window_type,
                window_size=config.window_size,
                blocksparse=config.blocksparse,
                rotary_emb_dim=rotary_emb_dim,
                rotary_emb_interleaved=config.interleave_rotary_embeddings,
                rotary_emb_compat="default",
                use_rotary_triton=False,
            )
        
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        if config.n_inner is None:
            config.n_inner = self.get_n_inner_dim(config.n_embd)

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

    def forward(
        self, x: torch.Tensor, attn_kwargs: Dict[str, Any] = {}, return_attn_probs: bool = False,
    ) -> torch.Tensor:

        if self.version == 1:
            residual = x
            if return_attn_probs:
                x, attn = self.attn(self.ln_1(x), **attn_kwargs, return_attn_probs=return_attn_probs)
            else:
                x = self.attn(self.ln_1(x), **attn_kwargs, return_attn_probs=return_attn_probs)
        else:
            if return_attn_probs:
                x, residual, attn = self.attn(self.ln_1(x), **attn_kwargs, use_window_mask=self.use_window_mask, return_attn_probs=True)
            else:
                x, residual = self.attn(self.ln_1(x), **attn_kwargs, use_window_mask=self.use_window_mask)

        x = x + residual
        x = x + self.mlp(self.ln_2(x))

        if return_attn_probs:
            return x, attn
        else:
            return x


class LlamaContextBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        # TODO: check if it has monotonic / clear attention map!
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        d_k = config.n_embd // config.n_head
        rotary_emb_dim = d_k if config.use_rotary_embeddings else 0
        self.use_window_mask = config.use_window_mask

        from samantha.components.ctiga.mha import MHA as MultiHeadAttention
        self.attn = MultiHeadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            cross_attn=False,
            qkv_proj_bias=config.attn_bias,
            out_proj_bias=config.attn_bias,
            dropout=0.0,
            causal=config.is_causal,
            layer_idx=layer_idx,
            rotary_emb_dim=rotary_emb_dim,
            fused_bias_fc=True,
            return_residual=True,
            use_flash_attn=True,
            version=config.version,
            window_type=config.window_type,
            window_size=config.window_size,
            blocksparse=config.blocksparse,
        )
        self.ln_c = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.cross_attn = MultiHeadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            cross_attn=True,
            qkv_proj_bias=config.attn_bias,
            out_proj_bias=config.attn_bias,
            dropout=0.0,
            causal=config.is_causal,
            layer_idx=layer_idx,
            rotary_emb_dim=0, # NOTE: only used in self-attn
            fused_bias_fc=True,
            return_residual=True,
            use_flash_attn=True,
            version=config.version,
            window_type=config.window_type,
            window_size=config.window_size,
            blocksparse=config.blocksparse,
        )
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        config.n_inner = self.get_n_inner_dim(config.n_embd)
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

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_kwargs: Dict[str, Any] = {}, context_attn_kwargs: Dict[str, Any] = {}) -> torch.Tensor:
        x, residual = self.attn(self.ln_1(x), **attn_kwargs, use_window_mask=self.use_window_mask)
        x = x + residual

        x, residual = self.cross_attn(self.ln_c(x), context, **context_attn_kwargs, use_window_mask=self.use_window_mask)
        x = x + residual

        x = x + self.mlp(self.ln_2(x))
        return x


class LlamaModel(BaseModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config=config)

        if config.vocab_size:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.h = nn.ModuleList([LlamaBlock(config, layer_idx=layer_idx) for layer_idx in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        # Tie all the RotaryEmbedding modules to share the same cos/sin cache
        if config.use_rotary_embeddings:
            for block in self.h[1:]:
                block.attn.rotary_emb = self.h[0].attn.rotary_emb

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

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if hasattr(self, "wte"):
            x = self.wte(x)
        
        attn_kwargs = {}
        batch_size, x_seq_len, _ = x.shape

        if attention_mask is not None:            
            x, x_indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                x, attention_mask
            )
            attn_kwargs["cu_seqlens"] = cu_seqlens
            attn_kwargs["max_seqlen"] = max_seqlen_in_batch

            if self.config.use_rotary_embeddings:
                attn_kwargs["indices"] = x_indices
                attn_kwargs["key_padding_mask"] = attention_mask

        for block in self.h:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, attn_kwargs=attn_kwargs)
            else:
                x = block(x, attn_kwargs=attn_kwargs)

        if attention_mask is not None:
            x = pad_input(x, x_indices, batch_size, x_seq_len)
        return self.ln_f(x)


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.logit_num = config.logit_num
        self.vocab_size = config.vocab_size
        self.transformer = LlamaModel(config)
        self.lm_head = nn.Linear(config.n_embd, self.output_dim, bias=False)

    @property
    def output_dim(self):
        if self.logit_num is not None:
            return self.logit_num
        elif self.vocab_size is not None:
            return self.vocab_size
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        last_logit_only: bool = False,
    ):
        x = self.transformer(input_ids, attention_mask=attention_mask)
        if last_logit_only:
            x = x[:, -1]
        return self.lm_head(x)

class LlamaContextModel(BaseModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config=config)
        self.h: nn.ModuleList[LlamaContextBlock] = nn.ModuleList([LlamaContextBlock(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        # Tie all the RotaryEmbedding modules to share the same cos/sin cache
        if config.use_rotary_embeddings:
            for block in self.h[1:]:
                block.attn.rotary_emb = self.h[0].attn.rotary_emb

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

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, context_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        attn_kwargs = {}
        context_attn_kwargs = {}
        batch_size, x_seq_len, _ = x.shape
        if attn_mask is None:
            attn_mask = torch.ones((batch_size, x_seq_len), device=x.device)
        
        x, x_indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
            x, attn_mask
        )
        attn_kwargs["cu_seqlens"] = cu_seqlens
        attn_kwargs["max_seqlen"] = max_seqlen_in_batch

        if self.config.use_rotary_embeddings:
            attn_kwargs["indices"] = x_indices
            attn_kwargs["key_padding_mask"] = attn_mask


        if context_attn_mask is not None:
            context_attn_kwargs["cu_seqlens"] = cu_seqlens
            context_attn_kwargs["max_seqlen"] = max_seqlen_in_batch
            context, _, cu_seqlens_k, max_seqlen_k_in_batch = unpad_input(
                context, context_attn_mask
            )
            context_attn_kwargs["cu_seqlens_k"] = cu_seqlens_k
            context_attn_kwargs["max_seqlen_k"] = max_seqlen_k_in_batch
            # NOTE: rotary embeddings is not used for cross-attn

        for block in self.h:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, context, attn_kwargs=attn_kwargs, context_attn_kwargs=context_attn_kwargs)
            else:
                x = block(x, context, attn_kwargs=attn_kwargs, context_attn_kwargs=context_attn_kwargs)

        x = pad_input(x, x_indices, batch_size, x_seq_len)
        return self.ln_f(x)


class LlamaEncoderModel(BaseModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config=config)
        self.h: nn.ModuleList[LlamaBlock] = nn.ModuleList([LlamaBlock(config, layer_idx) for layer_idx in range(config.n_layer)])

        # Tie all the RotaryEmbedding modules to share the same cos/sin cache
        if config.use_rotary_embeddings:
            for block in self.h[1:]:
                block.attn.rotary_emb = self.h[0].attn.rotary_emb

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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        attn_kwargs = {}
        batch_size, x_seq_len, _ = x.shape

        if attn_mask is not None:
            x, x_indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                x, attn_mask
            )

            attn_kwargs["cu_seqlens"] = cu_seqlens
            attn_kwargs["max_seqlen"] = max_seqlen_in_batch
            if self.config.use_rotary_embeddings:
                attn_kwargs["indices"] = x_indices
                attn_kwargs["key_padding_mask"] = attn_mask

        for block in self.h:
            x = block(x, attn_kwargs=attn_kwargs)

        
        if attn_mask is not None:
            x = pad_input(x, x_indices, batch_size, x_seq_len)
        return x
