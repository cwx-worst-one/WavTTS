from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# from byteformers.components.normalization import RMSNorm
from samantha.components.ctiga.ops.rms_norm import RMSNorm
from samantha.utils.ctiga.padding import pad_input, unpad_input

from byteformers.components.activations import Activation, build_activation
from byteformers.components.conv.causal import CausalConv1D
from byteformers.components.feedforward.mlp import MLP
from byteformers.models.base import BaseModel
from byteformers.models.utils import checkpoint


@dataclass
class ConformerConfig:
    n_layer: int
    n_head: int
    n_embd: int
    n_inner: int
    attn_bias: bool = False
    mlp_bias: bool = False
    mlp_dropout: float = 0.1
    attn_dropout: float = 0.1
    conv_dropout: float = 0.1
    rms_norm_epsilon: float = 1e-6
    conv_batch_norm: bool = False
    activation_fn: Activation = Activation.SiLU
    attention_kwargs: Optional[dict] = field(default_factory=dict)
    use_rotary_embeddings: bool = True
    interleave_rotary_embeddings: bool = False  # True = GPT-J, False = GPT-Neo
    conv_expansion_factor: int = 2
    conv_kernel_size: int = 5
    is_causal: bool = False

    # FlashAttention 2.0 parameters:
    version: str = "2.3"
    use_window_mask: bool = False
    window_type: int = 0  # [0: elemwise, 1: blockwise]
    window_size: Tuple[int, int] = (-1, -1)
    blocksparse: bool = False
    gradient_checkpointing: bool = False


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, padding, bias):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, stride=stride, groups=chan_in, bias=bias
        )

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        n_embd: int,
        activation_fn: Activation,
        is_causal: bool,
        expansion_factor: int,
        kernel_size: int,
        dropout: float,
        conv_batch_norm: bool,
        rms_norm_epsilon: float,
    ):
        super().__init__()
        inner_dim = int(n_embd * expansion_factor)

        if is_causal:
            # TODO verify if can be replaced with DepthWiseConv1d
            depthwise_conv = CausalConv1D(
                inner_dim,
                inner_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=None,
                bias=False,
            )
        else:
            padding = (
                calc_same_padding(kernel_size)
                if not is_causal
                else (kernel_size - 1, 0)
            )
            depthwise_conv = DepthWiseConv1d(
                inner_dim,
                inner_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            )

        self.net = nn.Sequential(
            RMSNorm(n_embd, eps=rms_norm_epsilon),
            Rearrange("b t d -> b d t"),
            nn.Conv1d(
                n_embd, inner_dim * 2, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.GLU(dim=1),
            depthwise_conv,
            (
                nn.BatchNorm1d(inner_dim)
                if not is_causal and conv_batch_norm
                else nn.Identity()
            ),
            build_activation(activation_fn),
            nn.Conv1d(
                inner_dim, n_embd, kernel_size=1, stride=1, padding=0, bias=False
            ),
            Rearrange("b d t -> b t d"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, config: ConformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.version = config.version
        self.use_rotary_embeddings = config.use_rotary_embeddings

        self.mlp_1 = MLP(
            config.n_embd,
            n_inner=config.n_inner,
            bias=config.mlp_bias,
            dropout=config.mlp_dropout,
        )

        self.attn_norm_1 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)

        if self.version == 1:
            from byteformers.components.attention import MultiHeadAttention
            self.attn = MultiHeadAttention(
                config.n_embd,
                config.n_head,
                dropout=config.attn_dropout,
                is_causal=config.is_causal,
                use_rotary_embeddings=self.use_rotary_embeddings,
                **config.attention_kwargs,
            )
        else:
            from samantha.components.ctiga.mha import MHA as MultiHeadAttention

            d_k = config.n_embd // config.n_head
            rotary_emb_dim = d_k if self.use_rotary_embeddings else 0
            self.use_window_mask = config.use_window_mask
            self.attn = MultiHeadAttention(
                embed_dim=config.n_embd,
                num_heads=config.n_head,
                cross_attn=False,
                qkv_proj_bias=config.attn_bias,
                out_proj_bias=config.attn_bias,
                dropout=0.0,
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

        self.conv = ConformerConvModule(
            n_embd=config.n_embd,
            activation_fn=config.activation_fn,
            is_causal=config.is_causal,
            expansion_factor=config.conv_expansion_factor,
            kernel_size=config.conv_kernel_size,
            dropout=config.conv_dropout,
            conv_batch_norm=config.conv_batch_norm,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
        self.mlp_2 = MLP(
            config.n_embd,
            n_inner=config.n_inner,
            bias=config.mlp_bias,
            dropout=config.mlp_dropout,
        )

        self.mlp_norm_1 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.mlp_norm_2 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.post_norm = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.mlp_scale = 0.5

    def forward(
        self, x: torch.Tensor, attn_kwargs: Dict[str, Any] = {},
    ) -> torch.Tensor:
        x = self.mlp_scale * self.mlp_1(self.mlp_norm_1(x)) + x

        # attention block
        if self.version == 1:
            x = self.attn(self.attn_norm_1(x), **attn_kwargs) + x
        else:
            x, residual = self.attn(
                self.attn_norm_1(x), **attn_kwargs, use_window_mask=self.use_window_mask
            )
            x = x + residual

        # conv block
        attention_mask = attn_kwargs.get("key_padding_mask")
        if attention_mask is not None:
            x = pad_input(x, attn_kwargs["indices"], attn_kwargs["batch_size"], attn_kwargs["seq_len"])
            x = x * attention_mask.unsqueeze(dim=-1)

        x = self.conv(x) + x
        x = self.mlp_scale * self.mlp_2(self.mlp_norm_2(x)) + x
        x = self.post_norm(x)

        if attention_mask is not None:
            x, _, _, _ = unpad_input(x, attention_mask)
        return x


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
        self.h = nn.ModuleList(
            [ConformerBlock(config, layer_idx) for layer_idx in range(config.n_layer)]
        )

        # Tie all the RotaryEmbedding modules to share the same cos/sin cache
        if config.use_rotary_embeddings:
            for block in self.h[1:]:
                block.attn.rotary_emb = self.h[0].attn.rotary_emb

    def _init_weights(self, module: nn.Module) -> None:
        pass

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        attn_kwargs = {}
        batch_size, seq_len, _ = x.shape

        if attention_mask is not None:            
            x, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                x, attention_mask
            )
            attn_kwargs["batch_size"] = batch_size
            attn_kwargs["seq_len"] = seq_len
            attn_kwargs["cu_seqlens"] = cu_seqlens
            attn_kwargs["max_seqlen"] = max_seqlen_in_batch
            attn_kwargs["indices"] = indices
            attn_kwargs["key_padding_mask"] = attention_mask

        for block in self.h:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, attn_kwargs=attn_kwargs)
            else:
                x = block(x, attn_kwargs=attn_kwargs)

        if attention_mask is not None:
            x = pad_input(x, indices, batch_size, seq_len)
        return x


if __name__ == "__main__":
    config = ConformerConfig(
        n_layer=4,
        n_head=8,
        n_embd=1024,
        n_inner=4096,
        use_rotary_embeddings=True,
        conv_expansion_factor=1,
        is_causal=False,
    )
    model = Conformer(config)
    model = model.to("cuda")

    x = torch.randn((2, 50, 1024), device="cuda")

    attention_mask = torch.zeros(x.shape[0], x.shape[1], device="cuda")
    attention_mask[0, :50] = 1.0
    attention_mask[:, :40] = 1.0

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        out = model.forward(x, attention_mask=attention_mask)
