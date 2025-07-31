from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from recipes.sacodec.models.components.convnext import ConvNeXtBlock, get_padding


class ConformerConfig(PretrainedConfig):
    model_type = "Conformer"

    def __init__(
        self,
        # normalize inputs
        input_channels=128, # n_mels
        
        # shared encoder
        hidden_size=1536,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rotary_embedding_base=10000,

        # ROPE
        rope_enhance_pos=7500, # random at training, fixed at validation
        # rope_enhance_pos = 0, # fixed rope at training and validation
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        # shared encoder
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.rotary_embedding_base = rotary_embedding_base
        # ROPE
        self.rope_enhance_pos = rope_enhance_pos

    def get(self, name, default):
        if hasattr(self, name):
            return self.__getattribute__(name)
        else:
            return default


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class ConformerRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = 0
        self.cached_rotary_positional_embedding = None
        if config.rope_enhance_pos > 0:
            self._set_cos_sin_cache(config.rope_enhance_pos)

    def _set_cos_sin_cache(self, sequence_length):
        with torch.cuda.amp.autocast(enabled=False):
            self.cached_sequence_length = sequence_length
            time_stamps = torch.arange(
                sequence_length, device=self.inv_freq.device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
            embeddings = torch.cat((freqs, freqs), dim=-1)
            cos_embeddings = embeddings.cos()[:, None, None, :]
            sin_embeddings = embeddings.sin()[:, None, None, :]
            self.cached_rotary_positional_embedding = torch.stack(
                [cos_embeddings, sin_embeddings]
            )

    def forward(self, hidden_states):
        batch, sequence_length, _ = hidden_states.shape
        if (
            sequence_length > self.cached_sequence_length
            or self.cached_rotary_positional_embedding is None
        ):
            self._set_cos_sin_cache(sequence_length)
        if self.cached_rotary_positional_embedding.device != hidden_states.device:
            self.cached_rotary_positional_embedding = (
                self.cached_rotary_positional_embedding.to(hidden_states.device)
            )
        # position augmentation
        if self.config.get("rope_enhance_pos", -1) > 0:
            if not self.training:
                return self.cached_rotary_positional_embedding[:, 0:sequence_length]
            beg_idx = torch.randint(
                size=[batch],
                low=0,
                high=self.cached_sequence_length - sequence_length + 1,
            )
            return torch.cat(
                [
                    self.cached_rotary_positional_embedding[
                        :, idx : idx + sequence_length
                    ]
                    for idx in beg_idx
                ],
                dim=2,
            )
        elif self.config.get("rope_enhance_pos", -1) == 0:
            return self.cached_rotary_positional_embedding[:, 0:sequence_length]
        else:
            return self.cached_rotary_positional_embedding[:, -sequence_length:]


class ConformerFeedForward(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

    def get_flops(self, b, t):
        def linear_flops(m):
            return m.weight.shape[0] * m.weight.shape[1] * 2

        return (
            b
            * t
            * (linear_flops(self.intermediate_dense) + linear_flops(self.output_dense))
        )


class ConformerSelfAttention(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()

        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if position_embeddings is not None:
            query_key_states = self._apply_rotary_embedding(
                query_key_states, position_embeddings
            )

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(
            batch_size, -1, self.num_heads, self.head_size
        )
        key = self.linear_k(query_key_states).view(
            batch_size, -1, self.num_heads, self.head_size
        )
        value = self.linear_v(value_states).view(
            batch_size, -1, self.num_heads, self.head_size
        )

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_flash=True, enable_mem_efficient=True
        ):
            hidden_states = F.scaled_dot_product_attention(
                query.float(),
                key.float(),
                value.float(),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        # => (batch, time1, hidden_size)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_size
        )
        hidden_states = self.linear_out(hidden_states)

        return hidden_states

    @torch.cuda.amp.autocast(enabled=False)
    def _apply_rotary_embedding(self, hidden_states, position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        hidden_states = hidden_states.view(
            batch_size, sequence_length, self.num_heads, self.head_size
        )

        cos = position_embeddings[0, :sequence_length, ...]
        sin = position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat(
            (-rotated_states_end, rotated_states_begin),
            dim=rotated_states_begin.ndim - 1,
        )
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = hidden_states.view(
            batch_size, sequence_length, self.num_heads * self.head_size
        )

        return hidden_states

    def get_flops(self, b, t):
        def linear_flops(m):
            return m.weight.shape[0] * m.weight.shape[1] * 2

        return (
            b
            * t
            * (
                linear_flops(self.linear_q)
                + linear_flops(self.linear_k)
                + linear_flops(self.linear_v)
                + linear_flops(self.linear_out)
            )
            + b * t * t * self.head_size * 2
            + b * t * self.head_size * t * 2
        )


class ConformerNextEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config: ConformerConfig, add_final_layer=True):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout
        intermediate_size = config.intermediate_size

        # Feed-forward 1
        self.ffn1 = ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.self_attn = ConformerSelfAttention(config)

        # ConvNeXt Convolution
        layer_scale_init_value = 1 / config.num_hidden_layers
        self.conv_module = ConvNeXtBlock(
            dim=embed_dim,
            intermediate_dim=intermediate_size,
            layer_scale_init_value=layer_scale_init_value,
            add_prenorm=True
        )

        # Feed-forward 2
        self.ffn2 = ConformerFeedForward(config)
        self.add_final_layer = add_final_layer
        if self.add_final_layer:
            self.final_layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.final_layer_norm = nn.Identity()

    def forward(
        self, hidden_states, position_embeddings: Optional[torch.Tensor] = None
    ):
        # hidden_states = b x l x c

        # 1. Feed-Forward 1 layer
        hidden_states = self.ffn1(hidden_states) * 0.5 + hidden_states

        # 2. Self-Attention layer
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, position_embeddings=position_embeddings
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        # residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2) # b x l x c -> b x c x l
        hidden_states = self.conv_module(hidden_states) # convnext needs transpose.
        hidden_states = hidden_states.transpose(1, 2) # b x l x c -> b x c x l
        # hidden_states = residual + hidden_states # ConvNextBlock already adds residual

        # 4. Feed-Forward 2 Layer
        hidden_states = self.ffn2(hidden_states) * 0.5 + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class ConformerNextBlock(nn.Module):
    """
    Combination of conformer and ConvNext
    """
    def __init__( # VOCOS
        self,
        config: ConformerConfig,
        add_final_layer=False
    ):
        super().__init__()
        self.config = config
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.layers = nn.ModuleList(
            [ConformerNextEncoderLayer(config, add_final_layer=add_final_layer) for _ in range(config.num_hidden_layers)]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, hidden_states):
        # inpput = B, L, D
        position_embeddings = self.embed_positions(hidden_states)
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states, position_embeddings=position_embeddings
            )
            hidden_states = layer_outputs
        return hidden_states

class ConformerNextBackboneDownUp(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        ratio=1,
        layer_scale_init_value: Optional[float] = None,
        apply_final_layer_norm=True,
        pre_embed_padding=None,
        hidden_dropout=0.1
    ):
        super().__init__()
        self.input_channels = input_channels

        if pre_embed_padding is None:
            pre_embed_padding = get_padding(7, 1)
        if ratio >= 1:
            self.pre_embed = weight_norm(nn.Conv1d(self.input_channels, dim, stride=ratio, kernel_size=7, padding=pre_embed_padding))
        else:
            self.pre_embed = weight_norm(nn.ConvTranspose1d(self.input_channels, dim, 7, round(1/ratio),
                                                            padding=pre_embed_padding))

        self.pre_norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.dropout = nn.Dropout(hidden_dropout)

        config = ConformerConfig(
            input_channels=input_channels,
            hidden_size=dim,
            intermediate_size=intermediate_dim,
            num_hidden_layers=num_layers
        )
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.conformer = nn.ModuleList(
            [
                ConformerNextEncoderLayer(
                    config=config, add_final_layer=False
                )
                for _ in range(num_layers)
            ]
        )
        if apply_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        else:
            self.final_layer_norm = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, C, L), where B is the batch size, L is the sequence length,
                    and C denotes the model dimension.
        """
        # x: bs x ch x l # same as ConvNext
        # out: bs x ch x l
        x = self.pre_embed(x)
        x = x.transpose(1, 2) # bs x l x ch
        x = self.pre_norm(x) 
        x = self.dropout(x)
        position_embeddings = self.embed_positions(x) # conformer
        for conformer_block in self.conformer:
            x = conformer_block(x, position_embeddings=position_embeddings)
        x = self.final_layer_norm(x)
        return x.transpose(1, 2) # bs x l x ch -> bs x ch x l
