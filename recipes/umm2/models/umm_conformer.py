import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from typing import Optional, Tuple


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class ConformerRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = 0
        self.cached_rotary_positional_embedding = None
        if config.get("rope_enhance_pos", -1) > 0:
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
    def __init__(self, config):
        super().__init__()
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


class ConformerConvolutionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError(
                "`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding"
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = torch.nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.batch_norm = (
            torch.nn.BatchNorm1d(config.hidden_size)
            if config.get("use_bn", True)
            else nn.Sequential(
                Transpose(), nn.LayerNorm(config.hidden_size), Transpose()
            )
        )
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = torch.nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = torch.nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

    def get_flops(self, n, l):
        def conv1d_flops(m):
            return (
                m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2] / m.groups * 2
            )

        return (
            n
            * l
            * (
                conv1d_flops(self.pointwise_conv1)
                + conv1d_flops(self.depthwise_conv)
                + conv1d_flops(self.pointwise_conv2)
            )
        )


class ConformerSelfAttention(nn.Module):
    def __init__(self, config):
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
        attn_mask: Optional[torch.Tensor] = None,
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

        if attn_mask is not None:
            additive_mask = (~attn_mask).unsqueeze(1).unsqueeze(2)  # [B, nchannel, T, feature_dim]
            additive_mask = additive_mask * -1e9
            attn_mask = additive_mask
        
        with torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_flash=True, enable_mem_efficient=True
        ):

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )

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


class ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.self_attn = ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, 
        hidden_states, 
        attn_mask: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[torch.Tensor] = None
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, attn_mask=attn_mask, position_embeddings=position_embeddings
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

    def get_flops(self, b, t):
        return (
            self.ffn1.get_flops(b, t)
            + self.self_attn.get_flops(b, t)
            + self.conv_module.get_flops(b, t)
            + self.ffn2.get_flops(b, t)
        )
