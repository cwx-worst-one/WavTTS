import math
from dataclasses import dataclass
from functools import partial, reduce
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import pack, rearrange, reduce, unpack
from torch import Tensor, int32, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from recipes.umm.models.dualumm_encoders import ConvStacksWithDownUpSampling
#from recipes.umm.models.dualumm_vector_quantizers import get_vq_codebook_distances
from recipes.umm.models.rmvpe import RMVPE
from recipes.umm.models.voc_modules.pitch_predictor.inference import (
    PerceptualPitchPredictor,
)
from recipes.umm.models.voc_modules.pitch_predictor.model import ConvBlocks
from recipes.umm.models.voc_modules.pitch_predictor.pitch_utils import (
    compute_min_lengths,
    raw_hz_to_log1p,
)
from recipes.umm.models.vq import EMAVectorQuantizer
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.transforms.speech import SpeechTransform, SpeechTransformModified
from recipes.umm.utils.mel_utils import torch_wav2spec


@dataclass
class UMMResult:
    hidden_states: torch.Tensor
    vq_ids: torch.Tensor
    vq_hidden_states: torch.Tensor
    vq_loss: Optional[torch.Tensor] = None


def f0_normalize(f0):
    _f0 = f0.clone()
    f0 = torch.log1p(f0)
    f0_mean = torch.mean(f0[_f0 != 0])
    f0_std = torch.std(f0[_f0 != 0])
    f0_std = torch.where(f0_std == 0, torch.ones_like(f0_std), f0_std)
    f0[_f0 != 0] = (f0[_f0 != 0] - f0_mean) / f0_std
    return f0


def get_vuv(f0):
    vuv = f0.clone()
    vuv[vuv != 0] = 1
    return vuv


def conv_flops(module, input_shape):
    output_shape = input_shape
    output_shape[1] = module.out_channels
    dims = len(input_shape) - 2
    flops = input_shape[0] * module.in_channels * module.out_channels
    for i in range(dims):
        new_kernel_size = module.dilation[i] * (module.kernel_size[i] - 1) + 1
        flops *= new_kernel_size
        output_shape[2 + i] = (
            input_shape[2 + i] + 2 * module.padding[i] - new_kernel_size
        ) // module.stride[i] + 1
        flops *= output_shape[2 + i]
    return 2 * flops, output_shape


def conv_transpose_flops(module, input_shape):
    output_shape = input_shape
    output_shape[1] = module.out_channels
    dims = len(input_shape) - 2
    flops = input_shape[0] * module.in_channels * module.out_channels
    for i in range(dims):
        new_kernel_size = module.dilation[i] * (module.kernel_size[i] - 1) + 1
        flops *= new_kernel_size
        output_shape[2 + i] = (
            (input_shape[2 + i] - 1) * module.stride[i]
            + module.output_padding[i]
            - 2 * module.padding[i]
            + new_kernel_size
        )
        flops *= input_shape[2 + i]
    return 2 * flops, output_shape


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

@dataclass
class ConformerEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


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
                query.float(),
                key.float(),
                value.float(),
                attn_mask=attn_mask,
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
        self, hidden_states, attn_mask: Optional[torch.Tensor] = None, position_embeddings: Optional[torch.Tensor] = None
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


class ConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_positions = ConformerRotaryPositionalEmbedding(config)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attn_mask=None, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        hidden_states = self.dropout(hidden_states)
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # create gradient checkpointing function
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), hidden_states, attn_mask, position_embeddings
                )
            else:
                layer_outputs = layer(
                    hidden_states, attn_mask=attn_mask, position_embeddings=position_embeddings
                )
            hidden_states = layer_outputs

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return ConformerEncoderOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


class Conv1dUpsampling(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True, act_fn=nn.ReLU):
        super().__init__()
        hidden_dim = input_dim * 4
        self.conv = nn.Sequential(
            Transpose(),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            act_fn(),
            nn.ConvTranspose1d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            act_fn(),
            nn.ConvTranspose1d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            act_fn(),
            Transpose(),
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

    def get_flops(self, b, t, d):
        return 0


class Conv2dUpsampling(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True, act_fn=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            # [1, 1, 750, 1024]
            nn.Conv2d(1, 64, 7, 1, 3),
            torch.nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            act_fn(),
            # [1, 64, 750, 1024]
            nn.ConvTranspose2d(64, 8, 6, 2, 2),
            torch.nn.BatchNorm2d(8) if use_bn else nn.Identity(),
            act_fn(),
            # [1, 8, 1500, 2048]
            nn.ConvTranspose2d(8, 1, 6, 2, 2),
            torch.nn.BatchNorm2d(1) if use_bn else nn.Identity(),
            act_fn(),
            # [1, 1, 3000, 4096]
        )
        self.linear = nn.Linear(input_dim * 4, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x

    def get_flops(self, b, t, d):
        flops1, out_shape1 = conv_flops(self.conv[0], [b, 1, t, d])
        flops2, out_shape2 = conv_transpose_flops(self.conv[3], out_shape1)
        flops3, _ = conv_transpose_flops(self.conv[6], out_shape2)
        return flops1 + flops2 + flops3


class Conv2dSubsampling(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel, padding, use_bn=True, act_fn=nn.ReLU
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel, 2, padding),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
            nn.Conv2d(256, 256, kernel, 2, padding),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
        )
        self.linear = nn.Linear(input_dim * 64, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x

    def get_flops(self, b, t, d):
        flops1, out_shape1 = conv_flops(self.conv[0], [b, 1, t, d])
        flops2, _ = conv_flops(self.conv[3], out_shape1)
        return flops1 + flops2


class Conv2dSubsamplingModified(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel, padding, use_bn=True, act_fn=nn.ReLU
    ):
        super().__init__()
        self.stride = [2, 2]
        self.kernel_size = kernel
        self.padding_size = padding
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel, self.stride[0], padding),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
            nn.Conv2d(256, 256, kernel, self.stride[1], padding),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
        )
        self.linear = nn.Linear(input_dim * 64, output_dim)

    def _compute_output_length(self, x_length):
        for s in self.stride:
            x_length = (x_length + 2 * self.padding_size - self.kernel_size) // s + 1
        return x_length
    
    def forward(self, x, x_lengths=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
  
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        if x_lengths is None:
            return x
        output_lengths = self._compute_output_length(x_lengths)
        return x, output_lengths
    
    def get_flops(self, b, t, d):
        flops1, out_shape1 = conv_flops(self.conv[0], [b, 1, t, d])
        flops2, _ = conv_flops(self.conv[3], out_shape1)
        return flops1 + flops2


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.get("cal_attention_mask", False):
            self.feature_encoder = Conv2dSubsamplingModified(
                config.num_channels,
                config.hidden_size,
                config.feature_encoder_kernel,
                config.feature_encoder_padding,
                use_bn=config.get("use_bn", True),
            )
        else:
            self.feature_encoder = Conv2dSubsampling(
                config.num_channels,
                config.hidden_size,
                config.feature_encoder_kernel,
                config.feature_encoder_padding,
                use_bn=config.get("use_bn", True),
            )
        self.conformer_layer = (
            ConformerEncoderLayer(config)
            if config.get("first_conformer", True)
            else nn.Identity()
        )

    def _calculate_masking(self, x, x_length):
        #https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # A boolean mask where a value of True indicates that the element should take part in attention
        # B = x_length.size(0)
        # max_len = torch.max(x_length).item()
        max_len = x.shape[-2]
        indices = torch.arange(max_len, device=x_length.device)
        mask = indices.unsqueeze(0) < x_length.view(-1).unsqueeze(1)
        return mask
    
    def forward(self, x, x_length=None):
        if x_length is None:
            x = self.feature_encoder(x)
            x = self.conformer_layer(x)
            return x
        else:
            x, x_length = self.feature_encoder(x, x_length)
            attn_mask = self._calculate_masking(x, x_length)
            x = self.conformer_layer(x, attn_mask=attn_mask)
            return x, attn_mask

    def get_flops(self, b, t, d):
        if self.config.get("first_conformer", True):
            return self.feature_encoder.get_flops(
                b, t, d
            ) + self.conformer_layer.get_flops(b, t)
        else:
            return self.feature_encoder.get_flops(b, t, d)


class EMAEmbedding(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim,
        decay=0.99,
        eps=1e-5,
        learnable=False,
        orthonormal_init=False,
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.learnable = learnable

        weight = torch.randn(codebook_size, codebook_dim, dtype=torch.float32)
        if orthonormal_init:
            weight = torch.qr(weight)[0]
            std = weight.std(dim=1).unsqueeze(1)
            weight = weight / std
        weight[0] = 0.0
        if not learnable:
            self.register_buffer("weight", weight)
        else:
            self.register_parameter("weight", nn.Parameter(weight))
        self.register_buffer("cluster_size", torch.zeros(codebook_size) + 8)
        self.register_buffer("embed_avg", weight.clone())
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size.data, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(
            new_embed_avg.data, alpha=1 - self.decay
        )

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized.data)
        if not self.learnable:
            # make sure the greedy algorithm to get best estimation
            self.weight.data[0] = 0.0
            self.embed_avg[0] = 0.0

    @torch.no_grad()
    def entropy(self):
        p = self.cluster_size / self.cluster_size.sum()
        p = p.clamp(1e-9)
        entropy = (-p * p.log()).sum()
        return entropy

    @torch.no_grad()
    def remap_weight(self, thres=2):
        cs = self.cluster_size.data
        avg_w = self.embed_avg.data
        w = self.weight.data
        # ranking
        topk_res = torch.topk(cs[1:], k=cs.shape[-1] - 1)  # 大到小
        indices = topk_res.indices + 1
        # remap
        new_cs = torch.zeros_like(cs)
        new_avg_w = torch.zeros_like(avg_w)
        new_w = torch.zeros_like(w)
        cnt = 0
        for j, ind in enumerate(indices):
            cs_tmp = cs[ind]
            if cs_tmp < thres:
                print(cs_tmp, j)
                new_w[j + 1] = w[indices[cnt]]
                new_avg_w[j + 1] = w[indices[cnt]] * cs[indices[cnt]] / 2
                new_cs[j + 1] = cs[indices[cnt]] / 2
                cnt += 1
            else:
                new_w[j + 1] = w[ind]
                new_avg_w[j + 1] = avg_w[ind]
                new_cs[j + 1] = cs[ind]
        if i == 0:
            new_cs[0] = 8
        else:
            new_cs[0] = cs[0]
        self.cluster_size.data.copy_(new_cs.data)
        self.embed_avg.data.copy_(new_avg_w.data)
        self.weight.data.copy_(new_w.data)


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self, codebook_size, codebook_dim, same_index_shape=True, decay=0.99, dist=True
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay

        # ema for dist
        self.dist = dist
        self.embedding = EMAEmbedding(
            self.codebook_size, self.codebook_dim, decay=decay, learnable=False
        )
        self.same_index_shape = same_index_shape

    def forward(self, z):
        z_flattened = rearrange(z, "b t d -> (b t) d")

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # [b*h]
        z_q = self.embedding(min_encoding_indices).view(
            z.shape
        )  # [b*h, c] -> [b, h, c]

        # EMA updating, use for
        if self.training and self.embedding.update:
            one_hot = F.one_hot(min_encoding_indices, self.codebook_size).type(
                z.dtype
            )  # [b*h, k]
            # EMA cluster size
            one_hot_sum = one_hot.sum(0)  # [k]
            if self.dist:
                torch.distributed.all_reduce(one_hot_sum)
            self.embedding.cluster_size_ema_update(one_hot_sum)
            # EMA embedding average
            embed_sum = (
                one_hot.transpose(0, 1) @ z_flattened
            )  # [k, b*h] * [b*h, c] = [k, c]
            if self.dist:
                torch.distributed.all_reduce(embed_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.codebook_size)

        loss = torch.mean((z_q.detach() - z) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.same_index_shape:
            min_encoding_indices = rearrange(
                min_encoding_indices, "(b t) -> b t", t=z.size(1)
            )

        return z_q, min_encoding_indices, loss

    @torch.no_grad()
    def entropy(self):
        return self.embedding.entropy()


class EMAVectorQuantizerEntropy(nn.Module):
    def __init__(
        self, codebook_size, codebook_dim, same_index_shape=True, decay=0.99, dist=True
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.dist = dist
        self.same_index_shape = same_index_shape
        self.embedding = EMAEmbedding(
            self.codebook_size, self.codebook_dim, decay=decay, learnable=True
        )

    def forward(self, z, e_scale=1.0):
        z_flattened = rearrange(z, "b t d -> (b t) d")
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )
        min_encoding_indices = torch.argmin(d, dim=1)  # [b*h]
        z_q = self.embedding(min_encoding_indices).view(
            z.shape
        )  # [b*h, c] -> [b, h, c]
        # EMA update
        if self.training and self.embedding.update:
            one_hot = F.one_hot(min_encoding_indices, self.codebook_size).type(
                z.dtype
            )  # [b*h, k]
            # EMA cluster size
            one_hot_sum = one_hot.sum(0)  # [k]
            if self.dist:
                torch.distributed.all_reduce(one_hot_sum)
            self.embedding.cluster_size_ema_update(one_hot_sum)
            # EMA embedding average
            embed_sum = (
                one_hot.transpose(0, 1) @ z_flattened
            )  # [k, b*h] * [b*h, c] = [k, c]
            if self.dist:
                torch.distributed.all_reduce(embed_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.codebook_size)
            
            loss = torch.mean((z_q.detach() - z) ** 2) + e_scale * self.entropy_loss(
                -d, loss_type="softmax"
            )
        else:
            loss = torch.zeros(1).to(z.device)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.same_index_shape:
            min_encoding_indices = rearrange(
                min_encoding_indices, "(b t) -> b t", t=z.size(1)
            )

        return z_q, min_encoding_indices, loss

    def entropy_loss(self, affinity, loss_type="softmax", temperature=0.7, eps=1e-10):
        # affinity: [b, t, d_n]
        flat_affinity = affinity.reshape(
            -1, affinity.shape[-1]
        )  # [b, t, d_n] -> [b*t, d_n]
        flat_affinity = flat_affinity / temperature
        probs = flat_affinity.softmax(dim=-1)  # [b*t, d_n]
        log_probs = (probs + eps).log()  # [b*t, d_n]
        if loss_type == "softmax":
            target_probs = probs
        elif loss_type == "argmax":
            codes = flat_affinity.argmax(dim=-1)  # [b*t]
            onehots = F.one_hot(codes, flat_affinity.shape[-1]).float()  # [b*t, d_n]
            onehots = probs - (probs - onehots).detach()  # [b*t, d_n]
            target_probs = onehots  # [b*t, d_n]
        else:
            raise ValueError("Entropy loss {} not supported".format(loss_type))
        avg_probs = torch.mean(target_probs, dim=0)  # [b*t, d_n] -> [d_n]
        avg_entropy = -torch.sum(avg_probs * (avg_probs + eps).log())  # [d_n] -> []
        sample_entropy = -torch.mean(
            torch.sum(target_probs * log_probs, dim=-1)
        )  # [b*t, d_n] * [b*t, d_n] -> [b*t] -> []
        loss = 0.1 * (sample_entropy - 2.5 * avg_entropy)

        return loss

    def entropy(self):
        return self.embedding.entropy()


class ClusteredVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim,
        beta=0.25,
        distance="cos",
        anchor="probrandom",
        first_batch=False,
        contras_loss=True,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.codebook_size, self.codebook_dim)
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )
        self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    def forward(self, z):
        z_flattened = rearrange(z, "b t d -> (b t) d")

        # clculate the distance
        if self.distance == "l2":
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = (
                -torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True)
                - torch.sum(self.embedding.weight**2, dim=1)
                + 2
                * torch.einsum(
                    "bd, dn-> bn",
                    z_flattened.detach(),
                    rearrange(self.embedding.weight, "n d-> d n"),
                )
            )
        elif self.distance == "cos":
            # cosine distances from z to embeddings e_j
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum(
                "bd,dn->bn",
                normed_z_flattened,
                rearrange(normed_codebook, "n d -> d n"),
            )

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        encodings = torch.zeros(
            encoding_indices.unsqueeze(1).shape[0], self.codebook_size, device=z.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # count
        avg_probs = torch.mean(encodings, dim=0)

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            # running average updates
            if self.anchor in ["closest", "random", "probrandom"] and (not self.init):
                # closest sampling
                if self.anchor == "closest":
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]
                # feature pool based random sampling
                elif self.anchor == "random":
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == "probrandom":
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = (
                    torch.exp(
                        -(self.embed_prob * self.codebook_size * 10) / (1 - self.decay)
                        - 1e-3
                    )
                    .unsqueeze(1)
                    .repeat(1, self.codebook_dim)
                )
                self.embedding.weight.data = (
                    self.embedding.weight.data * (1 - decay) + random_feat * decay
                )
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[
                    -max(1, int(sort_distance.size(0) / self.codebook_size)) :, :
                ].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[: int(sort_distance.size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(
                    dis,
                    torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device),
                )
                loss += contra_loss

        encoding_indices = rearrange(encoding_indices, "(b t) -> b t", t=z.size(1))

        return z_q, encoding_indices, loss

    @torch.no_grad()
    def entropy(self):
        p = self.embed_prob / self.embed_prob.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        return entropy

def get_vq_codebook_distances(codebook_data):
    """
    Calculate pairwise codebook distance statistics for monitoring on wandb.
    23APR2024 @hanoihantrakul copy pasted from dualumm_vector_quantizers.get_vq_codebook_distances
    otherwise I get a cyclic import.
    """

    embeddings = codebook_data
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    min_distance = torch.min(
        pairwise_distances
        + torch.eye(pairwise_distances.shape[0], device=pairwise_distances.device)
        * pairwise_distances.max()
    )
    return {
        "vq_mean_distance": pairwise_distances.mean(),
        "vq_min_distance": min_distance,
        "vq_max_distance": pairwise_distances.max(),
    }

class FeaturePool:
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """

    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if (
                features.size(0) > self.pool_size
            ):  # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(
                    0, features.size(0), (int(self.pool_size),)
                )
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features : num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(
                    0, features.size(0), (int(self.pool_size),)
                )
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[: features.size(0)]] = features

        return self.features


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, codebook_size):
        super().__init__()
        _recommended_levels = {
            256: [8, 6, 5],
            1024: [8, 5, 5, 5],
            4096: [7, 5, 5, 5, 5],
            16384: [8, 8, 8, 6, 5],
            32768: [8, 8, 8, 8, 8],
            65536: [8, 8, 8, 5, 5, 8],
        }
        if codebook_size not in _recommended_levels:
            raise KeyError(
                f"{codebook_size} is not in one of the recommended FiniteScalarQuantizer levels"
            )
        levels = _recommended_levels[codebook_size]
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis)

        self.dim = len(levels)
        self.n_codes = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(self.n_codes))
        self.register_buffer("implicit_codebook", implicit_codebook)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        zhat = self.quantize(z)
        indices = self.codes_to_indices(zhat)
        return zhat, indices

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return self._scale_and_shift_inverse(codes_non_centered)


class RandomProjectionQuantizer(nn.Module):
    """RandomProjectionQuantizer"""

    def __init__(self, config):
        """A quantizer based on random projection
        See: https://arxiv.org/pdf/2202.01855.pdf
        Args:
            dim: input dimension (channels)
            codebook_size: the number of code in the codebook
            codebook_dim: the dimension of the the code
            codebook_num: the number of quantizers.
                    See multi-softmax in https://arxiv.org/abs/2303.01037
            initialization_type: the initialization method of the projection matrix
        """
        super().__init__()
        self.input_dim = config.rq_input_dim
        self.codebook_size = config.rq_codebook_size
        self.codebook_dim = config.rq_codebook_dim
        self.codebook_num = config.rq_codebook_num

        self.register_buffer(
            "prototypes",
            torch.zeros(
                self.codebook_num,
                1,
                self.codebook_size,
                self.codebook_dim,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "proj",
            torch.zeros(
                self.input_dim,
                self.codebook_num * self.codebook_dim,
                requires_grad=False,
            ),
        )
        self._initialize()

    def _initialize(self):
        """Initialize the parameters."""
        nn.init.normal_(self.prototypes)
        F.normalize(self.prototypes, dim=-1, out=self.prototypes)

        fan_in = self.input_dim
        fan_out = self.codebook_dim
        gain = 1.0
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        with torch.no_grad():
            self.proj.normal_(0, std)

    def forward(self, x):
        """
        Forward a batch of representations to get discrete codes.
        Args:
            x: [batch_size, dim]
        Returns:
            codes: [batch_size, codebook_num], torch.int64
        """
        assert len(x.shape) == 2
        batch_size, _ = x.shape

        projected = torch.matmul(
            x, self.proj
        )  # [batch_size, codebook_num*codebook_dim]
        projected = F.normalize(
            projected.view(batch_size, self.codebook_num, self.codebook_dim),
            p=2,
            dim=-1,
        )  # [batch_size, codebook_num, codebook_dim]
        projected = projected.permute(1, 0, 2).view(
            self.codebook_num, batch_size, 1, self.codebook_dim
        )  # [codebook_num, batch_size, 1, codebook_dim]

        # self.prototypes: [codebook_num, 1, codebook_size, codebook_dim]
        # it is normalized in the function _initialize

        # TODO: configure multiple distances, such as cosine similarity
        # distances = torch.norm(projected - self.prototypes, p=2, dim=-1)
        # [codebook_num, batch_size, codebook_size]

        # Save spaces.
        distances = (
            projected.view(self.codebook_num, batch_size, self.codebook_dim)
            .pow(2)
            .sum(-1, keepdim=True)  # [codebook_num, batch_size, 1]
            + self.prototypes.view(
                self.codebook_num, self.codebook_size, self.codebook_dim
            )
            .pow(2)
            .sum(-1, keepdim=True)
            .transpose(1, 2)  # [codebook_num, 1, codebook_size]
            - 2
            * torch.bmm(
                projected.view(self.codebook_num, batch_size, self.codebook_dim),
                self.prototypes.view(
                    self.codebook_num, self.codebook_size, self.codebook_dim
                ).transpose(1, 2),
            )  # [codebook_num, batch_size, codebook_size]
        )

        codes = torch.argmin(distances, dim=-1).transpose(0, 1)
        # [batch_size, codebook_num]
        return codes


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


def binary_entropy(prob):
    return -prob * log(prob) - (1 - prob) * log(1 - prob)


class LookupFreeQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size,
        entropy_loss_weight=0.1,
        diversity_gamma=2.5,
        straight_through_activation=nn.Tanh(),
    ):
        super().__init__()
        assert math.log2(codebook_size).is_integer()
        self.codebook_dim = int(math.log2(codebook_size))
        self.activation = straight_through_activation
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.register_buffer("mask", 2 ** torch.arange(self.codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.zeros(1), persistent=False)

    def indices_to_codes(self, indices):
        # indices to codes, which are bits of either -1 or 1
        bits = ((indices[..., None].int() & self.mask) != 0).float()
        codes = bits * 2 - 1
        return codes

    def forward(self, x, inv_temperature=1.0):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        """
        # quantize by eq 3.

        ones = torch.ones_like(x)
        quantized = torch.where(x > 0, ones, -ones)

        # use straight-through gradients with tanh (or custom activation fn) if training

        if self.training:
            x = self.activation(x * inv_temperature)
            x = x - x.detach() + quantized
        else:
            x = quantized

        # calculate indices

        indices = reduce((x > 0).int() * self.mask.int(), "b n d -> b n", "sum")

        # entropy aux loss

        if self.training:
            prob = (x * inv_temperature).sigmoid()

            bit_entropy = binary_entropy(prob).mean()

            avg_prob = reduce(prob, "b n d -> b d", "mean")
            codebook_entropy = binary_entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each bit, so each scalar commits to one latent binary bit or the other
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used

            entropy_aux_loss = bit_entropy - self.diversity_gamma * codebook_entropy
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = self.zero

        entropy_aux_loss = entropy_aux_loss * self.entropy_loss_weight

        return x, indices, entropy_aux_loss


class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)
        self.encoder_layers = nn.ModuleList(
            [ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        if config.get("cal_attention_mask", False):
            self.audio_transform = SpeechTransformModified(
                sample_rate=config.sample_rate,
                n_mels=config.n_mels,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                f_min=0,
                f_max=config.sample_rate // 2,
            )
        else:
            self.audio_transform = SpeechTransform(
                sample_rate=config.sample_rate,
                n_mels=config.n_mels,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                f_min=0,
                f_max=config.sample_rate // 2,
            )
        if config.feature_cmvn is not None:
            self.audio_transform.load_from_checkpoint(config.feature_cmvn)

        self.config = config

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def interfere_audio(self, wav_batch):
        interfered_batch = wav_batch.clone()
        b, t = interfered_batch.size()
        primary_indices = np.random.binomial(
            size=b, n=1, p=self.config.mix_prob
        ).astype(bool)
        for primary_i, to_mix in enumerate(primary_indices):
            if to_mix:
                r = ((torch.rand(1) * 10 - 5) / 10)[0].to(wav_batch)
                secondary_i = np.random.randint(b)
                sampled_duration = np.random.randint(1, math.floor(t / 2))
                primary_start = np.random.randint(0, t - sampled_duration)
                secondary_start = np.random.randint(0, t - sampled_duration)
                primary_clip = wav_batch[
                    primary_i, primary_start : primary_start + sampled_duration
                ]
                secondary_clip = wav_batch[
                    secondary_i, secondary_start : secondary_start + sampled_duration
                ]
                scale = (wav_batch[primary_i].square().mean()) / (
                    (wav_batch[secondary_i].square().mean() * torch.pow(10, r) + 1.0e-5)
                ).sqrt()
                interfered_batch[
                    primary_i, primary_start : primary_start + sampled_duration
                ] = (primary_clip + scale * secondary_clip)
        return interfered_batch


class Stage1(Base):
    def __init__(self, config):
        super().__init__(config)
        if config.rq_input_layernorm:
            self.rq_input_layernorm = nn.LayerNorm(
                config.num_channels
                * pow(config.feature_encoder_kernel, config.feature_encoder_padding),
                elementwise_affine=False,
            )
        self.unfolder = nn.Unfold(
            kernel_size=(config.feature_encoder_kernel, 1),
            dilation=1,
            padding=(config.feature_encoder_padding, 0),
            stride=(2, 1),
        )
        self.rq = RandomProjectionQuantizer(config)
        self.rq_head = nn.Linear(
            config.hidden_size,
            config.rq_codebook_size * config.rq_codebook_num,
            bias=False,
        )

    def forward(self, input_dict):
        masked_feature = input_dict["masked_mel"]
        masked_indices = input_dict["masked_indices"]
        flops = self.audio_encoder.get_flops(*masked_feature.shape)
        encoded_masked_feature = self.audio_encoder(masked_feature)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        flops += (
            hidden_states.shape[0]
            * hidden_states.shape[1]
            * self.rq_head.weight.shape[0]
            * self.rq_head.weight.shape[1]
            * 2
        )
        logits = self.rq_head(hidden_states)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.config.rq_codebook_num
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.config.rq_codebook_num
        )

        feature = input_dict["mel"]
        target = self.get_rq_target(feature)
        masked_target = target[tuple(masked_indices.t())]
        masked_target = rearrange(masked_target, "b c -> (b c)")
        output_dict = {
            "rq_logits": logits,
            "rq_masked_logits": masked_logits,
            "rq_target": target,
            "rq_masked_target": masked_target,
            "flops": flops * 3,  # extra 2x for backward.
        }
        return output_dict

    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.unfolder(feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        feature = self._unfold(feature)
        feature = self._unfold(feature)
        return feature

    @torch.no_grad()
    def get_rq_target(self, feature):
        rq_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
        if self.config.rq_input_layernorm:
            rq_input = self.rq_input_layernorm(rq_input)
        target_tokens = self.rq(rq_input)
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=feature.size(0))
        return target_tokens

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        return {"mel": mel}


class Stage2(Base):
    def __init__(self, config):
        super().__init__(config)
        self.mel_head = Conv2dUpsampling(
            config.hidden_size, config.n_mels, use_bn=config.get("use_bn", True)
        )
        if config.get("add_ctc", True):
            self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.add_chroma:
            self.chroma_transform = ChromaSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                n_chroma=config.n_chroma,
                normalized=False,
            )
            self.chroma_head = Conv2dUpsampling(
                config.hidden_size, config.n_chroma, use_bn=config.get("use_bn", True)
            )
        if config.get("add_pitch", False):
            #  must be sr=16000, hop_length=160
            hop_length = config.hop_length * 16000 // config.sample_rate
            print("RMVPE hop_length (on 16k):", hop_length)
            self.rmvpe = RMVPE(hop_length=hop_length)
            self.f0_vuv_head = Conv2dUpsampling(config.hidden_size, 2)
        self.audio_transform.load_from_checkpoint(self.config.feature_cmvn)

    def forward(self, input_dict):
        feature = (
            input_dict["mel_interfered"]
            if self.config.interfere_audio
            else input_dict["mel"]
        )
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)
        output_dict = {"mel_out": mel_out, "flops": flops * 3}  # extra 2x for backward.
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(f0_out=f0_vuv_out[:, :, 0:1])
            output_dict.update(vuv_out=f0_vuv_out[:, :, 1:])
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        if self.config.get("interfere_audio", None):
            x_interfered = self.interfere_audio(x)
            mel_interfered = self.audio_transform(x_interfered, normalize=normalize)
            input_dict.update(mel_interfered=mel_interfered)
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        if self.config.get("add_pitch", False):
            f0 = self.rmvpe.batch_infer(
                x, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            f0 = f0_normalize(f0)
            input_dict.update(f0=f0, vuv=vuv)

        return input_dict


class Stage2Conv(Stage2):
    """Spike to test if training Stage2Conv first can help stabilize Stage3Conv()"""

    def __init__(self, config):
        super().__init__(config)
        Stage3MSSConvEnc_fn = partial(
            Stage3MSSConvEnc,
            config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            stride_times=0,
        )

        self.encoder_layers = nn.ModuleList(
            [Stage3MSSConvEnc_fn() for _ in range(config.num_hidden_layers)]
        )
        del self.embed_positions  # Conv-based encoder doesn't need a position embedding

    def forward(self, input_dict):
        feature = (
            input_dict["mel_interfered"]
            if self.config.interfere_audio
            else input_dict["mel"]
        )
        # Audio frontend preprocessing
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)

        # Apply main conv layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        # Auxiliary heads
        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)
        output_dict = {"mel_out": mel_out, "flops": flops * 3}  # extra 2x for backward.
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(f0_out=f0_vuv_out[:, :, 0:1])
            output_dict.update(vuv_out=f0_vuv_out[:, :, 1:])
        return output_dict


class Stage2Conv1D(Stage2Conv):
    """
    Spike to test if switching from Conv2D to Conv1D in recon heads will solve training stability.

    @hanoihantrakul 2/2/2024: We found that this model led to stable training compared to
    a baseline Stage2 Conformer model.

    @hanoihantrakul 2/22/2024: TODO: the method self.preprocessing() will normalize audio based
    on statistics passed in using `feature_cmvn`. We discovered we were using statistics from
    the MixDataset instead of the MixMSSDataset. This is only a constant difference, and
    should not affect the model performance.
    """

    def __init__(self, config):
        """@hanoihantrakul 31/1/2024: It was faster to delete the original Conv2D heads and replace them with Conv1D."""
        super().__init__(config)

        # Replace all Conv2DUpsampling with Conv1DUpsampling
        # Mel Head to Conv1D
        del self.mel_head
        self.mel_head = Conv1dUpsampling(
            config.hidden_size,
            config.n_mels,
            act_fn=torch.nn.ReLU
            if config.get("act_fn", "relu") == "relu"
            else torch.nn.GELU,
        )

        # Chroma Head to Conv1D
        if config.add_chroma:
            del self.chroma_head
            self.chroma_head = Conv1dUpsampling(
                config.hidden_size,
                config.n_chroma,
                act_fn=torch.nn.ReLU
                if config.get("act_fn", "relu") == "relu"
                else torch.nn.GELU,
            )

        # Supervised Pitch Head to Conv1D
        if config.get("add_pitch", False):
            del self.f0_vuv_head
            self.f0_vuv_head = Conv1dUpsampling(
                config.hidden_size,
                2,
                act_fn=torch.nn.ReLU
                if config.get("act_fn", "relu") == "relu"
                else torch.nn.GELU,
            )


class Stage2Conv1DPitchSupervised(Stage2Conv1D):
    """
    Spike to test if a Stage2Conv1D+SupervisedPitch head loss improves training.
    """

    def __init__(self, config):
        super().__init__(config)

    def preprocessing(self, x):
        """
        @hanoihantrakul Feb 2nd 2024:
        Recommend refactoring this after Conv1D results are verified since it duplicates
        Stage3MSSPitchSupervised() method.
        """
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        if self.config.get("interfere_audio", None):
            x_interfered = self.interfere_audio(x)
            mel_interfered = self.audio_transform(x_interfered, normalize=normalize)
            input_dict.update(mel_interfered=mel_interfered)
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        if self.config.get("add_pitch", False):
            f0 = self.rmvpe.batch_infer(
                x, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            # f0 = f0_normalize(f0) # do not normalize the pitch!
            f0 = raw_hz_to_log1p(f0)
            input_dict.update(f0=f0, vuv=vuv)
        return input_dict


class Stage2Conv1DPitchSupervisedPerceptual(Stage2Conv1DPitchSupervised):
    """
    Spike to test if a Stage2Conv1D+SupervisedPitch+PerceptualPitch improves training.
    """

    def __init__(self, config):
        super().__init__(config)
        if self.config.get("add_perceptual_pitch", False):
            """Replace original mel spec reconstruction head (n_mel=128) with perceptual loss reconstruction head (n_mel=160)."""
            # replace self.mel_head with mel head for perceptual pitch loss
            del self.mel_head
            # for decoding hidden states to mel spectrogram with n_mel=160 instead of n_mel=128
            self.mel_head_full = Stage3MSSDec(
                config, config.head_hidden_size, config.n_mels_tgt
            )
            # the pl_module handles loading the pretrained state_dict of perceptual pitch predictor
            self.pitchpdt = PerceptualPitchPredictor()
            # the pitch predictor needs mel 160 input, not the mel 128 input of default self.audio_transform()
            self.audio_transform_for_pitchpdt = lambda x: torch_wav2spec(
                x, num_mels=config.n_mels_tgt, sample_rate=config.sample_rate
            )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        """Add logic for injecting mel 160 features."""
        input_dict = super().preprocessing(x)
        if self.config.add_perceptual_pitch:
            mel_160 = self.audio_transform_for_pitchpdt(x)
            input_dict.update(mel_160=mel_160)
        return input_dict

    def forward(self, input_dict):
        """Need to remove logic related to n_mels=128 head, we are using only the n_mels_tgt=160 head"""
        feature = (
            input_dict["mel_interfered"]
            if self.config.interfere_audio
            else input_dict["mel"]
        )
        # Audio frontend preprocessing
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)

        # Apply main conv layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        output_dict = {"flops": flops}
        nonpadding = (feature.abs().sum(-1) > 0).float()[..., None]

        if self.config.get("add_perceptual_pitch", False):
            # get predicted hidden state from reconstructed mel spectrogram
            mel_out_full = self.mel_head_full(
                hidden_states, nonpadding
            )  # mel 160, not mel 128
            _ = self.pitchpdt.forward(mel_out_full)
            h_pred = self.pitchpdt.get_hidden_state()  # TODO: flops calculation
            # get reference hidden state from original data
            _ = self.pitchpdt.forward(input_dict["mel_160"])
            h_gt = self.pitchpdt.get_hidden_state()  # TODO: flops calculation
            # sometimes h_gt is 1 timestep longer than h_pred
            trim_len = compute_min_lengths(h_pred, h_gt, axis=1)
            h_pred, h_gt = h_pred[:, :trim_len, :], h_gt[:, :trim_len, :]
            output_dict.update(h_pred=h_pred, h_gt=h_gt, mel_out=mel_out_full)
        else:
            # default to normal mel 128 reconstruction
            flops += self.mel_head.get_flops(*hidden_states.shape)
            mel_out = self.mel_head(hidden_states)
            output_dict.update(mel_out=mel_out)

        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(f0_out=f0_vuv_out[:, :, 0:1])
            output_dict.update(vuv_out=f0_vuv_out[:, :, 1:])
        return output_dict


class Stage2MSS(Stage2):
    """
    Stage 2 Model definition with 2 additional heads for MSS vocal and MSS instrumental.
    """

    def __init__(self, config):
        super().__init__(config)
        self.mel_head = Conv2dUpsampling(
            config.hidden_size, config.n_mels, use_bn=config.get("use_bn", True)
        )
        self.mel_head_vocal = Conv2dUpsampling(
            config.hidden_size, config.n_mels, use_bn=config.get("use_bn", True)
        )
        self.mel_head_inst = Conv2dUpsampling(
            config.hidden_size, config.n_mels, use_bn=config.get("use_bn", True)
        )
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.add_chroma:
            self.chroma_transform = ChromaSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                n_chroma=config.n_chroma,
                normalized=False,
            )
            self.chroma_head = Conv2dUpsampling(
                config.hidden_size, config.n_chroma, use_bn=config.get("use_bn", True)
            )
        if config.get("add_pitch", False):
            #  must be sr=16000, hop_length=160
            hop_length = config.hop_length * 16000 // config.sample_rate
            print("RMVPE hop_length (on 16k):", hop_length)
            self.rmvpe = RMVPE(hop_length=hop_length)
            self.f0_vuv_head = Conv2dUpsampling(config.hidden_size, 2)

        self.flops_counter = 0  # Counter for accumulating flops in forward pass.

    def _reset_flop_counter(self):
        """Reset flop counter to 0 (usually every new forward pass)."""
        self.flops_counter = 0

    def _accumulate_flops(self, flops):
        """Accumulate flops from different computations."""
        self.flops_counter += flops

    def _get_curr_flop_count(self):
        return self.flops_counter

    def _apply_task_heads(self, hidden_states):
        """Apply all task heads to the hidden states and update flops calculation."""
        # Mel Full Audio Head
        self._accumulate_flops(self.mel_head.get_flops(*hidden_states.shape))
        mel_out = self.mel_head(hidden_states)

        # Mel Vocal Head
        self._accumulate_flops(self.mel_head_vocal.get_flops(*hidden_states.shape))
        mel_vocal_out = self.mel_head_vocal(hidden_states)

        # Mel Instrumental Head
        self._accumulate_flops(self.mel_head_inst.get_flops(*hidden_states.shape))
        mel_inst_out = self.mel_head_inst(hidden_states)

        # CTC Head
        self._accumulate_flops(
            2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
        )
        ctc_out = self.ctc_head(hidden_states)
        return mel_out, mel_vocal_out, mel_inst_out, ctc_out

    def _get_encoder_flops(self, hidden_states):
        """Calculate total encoder flops based on hidden state dimensions."""
        return self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )

    def _apply_audio_frontend_encoder(self, feature):
        """Apply encoder on original audio mel features. Different from stack of `self.encoder_layers` which will later be used for VQ training"""
        self._accumulate_flops(self.audio_encoder.get_flops(*feature.shape))
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        return hidden_states

    def _apply_encoder_layers(self, hidden_states, position_embeddings):
        """Apply encoder layers. (In Stage 3 these are modified with Vector Quantization)."""
        self._accumulate_flops(self._get_encoder_flops(hidden_states))
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return hidden_states

    def forward(self, input_dict):
        """
        @hanoihantrakul 11/21/2023:
        Refactored logic from Stage2().forward() to make methods inheritable to Stage3MSS().forward()
        """
        # Reset every new forward pass.
        self._reset_flop_counter()

        # Use mel features from input audio.
        feature = (
            input_dict["mel_interfered"]
            if self.config.interfere_audio
            else input_dict["mel"]
        )

        # Audio Encoder "Frontend" (to disambiguate it from self.encoder_layers later used in VQ training).
        hidden_states = self._apply_audio_frontend_encoder(feature)

        # Positional Embedding.
        position_embeddings = self.embed_positions(hidden_states)

        # Encoder Layers (which later become VQ layers in Stage3).
        hidden_states = self._apply_encoder_layers(hidden_states, position_embeddings)

        # Apply all task heads (1 CTC and 3 Mel) to hidden states.
        mel_out, mel_vocal_out, mel_inst_out, ctc_out = self._apply_task_heads(
            hidden_states
        )

        # Prepare output dictionary.
        output_dict = {
            "mel_out": mel_out,
            "mel_vocal_out": mel_vocal_out,
            "mel_inst_out": mel_inst_out,
            "ctc_out": ctc_out,
            "flops": self._get_curr_flop_count() * 3,  # extra 2x for backward.
        }

        # Apply Chroma Head
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)

        # Apply Pitch Head (Not used in default UMM MSS training)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(
                f0_out=f0_vuv_out[:, :, 0:1], vuv_out=f0_vuv_out[:, :, 1:]
            )
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, audio_dict):
        """
        Arguments()
            audio_dict:
                `audio`: full mix should be routed to normal mel spectrogram, chroma and pitch.
                `audio_vocal`: routed to a vocal mel spectrogram
                `audio_inst`: routed to a instruemtnal mel spectrogram

        Returns:
            input_dict:
                "mel": mel spectrogram of audio full mix
                "mel_vocal" : mel spectrogram of audio vocals
                "mel_inst" : mel spectrogram of audio instrumental
                "chroma" : chroma spectrogram of audio full mix

        @hanoihantrakul 11-25-2023
        The logic of this code should be read in conjunction with `lit_module.Stage2MSS().prepare_feature()`
        """

        def _interfere_audio_handler(audio):
            """Not used in UMM training."""
            audio_interfered = self.interfere_audio(audio)
            mel_interfered = self.audio_transform(audio_interfered, normalize=normalize)
            return mel_interfered

        def _add_chroma_handler(audio):
            chroma = self.chroma_transform(audio)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            return chroma

        def _add_pitch_handler(audio):
            """Not used in UMM training."""
            f0 = self.rmvpe.batch_infer(
                audio, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            f0 = f0_normalize(f0)
            return f0, vuv

        normalize = self.config.feature_cmvn is not None
        input_dict = {
            "mel": self.audio_transform(audio_dict["audio"], normalize=normalize),
            "mel_vocal": self.audio_transform(
                audio_dict["audio_vocal"], normalize=normalize
            ),
            "mel_inst": self.audio_transform(
                audio_dict["audio_inst"], normalize=normalize
            ),
        }
        if self.config.get("interfere_audio", None):
            # "interfere_audio" is a historical flag and should be assumed to be False by default.
            input_dict.update(
                mel_interfered=_interfere_audio_handler(audio_dict["audio"])
            )
        if self.config.add_chroma:
            input_dict.update(chroma=_add_chroma_handler(audio_dict["audio"]))
        if self.config.get("add_pitch", False):
            # "add_pitch" is from TTS team and should be assumed to be False by default.
            f0, vuv = _add_pitch_handler(audio_dict["audio"])
            input_dict.update(f0=f0, vuv=vuv)

        return input_dict


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class Stage3(Stage2):
    def __init__(self, config):
        super().__init__(config)
        if config.get("vq_type", None) == "CVQ":
            self.vq = ClusteredVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                distance=config.get("vq_distance", "cos"),
            )
        elif config.get("vq_type", None) == "FSQ":
            self.vq = FiniteScalarQuantizer(codebook_size=config.vq_codebook_size)
        elif config.get("vq_type", None) == "LFQ":
            self.vq = LookupFreeQuantizer(codebook_size=config.vq_codebook_size)
        elif config.get("vq_type", None) == "EMAEntropy":
            self.vq = EMAVectorQuantizerEntropy(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        else:
            self.vq = EMAVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        if config.get("vq_proj_norm", None) == "bn":
            self.vq_proj_in = nn.Sequential(
                Transpose(),
                WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1)
                if config.hidden_size != config.vq_codebook_dim
                else nn.Identity(),
                nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
                Transpose(),
            )
            self.vq_proj_out = nn.Sequential(
                Transpose(),
                WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity(),
                Transpose(),
            )
        elif config.get("vq_proj_norm", None) == "ln":
            self.vq_proj_in = nn.Sequential(
                nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False)
                if config.hidden_size != config.vq_codebook_dim
                else nn.Identity(),
                nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
            )
            self.vq_proj_out = nn.Sequential(
                nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity()
            )
        else:
            self.vq_proj_in = nn.Linear(
                config.hidden_size, config.vq_codebook_dim, bias=False
            )
            self.vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

    def forward(self, input_dict):
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(
                        hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0
                    )
                else:
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                hidden_states = self.vq_proj_out(vq_embs)
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)
        output_dict = {
            "mel_out": mel_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "flops": flops * 3,  # extra 2x for backward.
        }
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.get("vq_proj_noise", False):
            output_dict.update(noise_scale=noise_scale)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        return output_dict

    def forward_layers(
        self, audio_embedding: torch.Tensor, layer_idx: int
    ) -> UMMResult:
        hidden_states = self.encoder_input_dropout(audio_embedding)
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.encoder_layers):
            if i == layer_idx:
                pre_vq_in = self.vq_proj_in(hidden_states)
                vq_embs, vq_ids, vq_loss = self.vq(pre_vq_in)
                return UMMResult(
                    hidden_states=hidden_states,
                    vq_ids=vq_ids,
                    vq_hidden_states=vq_embs,
                    vq_loss=vq_loss,
                )

            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _prepare_wav(self, wav):
        """Check audio dimensions and pad."""
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        return self.pad_audio(wav.float())

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _get_vq_ids(self, hidden_states, position_embeddings):
        """Apply Vector Quantization and only get the ID's."""
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                vq_hidden_states = self.vq_proj_in(hidden_states)
                _, vq_ids, _ = self.vq(vq_hidden_states)
                return {"vq_ids": vq_ids, "hidden_states": hidden_states,  "vq_hidden_states": vq_hidden_states}
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return {"vq_ids": vq_ids, "hidden_states": hidden_states}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        """Convert audio file to tokens (after Vector Quantization)."""
        wav = self._prepare_wav(wav)
        feature = self.preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        result = self._get_vq_ids(hidden_states, position_embeddings)
        return result["vq_ids"]

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token_alloutputs(self, wav, mel: Optional[torch.Tensor] = None):
        wav = self._prepare_wav(wav)
        feature = self.preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        result = self._get_vq_ids(hidden_states, position_embeddings)
        return result

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2audio_embed(self, wav):
        """Convert audio file to mel spectrogram embeddings."""
        wav = self._prepare_wav(wav)
        feature = self.preprocessing(wav)["mel"]
        encoded_feature = self.audio_encoder(feature)
        return encoded_feature

    def wav2hidden_states(self, audio: torch.Tensor, layer_idx: int) -> UMMResult:
        """Convert audio file to hidden states (before Vector Quantization)."""
        audio_embedding = self.wav2audio_embed(audio)
        return self.forward_layers(audio_embedding, layer_idx)


class Stage3MSS(Stage2MSS, Stage3):
    """
    Stage3 UMM Model configured to receive Stage2 UMM checkpoint.

    @hanoihantrakul 11/21/2023
    In writing this class, it was easiest to inherit from both Stage3() and Stage2MSS().
    - Stage3() contains configuration for different VQ options in the __init__() method.
    - Stage2MSS() contains configuration for the 2 mel heads that are kept during Stage3 training.

    @hanoihantrakul 11/25/2023
    - Stage2MSS() should be inherited first so that order-of-resolution `self.preprocessing()`
    calls the methods from Stage2MSS (which process 3 audio inputs) and not Stage3.
    """

    def __init__(self, config):
        # Define identical model from Stage2MSS() class so weights can be loaded.
        # Reuse VQ configuration logic from original Stage3() class.
        super().__init__(config)

    def _apply_vq_and_encoder_layers(self, hidden_states, position_embeddings):
        """Apply VQ. Logic copy-pasted from Stage3().forward() and combined with Stage2MSS()._apply_encoder_layers."""
        self._accumulate_flops(self._get_encoder_flops(hidden_states))

        for i, layer in enumerate(self.encoder_layers):
            # Handle Vector Quanization.
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                else:
                    noise_scale = 0
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(
                        hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0
                    )
                else:
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                hidden_states = self.vq_proj_out(vq_embs)
            # Handle encoder layers.
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return hidden_states, vq_ids, vq_loss, noise_scale

    def forward(self, input_dict):
        # Reset every new forward pass.
        self._reset_flop_counter()

        # Use mel features from input audio.
        feature = (
            input_dict["mel_interfered"]
            if self.config.interfere_audio
            else input_dict["mel"]
        )

        # Audio Encoder "Frontend" (to disambiguate it from self.encoder_layers later used in VQ training).
        hidden_states = self._apply_audio_frontend_encoder(feature)

        # Positional Embedding.
        position_embeddings = self.embed_positions(hidden_states)

        # Encoder Layers from Stage 2 are now used as part of VQ in Stage 3.
        hidden_states, vq_ids, vq_loss, noise_scale = self._apply_vq_and_encoder_layers(
            hidden_states, position_embeddings
        )

        # Apply all task heads (1 CTC and 3 Mel) to VQ hidden states.
        mel_out, mel_vocal_out, mel_inst_out, ctc_out = self._apply_task_heads(
            hidden_states
        )

        # Prepare output dictionary.
        output_dict = {
            "mel_out": mel_out,
            "mel_vocal_out": mel_vocal_out,
            "mel_inst_out": mel_inst_out,
            "ctc_out": ctc_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "noise_scale": noise_scale,
            "flops": self._get_curr_flop_count() * 3,  # extra 2x for backward.
        }

        # Apply Chroma Head
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)

        # Apply Pitch Head (Not used in default UMM MSS training)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(
                f0_out=f0_vuv_out[:, :, 0:1], vuv_out=f0_vuv_out[:, :, 1:]
            )
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        """
        Convert audio files to tokens.

        @hanoihantrakul 25-11-2023
        Training a downstream decoder ontop of Stage3MSS requires this method to be called on a mixture of audio datasets.
        By default, these downstream decoder datasets have one audio input (full mix),
        whereas the Stage3MSS model expected 3 audio inputs (full mix, instrumental and vocal).
        The easiest way to modify Stage3MSS behavior was to explicitly call Stage3().preprocessing
        (which expects 1 audio input) instead of the inherited default Stage2MSS().preprocessing function (which expects 3 audio inputs).
        """
        wav = self._prepare_wav(wav)
        # Using super() like this is unpythonic, but it was the fastest way to favor experimentation speed.
        feature = super(Stage3, self).preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        vq_ids = self._get_vq_ids(hidden_states, position_embeddings)
        return vq_ids["vq_ids"]


class Stage3MSSDec(nn.Module):
    def __init__(
        self, config, hidden_size, output_dim, expand_times=2, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.proj_in1 = nn.Conv1d(config.hidden_size, hidden_size, 5, padding=2)
        self.proj_in2 = nn.Conv1d(hidden_size, hidden_size, 5, padding=2)
        self.conv_stacks = ConvBlocks(
            hidden_size,
            output_dim,
            None,
            5,
            layers_in_block=2,
            num_layers=8,
            norm_type="ln",
            dropout=0,
            post_net_kernel=3,
            is_BTC=False,
        )
        self.expand_times = expand_times

    def forward(self, x, nonpadding):
        if nonpadding is None:
            nonpadding = (x.abs().sum(-1) > 0).float()[..., None]
        nonpadding = nonpadding.transpose(1, 2)
        x = x.transpose(1, 2)
        if self.expand_times == 2:
            x = self.proj_in1(x) * F.interpolate(
                nonpadding, size=x.shape[-1], mode="nearest"
            )
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.proj_in2(x) * F.interpolate(
                nonpadding, size=x.shape[-1], mode="nearest"
            )
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        elif self.expand_times == 1:
            x = self.proj_in1(x) * F.interpolate(
                nonpadding, size=x.shape[-1], mode="nearest"
            )
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_stacks(
            x, F.interpolate(nonpadding, size=x.shape[-1], mode="nearest")
        )
        x = x.transpose(1, 2)
        return x


class Stage3MSSConvEnc(nn.Module):
    def __init__(
        self, hidden_size, input_dim, output_dim, stride_times=2, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if stride_times == 2:
            self.proj_in1 = nn.Conv1d(input_dim, hidden_size, 5, padding=2, stride=2)
            self.proj_in2 = nn.Conv1d(hidden_size, output_dim, 5, padding=2, stride=2)
        else:
            self.proj_in1 = nn.Conv1d(input_dim, hidden_size, 3, padding=1, stride=1)
            self.proj_in2 = nn.Conv1d(hidden_size, output_dim, 3, padding=1, stride=1)
        self.conv_stacks = ConvBlocks(
            hidden_size,
            hidden_size,
            None,
            5,
            layers_in_block=2,
            num_layers=8,
            norm_type="ln",
            dropout=0,
            post_net_kernel=3,
            is_BTC=False,
        )
        self.stride_times = stride_times

    def forward(self, x, nonpadding=None):
        if nonpadding is None:
            nonpadding = (x.abs().sum(-1) > 0).float()[..., None]
        nonpadding = nonpadding.transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.proj_in1(x)
        nonpadding = F.interpolate(nonpadding, size=x.shape[-1], mode="nearest")
        x = x * nonpadding
        x = self.conv_stacks(x, nonpadding)
        x = self.proj_in2(x)
        nonpadding = F.interpolate(nonpadding, size=x.shape[-1], mode="nearest")
        x = x * nonpadding
        x = x.transpose(1, 2)
        return x


class Stage3Conv1D(Stage2Conv1D, Stage3):
    """
    Spike to test if switching from Conv2D to Conv1D in recon heads will solve training stability.

    @hanoihantrakul 2/4/2024: We found that the Stage2Conv1D was stable. Spiking to see if
    Stage3Conv1D is also stable.

    @hanoihantrakul 2/4/2024: TODO: this inheritance structure is becoming extremely
    difficult to debug. Recommend refactoring stage1,2,3 code after spike is complete.
    For example, this class requires inheriting from two related stages (Stage2Conv1D and Stage3)

    @hanoihantrakul 2/22/2024: TODO: the method self.preprocessing() will normalize audio based
    on statistics passed in using `feature_cmvn`. We discovered we were using statistics from
    the MixDataset instead of the MixMSSDataset. This is only a constant difference, and
    should not affect the model performance.
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_dict):
        """Override Stage3 method and remove positional embedding."""
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)

        for i, layer in enumerate(self.encoder_layers):
            # at specific layer idx, vector quantize hidden states before applying the layer
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(
                        hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0
                    )
                else:
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                hidden_states = self.vq_proj_out(vq_embs)
            # apply layer
            hidden_states = layer(hidden_states)
        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)
        output_dict = {
            "mel_out": mel_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "flops": flops * 3,  # extra 2x for backward.
        }
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.get("vq_proj_noise", False):
            output_dict.update(noise_scale=noise_scale)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(f0_out=f0_vuv_out[:, :, 0:1])
            output_dict.update(vuv_out=f0_vuv_out[:, :, 1:])
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _get_vq_ids(self, hidden_states):
        """Override Stage3 method and remove positional embedding."""
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                _, vq_ids, _ = self.vq(hidden_states)
                return vq_ids
            hidden_states = layer(hidden_states)
        return vq_ids

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        """Override Stage3 method and remove embed_positions."""
        wav = self._prepare_wav(wav)
        feature = self.preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        # position_embeddings = self.embed_positions(hidden_states)
        vq_ids = self._get_vq_ids(hidden_states)
        return vq_ids


class Stage3Conv1D_v2(Stage3Conv1D):
    """
    Stage3Conv1D_v2 model that corrects the audio encoder to use Conv1D. Also
    note that by the time this class was written (29Feb2024) we discovered
    that it is possible to train direct-to-stage3. Thus, this class definition
    expects to be trained directly without stage 1 and 2.

    @hanoihantrakul 29Feb2024: We discovered that the original Stage3Conv1D
    model was correctly using Conv1D in self.encoder_layers and self.mel_head
    but was incorrectly using Conv2D in self.audio_encoders. In this
    implementation I use the same self.audio_encoder definition as
    the DualUMM model defined in `dualumm.py`

    TODO: If this works @hanoihantrakul will refactor ConvUMM into
    a standalone file.
    """

    def __init__(self, config):
        """
        Replace self.audio_encoder with DualUMM ConvEncoder.

        @hanoihantrakul 29Feb2024:
        - Note that the old self.audio_encoder used
        umm_mkii.Conv2dSubsampling() with 2 sets of stride 2 i.e 4x downsampling.
        Here we use dualumm.ConvStacksWithDownUpSampling() with identical
        4x downsampling.

        - Another difference is dualumm.ConvStacksWithDownUpSampling() targets
        a mel 160 spectrogram, whereas umm_mkii.Conv2dSubsampling() targets a
        mel 128 spectrogram. The reason is only because I want to use the same
        functions developed in DualUMM.
        """
        super().__init__(config)
        # Remove Conv1D based initialization.
        del self.audio_encoder
        self.audio_encoder = ConvStacksWithDownUpSampling(
            config.hidden_size,
            config.n_mels_tgt,
            config.hidden_size,
            downsampling=config.downsampling,
            upsampling=config.upsampling,
        )

        # Remove mel-128 audio transform and replace with mel-160
        del self.audio_transform
        self.audio_transform = lambda x: torch_wav2spec(
            x, num_mels=config.n_mels_tgt, sample_rate=config.sample_rate
        )

        # Replace all Conv2DUpsampling with Conv1DUpsampling
        # Mel Head to Conv1D
        del self.mel_head
        self.mel_head = Conv1dUpsampling(
            config.hidden_size,
            config.n_mels_tgt,
            act_fn=torch.nn.ReLU
            if config.get("act_fn", "relu") == "relu"
            else torch.nn.GELU,
        )

        # Chroma Head to Conv1D
        if config.add_chroma:
            del self.chroma_head
            self.chroma_head = Conv1dUpsampling(
                config.hidden_size,
                config.n_chroma,
                act_fn=torch.nn.ReLU
                if config.get("act_fn", "relu") == "relu"
                else torch.nn.GELU,
            )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        """
        @hanoihantrakul 3Mar2024
        - Just change `mel = self.audio_transform(x, normalize=normalize)` to mel = self.audio_transform(x)
        - If this works, recommend refactoring this class into a new one.
        """
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x)
        input_dict = {"mel": mel}
        if self.config.get("interfere_audio", None):
            x_interfered = self.interfere_audio(x)
            mel_interfered = self.audio_transform(x_interfered, normalize=normalize)
            input_dict.update(mel_interfered=mel_interfered)
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        if self.config.get("add_pitch", False):
            f0 = self.rmvpe.batch_infer(
                x, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            f0 = f0_normalize(f0)
            input_dict.update(f0=f0, vuv=vuv)

        return input_dict

    def forward(self, input_dict):
        """Override Stage3 method by removing positional embedding and adding length checks"""
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)

        for i, layer in enumerate(self.encoder_layers):
            # at specific layer idx, vector quantize hidden states before applying the layer
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(
                        hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0
                    )
                else:
                    # default
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                    # calculate codebook distances by accessing the VQ's internal matrix representing the actual codebook
                    codebook_distance_stats = get_vq_codebook_distances(self.vq.embedding.weight.data)
                hidden_states = self.vq_proj_out(vq_embs)
            # apply layer
            hidden_states = layer(hidden_states)
        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)

        """
        @hanoihantrakul 3Mar2024
        `mel` (ground truth) can sometimes be 1 sample longer than `mel_out` (predicted)
        - Just for this one config only, correct for this 1 sample difference.
        - e.g. [6, 2917, 160] vs [6, 2916, 160]
        - This will not be a problem if a compeletely new class is defined without inheritance from Stage1 and Stage2
        """
        trim_len = compute_min_lengths(input_dict["mel"], mel_out, axis=1)
        input_dict["mel"] = input_dict["mel"][:, :trim_len, :]
        mel_out = mel_out[:, :trim_len, :]

        output_dict = {
            "mel_out": mel_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "flops": flops * 3,  # extra 2x for backward.
        }
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.get("vq_proj_noise", False):
            output_dict.update(noise_scale=noise_scale)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(f0_out=f0_vuv_out[:, :, 0:1])
            output_dict.update(vuv_out=f0_vuv_out[:, :, 1:])
        # add the codebook stats just for Stage3Conv1D_v2 models
        output_dict.update(codebook_distance_stats)
        return output_dict


class Stage3MSSPitchSupervised(Stage3):
    """
    Create a Stage3 model for testing the effect of adding a supervised pitch head.
    """

    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        """
        @hanoihantrakul 1/22/2024:
        Inject logic so that f0 is not normalized. Everything else identical to Stage3().
        """
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        if self.config.get("interfere_audio", None):
            x_interfered = self.interfere_audio(x)
            mel_interfered = self.audio_transform(x_interfered, normalize=normalize)
            input_dict.update(mel_interfered=mel_interfered)
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        if self.config.get("add_pitch", False):
            f0 = self.rmvpe.batch_infer(
                x, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            # f0 = f0_normalize(f0) # do not normalize the pitch!
            f0 = raw_hz_to_log1p(f0)
            input_dict.update(f0=f0, vuv=vuv)
        return input_dict


class Stage3MSSPitchSupervisedPerceptual(Stage3MSSPitchSupervised):
    """
    Create a Stage3 model for testing the effect of adding a supervised pitch head
    and a perceptual pitch head. It is a quick spike to verify whether these pitch
    losses are effective and useful for UMMv2 training on MSS data.

    @hanoihantrakul 1/22/2024: This class is similar in logic to Stage3 except:
    - There is an additional `add_perceptual_pitch=True` configuration. I assume
    this class will only ever be used with `add_pitch=True` and `add_perceptual_pitch=True`
    """

    def __init__(self, config):
        super().__init__(config)
        if self.config.get("add_perceptual_pitch", False):
            """Replace original mel spec reconstruction head (n_mel=128) with perceptual loss reconstruction head (n_mel=160)."""
            # replace self.mel_head with mel head for perceptual pitch loss
            del self.mel_head
            # for decoding hidden states to mel spectrogram with n_mel=160
            self.mel_head_full = Stage3MSSDec(
                config, config.head_hidden_size, config.n_mels_tgt
            )
            # the pl_module handles loading the pretrained state_dict of perceptual pitch predictor
            self.pitchpdt = PerceptualPitchPredictor()
            # the pitch predictor needs mel 160 input, not the mel 128 input of self.audio_transform()
            self.audio_transform_for_pitchpdt = lambda x: torch_wav2spec(
                x, num_mels=config.n_mels_tgt, sample_rate=config.sample_rate
            )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        """Add logic for injecting mel 160 features."""
        input_dict = super().preprocessing(x)
        if self.config.add_perceptual_pitch:
            mel_160 = self.audio_transform_for_pitchpdt(x)
            input_dict.update(mel_160=mel_160)
        return input_dict

    def forward(self, input_dict):
        """
        Identical to Stage3.forward() except:
        - If `add_perceptual_pitch=True` then replace default mel head
        - It was too complicated to re-use nicely refactored Stage2MSS(). Oh well.
        """
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(
            feature
        )  # we still need mel 128 here to create the encoder input
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )
        nonpadding = (feature.abs().sum(-1) > 0).float()[..., None]

        # Vector Quantization Step
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(
                        hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0
                    )
                else:
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                hidden_states = self.vq_proj_out(vq_embs)
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        output_dict = {
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "flops": flops * 3,  # extra 2x for backward.
        }

        if self.config.get("add_perceptual_pitch", False):
            # get predicted hidden state from reconstructed mel spectrogram
            mel_out_full = self.mel_head_full(
                hidden_states, nonpadding
            )  # mel 160, not mel 128
            _ = self.pitchpdt.forward(mel_out_full)
            h_pred = self.pitchpdt.get_hidden_state()  # TODO: flops calculation
            # get reference hidden state from original data
            _ = self.pitchpdt.forward(input_dict["mel_160"])
            h_gt = self.pitchpdt.get_hidden_state()  # TODO: flops calculation
            # sometimes h_gt is 1 timestep longer than h_pred
            trim_len = compute_min_lengths(h_pred, h_gt, axis=1)
            h_pred, h_gt = h_pred[:, :trim_len, :], h_gt[:, :trim_len, :]
            output_dict.update(h_pred=h_pred, h_gt=h_gt)
        else:
            # default to normal mel 128 reconstruction
            flops += self.mel_head.get_flops(*hidden_states.shape)
            mel_out = self.mel_head(hidden_states)
            output_dict.update(mel_out=mel_out)

        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.get("vq_proj_noise", False):
            output_dict.update(noise_scale=noise_scale)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(
                f0_out=f0_vuv_out[:, :, 0:1], vuv_out=f0_vuv_out[:, :, 1:]
            )
        return output_dict
