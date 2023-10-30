import math
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, reduce, pack, unpack
from torch import Tensor, int32, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.transforms.speech import SpeechTransform
from samantha.utils.hparams import DotDict

from mariana.models.audio.usm_encoder import UsmEncoder
from mariana.data.audio.transforms import KaldiFbank, CMVN
from mariana.models.audio.infer_utils import PantherInfer, replace_with_panther_conformer_layer
from mariana.models.audio.misc import rnnt_transpose


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


@dataclass
class ConformerEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ConformerRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = 0
        self.cached_rotary_positional_embedding = None

    def _set_cos_sin_cache(self, sequence_length):
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
        sequence_length = hidden_states.shape[1]
        if (
            sequence_length > self.cached_sequence_length
            or self.cached_rotary_positional_embedding is None
        ):
            self._set_cos_sin_cache(sequence_length)
        return self.cached_rotary_positional_embedding[:, -sequence_length:].to(
            dtype=hidden_states.dtype
        )


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

        return b * t * (
            linear_flops(self.intermediate_dense) +
            linear_flops(self.output_dense)
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
        self.batch_norm = torch.nn.BatchNorm1d(config.hidden_size)
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
            return m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2] / m.groups * 2

        return n * l * (
            conv1d_flops(self.pointwise_conv1) +
            conv1d_flops(self.depthwise_conv) +
            conv1d_flops(self.pointwise_conv2)
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

        return b * t * (
            linear_flops(self.linear_q) +
            linear_flops(self.linear_k) +
            linear_flops(self.linear_v) +
            linear_flops(self.linear_out)
        ) + b * t * t * self.head_size * 2 + b * t * self.head_size * t * 2


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = q.size()

        query = self.linear_q(q).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(k).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(v).view(batch_size, -1, self.num_heads, self.head_size)

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
        self, hidden_states, position_embeddings: Optional[torch.Tensor] = None
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
            hidden_states=hidden_states, position_embeddings=position_embeddings
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
            self.ffn1.get_flops(b, t) +
            self.self_attn.get_flops(b, t) +
            self.conv_module.get_flops(b, t) +
            self.ffn2.get_flops(b, t)
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

    def forward(self, hidden_states, output_hidden_states=False):
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
                    create_custom_forward(layer), hidden_states, position_embeddings
                )
            else:
                layer_outputs = layer(
                    hidden_states, position_embeddings=position_embeddings
                )
            hidden_states = layer_outputs

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return ConformerEncoderOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


class Conv2dUpsampling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            # [1, 1, 750, 1024]
            nn.Conv2d(1, 64, 7, 1, 3),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            # [1, 64, 750, 1024]
            nn.ConvTranspose2d(64, 8, 6, 2, 2),
            torch.nn.BatchNorm2d(8),
            nn.ReLU(),
            # [1, 8, 1500, 2048]
            nn.ConvTranspose2d(8, 1, 6, 2, 2),
            torch.nn.BatchNorm2d(1),
            nn.ReLU(),
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


class Conv2dSubsampling(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel, 2, padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel, 2, padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.linear = nn.Linear(input_dim * 64, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_encoder = Conv2dSubsampling(
            config.num_channels,
            config.hidden_size,
            config.feature_encoder_kernel,
            config.feature_encoder_padding,
        )
        self.conformer_layer = ConformerEncoderLayer(config)

    def forward(self, x):
        x = self.feature_encoder(x)
        x = self.conformer_layer(x)
        return x

    def get_flops(self, b, t):
        return self.conformer_layer.get_flops(b, t / 4)


class EMAEmbedding(nn.Module):
    def __init__(self, codebook_size, codebook_dim, decay=0.99, eps=1e-5, learnable=False, orthonormal_init=False):
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
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg.data, alpha=1 - self.decay)

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
        topk_res = torch.topk(cs[1:], k=cs.shape[-1] - 1) # 大到小
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
        z_q = self.embedding(min_encoding_indices).view(z.shape)  # [b*h, c] -> [b, h, c]
        # EMA update
        if self.training and self.embedding.update:
            one_hot = F.one_hot(min_encoding_indices, self.codebook_size).type(z.dtype)  # [b*h, k]
            # EMA cluster size
            one_hot_sum = one_hot.sum(0)  # [k]
            if self.dist:
                torch.distributed.all_reduce(one_hot_sum)
            self.embedding.cluster_size_ema_update(one_hot_sum)
            # EMA embedding average
            embed_sum = (one_hot.transpose(0, 1) @ z_flattened)  # [k, b*h] * [b*h, c] = [k, c]
            if self.dist:
                torch.distributed.all_reduce(embed_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.codebook_size)

        loss = torch.mean((z_q.detach() - z) ** 2) + e_scale * self.entropy_loss(-d, loss_type='softmax')
        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.same_index_shape:
            min_encoding_indices = rearrange(
                min_encoding_indices, "(b t) -> b t", t=z.size(1)
            )

        return z_q, min_encoding_indices, loss

    def entropy_loss(self, affinity, loss_type="softmax", temperature=0.7, eps=1e-10):
        # affinity: [b, t, d_n]
        flat_affinity = affinity.reshape(-1, affinity.shape[-1])  # [b, t, d_n] -> [b*t, d_n]
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
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
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

        encoding_indices = rearrange(
            encoding_indices, "(b t) -> b t", t=z.size(1)
        )

        return z_q, encoding_indices, loss

    @torch.no_grad()
    def entropy(self):
        p = self.embed_prob / self.embed_prob.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        return entropy


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


# helper functions
def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def binary_entropy(prob):
    return -prob * log(prob) - (1 - prob) * log(1 - prob)

class LookupFreeQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size,
        entropy_loss_weight = 0.1,
        diversity_gamma = 2.5,
        straight_through_activation = nn.Tanh()
    ):
        super().__init__()
        assert math.log2(codebook_size).is_integer()
        self.codebook_dim = int(math.log2(codebook_size))
        self.activation = straight_through_activation
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.register_buffer('mask', 2 ** torch.arange(self.codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.zeros(1,), persistent = False)

    def indices_to_codes(self, indices):
        # indices to codes, which are bits of either -1 or 1
        bits = ((indices[..., None].int() & self.mask) != 0).float()
        codes = bits * 2 - 1
        return codes

    def forward(self, x, inv_temperature = 1.0):
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
        indices = reduce((x > 0).int() * self.mask.int(), 'b n d -> b n', 'sum')
        # entropy aux loss
        if self.training:
            prob = (x * inv_temperature).sigmoid()
            bit_entropy = binary_entropy(prob).mean()
            avg_prob = reduce(prob, 'b n d -> b d', 'mean')
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
        flops = 0

        flops += self.audio_encoder.get_flops(*masked_feature.shape[0:2])
        encoded_masked_feature = self.audio_encoder(masked_feature)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)
        position_embeddings = self.embed_positions(hidden_states)
        for layer in self.encoder_layers:
            flops += layer.get_flops(*hidden_states.shape[0:2])
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        flops += (
            hidden_states.shape[0] * hidden_states.shape[1] *
            self.rq_head.weight.shape[0] * self.rq_head.weight.shape[1] * 2
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
            "flops": flops * 3  # extra 2x for backward.
        }
        return output_dict

    @torch.cuda.amp.autocast(enabled=False)
    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.unfolder(feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    @torch.cuda.amp.autocast(enabled=False)
    def _subsample(self, feature):
        feature = self._unfold(feature)
        feature = self._unfold(feature)
        return feature

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
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
        self.mel_head = Conv2dUpsampling(config.hidden_size, config.n_mels)
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
            self.chroma_head = Conv2dUpsampling(config.hidden_size, config.n_chroma)

    def forward(self, input_dict):
        feature = (
            input_dict["mel_interfered"]
            if self.config.interfere_audio
            else input_dict["mel"]
        )
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        mel_out = self.mel_head(hidden_states)
        ctc_out = self.ctc_head(hidden_states)
        output_dict = {"mel_out": mel_out, "ctc_out": ctc_out}
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        if self.config.interfere_audio:
            x_interfered = self.interfere_audio(x)
            mel_interfered = self.audio_transform(x_interfered, normalize=normalize)
            input_dict.update(mel_interfered=mel_interfered)
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
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
            self.vq = FiniteScalarQuantizer(
                codebook_size=config.vq_codebook_size,
            )
        elif config.get("vq_type", None) == "LFQ":
            self.vq = LookupFreeQuantizer(
                codebook_size=config.vq_codebook_size,
            )
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
                WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1) if config.hidden_size != config.vq_codebook_dim else nn.Identity(),
                nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
                Transpose(),
            )
            self.vq_proj_out = nn.Sequential(
                Transpose(),
                WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1) if config.vq_codebook_dim != config.hidden_size else nn.Identity(),
                Transpose(),
            )
        elif config.get("vq_proj_norm", None) == "ln":
            self.vq_proj_in = nn.Sequential(
                nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False) if config.hidden_size != config.vq_codebook_dim else nn.Identity(),
                nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
            )
            self.vq_proj_out = nn.Sequential(
                nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False) if config.vq_codebook_dim != config.hidden_size else nn.Identity(),
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
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        # For AMP, the follwing three parts use precision 32
        # 1. VQ layer
        # 2. the last layer of conformer
        # 3. mel head, ctc head, chroma head
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                with torch.cuda.amp.autocast(enabled=False):
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
                        vq_embs, vq_ids, vq_loss = self.vq(hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0)
                    else:
                        vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                    hidden_states = self.vq_proj_out(vq_embs)
                    hidden_states = layer(
                        hidden_states, position_embeddings=position_embeddings
                    )
            elif i == len(self.encoder_layers) - 1:
                with torch.cuda.amp.autocast(enabled=False):
                    hidden_states = layer(
                        hidden_states, position_embeddings=position_embeddings
                    )
            else:
                hidden_states = layer(
                    hidden_states, position_embeddings=position_embeddings
                )
        with torch.cuda.amp.autocast(enabled=False):
            mel_out = self.mel_head(hidden_states)
            ctc_out = self.ctc_head(hidden_states)
            output_dict = {
                "mel_out": mel_out,
                "ctc_out": ctc_out,
                "vq_ids": vq_ids,
                "vq_loss": vq_loss,
            }
            if self.config.get("vq_proj_noise", False):
                output_dict.update(noise_scale=noise_scale)
            if self.config.add_chroma:
                chroma_out = self.chroma_head(hidden_states)
                output_dict.update(chroma_out=chroma_out)

        return output_dict

    @torch.no_grad()
    def wav2token(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        wav = self.pad_audio(wav.float())
        feature = self.preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                return vq_ids
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return vq_ids


class USMStage2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        self.fbank_fn = KaldiFbank(dither=0.0, out_numpy=False, device='cuda')
        self.cmvn_fn = CMVN(key="fbank",
                            cmvn_mean=np.load('recipes/datasets/mcc/usm_mean.npy'),
                            cmvn_var=np.load('recipes/datasets/mcc/usm_var.npy'))
        # model define
        usm_config = DotDict(config.usm_config)
        self.audio_encoder = UsmEncoder(usm_config.network)
        self.mel_head = Conv2dUpsampling(config.hidden_size, config.n_mels)
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
            self.chroma_head = Conv2dUpsampling(config.hidden_size, config.n_chroma)

    def forward(self, input_dict, return_hidden_states=False):
        feature = input_dict["fbank"]
        src_mask = input_dict["src_mask"]
        ### one forward ###
        # hidden_states = self.audio_encoder(feature, src_mask, is_training=self.training)
        ### Let's do it step by step ###
        # copy from `mariana.models.audio.usm_encoder.py`
        # copy from `mariana.models.audio.conformer.py`
        front_end_out, backbone_mask, frontend_shape = self.audio_encoder.frontend(feature, src_mask)
        conformers = self.audio_encoder.acoustic_backbone_module
        all_hidden_states = []
        conformer_input = conformers.pos_enc(front_end_out)
        if backbone_mask is None:
            conformer_mask = None
        else:
            conformer_mask = backbone_mask.unsqueeze(1)
        attn_weights = None
        for i, layer in enumerate(conformers.encoders):
            conformer_input, conformer_mask = layer(
                [conformer_input, conformer_mask], is_training=self.training
            )
            if return_hidden_states and not conformers.normalize_before:
                all_hidden_states.append(conformer_input)
        if isinstance(conformer_input, (tuple, list)):
            conformer_input = conformer_input[0]
        if conformers.normalize_before:
            conformer_input = conformers.after_norm(conformer_input)
            if return_hidden_states:
                all_hidden_states.append(conformer_input)
        # return conformer_input
        with torch.cuda.amp.autocast(enabled=False):
            hidden_states = conformer_input
            mel_out = self.mel_head(hidden_states)
            ctc_out = self.ctc_head(hidden_states)
            output_dict = {"mel_out": mel_out, "ctc_out": ctc_out}
            if self.config.add_chroma:
                chroma_out = self.chroma_head(hidden_states)
                output_dict.update(chroma_out=chroma_out)
            if return_hidden_states:
                all_hidden_states = [h[0] if isinstance(h, (list, tuple)) else h for h in all_hidden_states]
                output_dict["hidden_states"] = all_hidden_states
        return output_dict

    @torch.no_grad()
    def extract_features(self, wavs, dtype=torch.float32):
        is_amp = (dtype in [torch.float16, torch.bfloat16])
        input_dict = self.preprocessing(wavs)
        with torch.cuda.amp.autocast(enabled=is_amp, dtype=dtype):
            out_dict = self.forward(input_dict, return_hidden_states=True)
        return out_dict["hidden_states"]

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        assert x.dtype == torch.float32
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        # kaldif fbank setting, 400 for 25ms, 160 for 10ms
        x_pad = F.pad(x, ((400 - 160) // 2, (400 - 160) // 2), mode='reflect')
        fbank = self.fbank_fn({"waveform": x_pad.unsqueeze(1) * 32768.0})["fbank"]
        input_dict["fbank"] = fbank
        input_dict["src_mask"] = torch.ones_like(fbank[:, :, 0])
        input_dict = self.cmvn_fn(input_dict)
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        return input_dict


class USMStage3(USMStage2):
    def __init__(self, config):
        super().__init__(config)
        if config.get("vq_type", None) == "CVQ":
            self.vq = ClusteredVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                distance=config.get("vq_distance", "cos"),
            )
        elif config.get("vq_type", None) == "FSQ":
            self.vq = FiniteScalarQuantizer(
                codebook_size=config.vq_codebook_size,
            )
        elif config.get("vq_type", None) == "LFQ":
            self.vq = LookupFreeQuantizer(
                codebook_size=config.vq_codebook_size,
            )
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
                WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1) if config.hidden_size != config.vq_codebook_dim else nn.Identity(),
                nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
                Transpose(),
            )
            self.vq_proj_out = nn.Sequential(
                Transpose(),
                WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1) if config.vq_codebook_dim != config.hidden_size else nn.Identity(),
                Transpose(),
            )
        elif config.get("vq_proj_norm", None) == "ln":
            self.vq_proj_in = nn.Sequential(
                nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False) if config.hidden_size != config.vq_codebook_dim else nn.Identity(),
                nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
            )
            self.vq_proj_out = nn.Sequential(
                nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False) if config.vq_codebook_dim != config.hidden_size else nn.Identity(),
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

    def forward(self, input_dict, return_hidden_states=False, return_vq_ids=False):
        feature = input_dict["fbank"]
        src_mask = input_dict["src_mask"]
        ### one forward ###
        # hidden_states = self.audio_encoder(feature, src_mask, is_training=self.training)
        ### Let's do it step by step ###
        # copy from `mariana.models.audio.usm_encoder.py`
        # copy from `mariana.models.audio.conformer.py`
        front_end_out, backbone_mask, frontend_shape = self.audio_encoder.frontend(feature, src_mask)
        conformers = self.audio_encoder.acoustic_backbone_module
        all_hidden_states = []
        conformer_input = conformers.pos_enc(front_end_out)
        if backbone_mask is None:
            conformer_mask = None
        else:
            conformer_mask = backbone_mask.unsqueeze(1)
        attn_weights = None
        for i, layer in enumerate(conformers.encoders):
            if i == self.config.vq_layer_idx:
                with torch.cuda.amp.autocast(enabled=False):
                    vq_inputs = conformer_input[0] if isinstance(conformer_input, (list, tuple)) else conformer_input
                    vq_inputs = self.vq_proj_in(vq_inputs)
                    if self.config.get("vq_proj_noise", 0) > 0:
                        noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                            0
                        ) / self.config.vq_proj_noise
                        vq_inputs = (
                            vq_inputs + torch.randn_like(vq_inputs) * noise_scale
                        )
                        self.cnt.add_(1)
                    if self.config.get("vq_type", None) == "FSQ":
                        vq_embs, vq_ids = self.vq(vq_inputs)
                        vq_loss = None
                    elif self.config.get("vq_type", None) == "EMAEntropy":
                        vq_embs, vq_ids, vq_loss = self.vq(vq_inputs, e_scale=1.0 if self.cnt < 30_000 else 0.0)
                    else:
                        vq_embs, vq_ids, vq_loss = self.vq(vq_inputs)
                    if return_vq_ids:
                        return vq_ids
                    vq_inputs = self.vq_proj_out(vq_embs)
                    conformer_input = (vq_inputs,) + conformer_input[1:] if isinstance(conformer_input, (list, tuple)) else vq_inputs
                    conformer_input, conformer_mask = layer(
                        [conformer_input, conformer_mask], is_training=self.training
                    )
            else:
                conformer_input, conformer_mask = layer(
                    [conformer_input, conformer_mask], is_training=self.training
                )
            if return_hidden_states and not conformers.normalize_before:
                all_hidden_states.append(conformer_input)
        if isinstance(conformer_input, (list, tuple)):
            conformer_input = conformer_input[0]
        if conformers.normalize_before:
            conformer_input = conformers.after_norm(conformer_input)
            if return_hidden_states:
                all_hidden_states.append(conformer_input)
        # return conformer_input
        hidden_states = conformer_input
        with torch.cuda.amp.autocast(enabled=False):
            mel_out = self.mel_head(hidden_states)
            ctc_out = self.ctc_head(hidden_states)
            output_dict = {
                "mel_out": mel_out,
                "ctc_out": ctc_out,
                "vq_ids": vq_ids,
                "vq_loss": vq_loss,
            }
            if self.config.get("vq_proj_noise", False):
                output_dict.update(noise_scale=noise_scale)
            if self.config.add_chroma:
                chroma_out = self.chroma_head(hidden_states)
                output_dict.update(chroma_out=chroma_out)
        if return_hidden_states:
            all_hidden_states = [h[0] if isinstance(h, (list, tuple)) else h for h in all_hidden_states]
            output_dict["hidden_states"] = all_hidden_states
        return output_dict

    @torch.no_grad()
    def extrature_features(self, wavs, dtype=torch.float32):
        if dtype in [torch.float16, torch.bfloat16]:
            is_amp = True
        input_dict = self.preprocessing(wavs)
        with torch.cuda.amp.autocast(enabled=is_amp, dtype=dtype):
            out_dict = self.forward(input_dict, return_hidden_states=True)
        return out_dict["hidden_states"]

    @torch.no_grad()
    def wav2token(self, wav):
        if dtype in [torch.float16, torch.bfloat16]:
            is_amp = True
        input_dict = self.preprocessing(wavs)
        with torch.cuda.amp.autocast(enabled=is_amp, dtype=dtype):
            vq_ids = self.forward(input_dict, return_vq_ids=True)
        return vq_ids
