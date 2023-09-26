import math
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from recipes.umm.models.vocoder import BigVGAN
from recipes.umm.models.vq import EMAVectorQuantizer
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.transforms.speech import SpeechTransform


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


class EMAEmbedding(nn.Module):
    def __init__(self, codebook_size, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps

        weight = torch.randn(codebook_size, codebook_dim, dtype=torch.float32)
        weight[0] = 0.0
        self.register_buffer("weight", weight)
        self.register_buffer("cluster_size", torch.zeros(codebook_size) + 8)
        self.register_buffer("embed_avg", weight.clone())
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)
        # make sure the greedy algorithm to get best estimation
        self.weight.data[0] = 0.0
        self.embed_avg[0] = 0.0

    @torch.no_grad()
    def entropy(self):
        p = self.cluster_size / self.cluster_size.sum()
        entropy = (-p * p.log()).sum()
        return entropy


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
            self.codebook_size, self.codebook_dim, decay=decay
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


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cached(max_position_embeddings)

    def _set_cos_sin_cached(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cached(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings,
        is_cross_attention=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings
        )
        self.is_cross_attention = is_cross_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, nh, t, hd]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=False
        ):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states.float(),
                key_states.float(),
                value_states.float(),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            ).to(query_states.dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class LlamaCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_position_embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        kv_len = encoder_hidden_states.size(1)
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(encoder_hidden_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(encoder_hidden_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states.float(),
                key_states.float(),
                value_states.float(),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).to(query_states.dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.ar_hidden_size,
            num_attention_heads=config.ar_num_attention_heads,
            max_position_embeddings=config.ar_max_position_embeddings,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.ar_hidden_size,
            intermediate_size=config.ar_intermediate_size,
            hidden_act=config.ar_hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(
            config.ar_hidden_size, eps=config.ar_rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.ar_hidden_size, eps=config.ar_rms_norm_eps
        )

        if config.ar_add_cross_attention:
            self.cross_attn = LlamaCrossAttention(
                hidden_size=config.ar_hidden_size,
                num_attention_heads=config.ar_num_attention_heads,
                max_position_embeddings=config.ar_max_position_embeddings,
            )
            self.cross_attention_layernorm = nn.LayerNorm(
                config.ar_hidden_size, eps=config.ar_rms_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states, position_ids=position_ids
        )
        hidden_states = residual + hidden_states

        # Cross Attention
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layernorm(hidden_states)
            cross_attn_outputs = self.cross_attn(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )
            # residual connection
            hidden_states = residual + cross_attn_outputs

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


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

        encoded_masked_feature = self.audio_encoder(masked_feature)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)
        position_embeddings = self.embed_positions(hidden_states)
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
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
        feature = input_dict["mel"]
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
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        return input_dict


class Stage3(Stage2):
    def __init__(self, config):
        super().__init__(config)
        self.vq_proj_in = nn.Linear(
            config.hidden_size, config.vq_codebook_dim, bias=False
        )
        self.vq = EMAVectorQuantizer(
            codebook_size=config.vq_codebook_size,
            codebook_dim=config.vq_codebook_dim,
            decay=config.vq_decay,
        )
        self.vq_proj_out = nn.Linear(
            config.vq_codebook_dim, config.hidden_size, bias=False
        )

    def forward(self, input_dict):
        feature = input_dict["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                hidden_states = self.vq_proj_out(vq_embs)
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        mel_out = self.mel_head(hidden_states)
        ctc_out = self.ctc_head(hidden_states)
        output_dict = {
            "mel_out": mel_out,
            "ctc_out": ctc_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
        }
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
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


class Stage3AR(Stage3):
    def __init__(self, config):
        super().__init__(config)
        self.ar_embedding = nn.Embedding(1025, config.ar_hidden_size)
        self.ar_layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.ar_num_layers)]
        )
        self.vq_proj_ar = nn.Linear(config.vq_codebook_dim, config.ar_hidden_size)
        self.ar_head = nn.Linear(config.ar_hidden_size, 1024)

    def forward(self, input_dict):
        feature = input_dict["mel"]
        ar_ids = input_dict["ar_ids"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                hidden_states = self.vq_proj_out(vq_embs)
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        mel_out = self.mel_head(hidden_states)
        ctc_out = self.ctc_head(hidden_states)

        # AR
        ar_inputs_ids = torch.cat(
            [
                torch.zeros(
                    size=[ar_ids.size(0), 1], dtype=ar_ids.dtype, device=ar_ids.device
                )
                + 1024,
                ar_ids[:, :-1],
            ],
            dim=1,
        )
        ar_hidden_states = self.ar_embedding(ar_inputs_ids)
        seq_length = max(ar_inputs_ids.size(1), vq_embs.size(1))
        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=ar_ids.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        ar_encoder_embeds = self.vq_proj_ar(vq_embs)
        for layer in self.ar_layers:
            ar_hidden_states = layer(ar_hidden_states, ar_encoder_embeds, position_ids)
        ar_out = self.ar_head(ar_hidden_states)

        output_dict = {
            "mel_out": mel_out,
            "ctc_out": ctc_out,
            "ar_out": ar_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
        }
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        return output_dict


class ASR(Base):
    def __init__(self, config):
        super().__init__(config)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_dict):
        feature = input_dict["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        for layer in self.encoder_pre_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        for layer in self.encoder_post_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        ctc_out = self.ctc_head(hidden_states)
        output_dict = {"ctc_out": ctc_out}
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        return input_dict


class MKII(nn.Module):
    def __init__(self, config):
        super().__init__()

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

        self.chroma_transform = ChromaSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_chroma=config.n_chroma,
            normalized=False,
        )

        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)
        self.encoder_layers = nn.ModuleList(
            [ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.syllable_pre_layers = nn.ModuleList(
            [
                ConformerEncoderLayer(config)
                for _ in range(config.num_syllable_pre_layers)
            ]
        )
        self.chroma_pre_layers = nn.ModuleList(
            [ConformerEncoderLayer(config) for _ in range(config.num_chroma_pre_layers)]
        )
        self.syllable_post_layers = nn.ModuleList(
            [
                ConformerEncoderLayer(config)
                for _ in range(config.num_syllable_post_layers)
            ]
        )
        self.chroma_post_layers = nn.ModuleList(
            [
                ConformerEncoderLayer(config)
                for _ in range(config.num_chroma_post_layers)
            ]
        )
        self.chroma_head = Conv2dUpsampling(config.hidden_size, config.n_chroma)
        self.mel_head = Conv2dUpsampling(config.hidden_size, config.n_mels)
        self.syllable_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        if config.add_vq:
            self.syllable_vq_proj_in = nn.Linear(
                config.hidden_size, config.vq_codebook_dim, bias=False
            )
            self.syllable_vq = EMAVectorQuantizer(
                codebook_size=config.vq_syllable_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
            self.syllable_vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )
            self.chroma_vq_proj_in = nn.Linear(
                config.hidden_size, config.vq_codebook_dim, bias=False
            )
            self.chroma_vq = EMAVectorQuantizer(
                codebook_size=config.vq_chroma_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
            self.chroma_vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )

        self.config = config

    def forward(self, input_dict):
        feature = input_dict["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        chroma_hidden_states = hidden_states
        for layer in self.chroma_pre_layers:
            layer_outputs = layer(
                chroma_hidden_states, position_embeddings=position_embeddings
            )
            chroma_hidden_states = layer_outputs
        if self.config.add_vq:
            chroma_vq_input = self.chroma_vq_proj_in(chroma_hidden_states)
            chroma_vq_embeds, chroma_vq_indices, chroma_vq_loss = self.chroma_vq(
                chroma_vq_input
            )
            chroma_hidden_states = self.chroma_vq_proj_out(chroma_vq_embeds)
        for layer in self.chroma_post_layers:
            layer_outputs = layer(
                chroma_hidden_states, position_embeddings=position_embeddings
            )
            chroma_hidden_states = layer_outputs

        syllable_hidden_states = hidden_states
        for layer in self.syllable_pre_layers:
            layer_outputs = layer(
                syllable_hidden_states, position_embeddings=position_embeddings
            )
            syllable_hidden_states = layer_outputs
        if self.config.add_vq:
            syllable_vq_input = self.syllable_vq_proj_in(syllable_hidden_states)
            if self.config.use_rvq:
                syllable_vq_input = syllable_vq_input - chroma_vq_embeds.detach()
            (
                syllable_vq_embeds,
                syllable_vq_indices,
                syllable_vq_loss,
            ) = self.syllable_vq(syllable_vq_input)
            syllable_hidden_states = self.syllable_vq_proj_out(syllable_vq_embeds)
        for layer in self.syllable_post_layers:
            layer_outputs = layer(
                syllable_hidden_states, position_embeddings=position_embeddings
            )
            syllable_hidden_states = layer_outputs

        syllable_out = self.syllable_head(syllable_hidden_states)
        mel_out = self.mel_head(chroma_hidden_states)
        chroma_out = self.chroma_head(chroma_hidden_states)
        output_dict = {
            "chroma_out": chroma_out,
            "mel_out": mel_out,
            "syllable_out": syllable_out,
        }
        if self.config.add_vq:
            output_dict.update(
                {
                    "chroma_vq_indices": chroma_vq_indices,
                    "chroma_vq_loss": chroma_vq_loss,
                    "syllable_vq_indices": syllable_vq_indices,
                    "syllable_vq_loss": syllable_vq_loss,
                }
            )
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        if x.size(-1) % (self.config.hop_length * 4) > 0:
            return F.pad(
                x,
                (
                    0,
                    self.config.hop_length * 4
                    - (x.size(-1) % (self.config.hop_length * 4)),
                ),
                "constant",
                0,
            )
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
        chroma = F.normalize(chroma, p=2, dim=-1)
        return {"mel": mel, "chroma": chroma}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def tokenize(self, x):
        feature = self.preprocessing(x)["mel"]

        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        chroma_hidden_states = hidden_states
        for layer in self.chroma_pre_layers:
            layer_outputs = layer(
                chroma_hidden_states, position_embeddings=position_embeddings
            )
            chroma_hidden_states = layer_outputs
        chroma_vq_input = self.chroma_vq_proj_in(chroma_hidden_states)
        chroma_vq_embeds, chroma_vq_indices, chroma_vq_loss = self.chroma_vq(
            chroma_vq_input
        )

        syllable_hidden_states = hidden_states
        for layer in self.syllable_pre_layers:
            layer_outputs = layer(
                syllable_hidden_states, position_embeddings=position_embeddings
            )
            syllable_hidden_states = layer_outputs
        syllable_vq_input = self.syllable_vq_proj_in(syllable_hidden_states)
        if self.config.use_rvq:
            syllable_vq_input = syllable_vq_input - chroma_vq_embeds.detach()
        syllable_vq_embeds, syllable_vq_indices, syllable_vq_loss = self.syllable_vq(
            syllable_vq_input
        )

        return torch.stack([syllable_vq_embeds, chroma_vq_embeds], dim=2), torch.stack(
            [syllable_vq_indices, chroma_vq_indices], dim=2
        )


class MKIIVocoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)
        self.encoder_layers = nn.ModuleList(
            [ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.vq_proj_in = nn.Linear(
            config.hidden_size, config.vq_codebook_dim, bias=False
        )
        self.syllable_vq = EMAVectorQuantizer(
            codebook_size=config.vq_syllable_codebook_size,
            codebook_dim=config.vq_codebook_dim,
        )
        self.chroma_vq = EMAVectorQuantizer(
            codebook_size=config.vq_chroma_codebook_size,
            codebook_dim=config.vq_codebook_dim,
        )
        self.vocoder_vq_proj_out = nn.Linear(
            config.vq_codebook_dim, config.hidden_size, bias=False
        )
        self.syllable_vq_proj_out = nn.Linear(
            config.vq_codebook_dim, config.hidden_size, bias=False
        )
        self.chroma_vq_proj_out = nn.Linear(
            config.vq_codebook_dim, config.hidden_size, bias=False
        )
        self.syllable_post_layers = nn.ModuleList(
            [
                ConformerEncoderLayer(config)
                for _ in range(config.num_syllable_post_layers)
            ]
        )
        self.chroma_post_layers = nn.ModuleList(
            [
                ConformerEncoderLayer(config)
                for _ in range(config.num_chroma_post_layers)
            ]
        )
        # self.syllable_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.chroma_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.vocoder_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.chroma_head = Conv1dUpsampling(config.hidden_size, config.num_channels)
        self.syllable_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.vocoder = BigVGAN(config)
        self.downsample_rate = reduce(lambda x, y: x * y, config.downsample_rates)
        self.frame_rate = int(config.sample_rate / self.downsample_rate)
        self.sample_rate = config.sample_rate
        self.vocoder_sec = config.vocoder_sec

    def forward(self, wav):
        b, _, t = wav.size()
        audio_feature = self.audio_encoder(wav)

        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        vq_input = self.vq_proj_in(hidden_states)
        syllable_vq_embeds, syllable_vq_indices, syllable_vq_loss = self.syllable_vq(
            vq_input
        )
        syllable_hidden_states = self.syllable_vq_proj_out(syllable_vq_embeds)
        for layer in self.syllable_post_layers:
            syllable_hidden_states = layer(
                syllable_hidden_states, position_embeddings=position_embeddings
            )
        # syllable_hidden_states = self.syllable_layer_norm(syllable_hidden_states)

        vq_input = vq_input - syllable_vq_embeds.detach()
        chroma_vq_embeds, chroma_vq_indices, chroma_vq_loss = self.chroma_vq(vq_input)
        chroma_hidden_states = self.chroma_vq_proj_out(chroma_vq_embeds)
        for layer in self.chroma_post_layers:
            chroma_hidden_states = layer(
                chroma_hidden_states, position_embeddings=position_embeddings
            )
        # chroma_hidden_states = self.chroma_layer_norm(chroma_hidden_states)

        start_indices = F.one_hot(
            torch.randint(
                low=0, high=t // (self.sample_rate * self.vocoder_sec), size=(b,)
            ),
            num_classes=audio_feature.size(1),
        )

        time_domain_indices = torch.nonzero(
            start_indices.repeat_interleave(self.sample_rate * self.vocoder_sec, dim=1)
        )
        token_domain_indices = torch.nonzero(
            start_indices.repeat_interleave(self.frame_rate * self.vocoder_sec, dim=1)
        )
        # vocoder_taget = wav.squeeze(1)[tuple(time_domain_indices.t())].unsqueeze(1)
        vocoder_taget = wav.squeeze(1)[time_domain_indices].unsqueeze(1)
        vocoder_input = (syllable_vq_embeds + chroma_vq_embeds)[
            tuple(token_domain_indices.t())
        ]

        vocoder_input = self.vocoder_vq_proj_out(vocoder_input)
        # vocoder_input = self.vocoder_layer_norm(vocoder_input)
        vocoder_out = self.vocoder(vocoder_input)

        syllable_out = self.syllable_head(syllable_hidden_states)
        chroma_out = self.chroma_head(chroma_hidden_states)

        output_dict = {
            "chroma_out": chroma_out,
            "chroma_vq_indices": chroma_vq_indices,
            "chroma_vq_loss": chroma_vq_loss,
            "syllable_out": syllable_out,
            "syllable_vq_indices": syllable_vq_indices,
            "syllable_vq_loss": syllable_vq_loss,
            "vocoder_out": vocoder_out,
            "vocoder_taget": vocoder_taget,
        }

        return output_dict


class MaskingScheme(nn.Module):
    def __init__(self, uniform_sample: bool, mask_id: int):
        super().__init__()
        self.uniform_sample = uniform_sample
        self.mask_id = mask_id

    def forward(
        self, acoustic_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, q = acoustic_tokens.size()
        device = acoustic_tokens.device
        rand_ts = torch.randint(0, t - 1, (b,), device=device)
        if self.uniform_sample:
            rand_qs = self.sample_qs(b, q, device)
        else:
            rand_qs = (
                (1 - self.cosine_schedule(torch.empty(b, device=device).uniform_(0, 1)))
                * q
            ).long()

        rand_times = torch.empty(b, 1, device=device).uniform_(0, 1)
        rand_probs = self.cosine_schedule(rand_times)  # [b, ]
        num_tokens_mask = (rand_probs * t).clamp(min=1.0).long()

        mask = torch.full_like(acoustic_tokens, False, dtype=torch.bool)
        for i, sampled_q in enumerate(rand_qs):
            rand_indices_no_replacement = torch.randperm(t, device=device)[
                : num_tokens_mask[i]
            ]
            sampled_t = rand_ts[i]
            rand_indices_no_replacement = rand_indices_no_replacement[
                rand_indices_no_replacement >= sampled_t
            ]
            mask[i, rand_indices_no_replacement, sampled_q] = True
            mask[i, sampled_t:, sampled_q + 1 :] = True
        masked_audio_tokens = torch.where(mask, self.mask_id, acoustic_tokens)
        return masked_audio_tokens, rand_qs

    def cosine_schedule(self, ratio: torch.Tensor) -> torch.Tensor:
        return torch.cos(ratio * math.pi / 2.0)

    def sample_qs(self, batch_size: int, n_quantizers: int, device):
        n_randperms = math.ceil(batch_size / n_quantizers)
        qs = []
        for _ in range(n_randperms):
            qs.append(torch.randperm(n_quantizers, device=device))
        return torch.cat(qs)[:batch_size]


class SoundStorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.acoustic_embedding = nn.ModuleList(
            [
                nn.Embedding(config.ss_codebook_size + 1, config.hidden_size)
                for _ in range(config.ss_num_quantizers)
            ]
        )
        self.semantic_embedding = nn.Embedding(
            config.vq_syllable_codebook_size + config.vq_chroma_codebook_size,
            config.hidden_size,
        )
        self.conformer = ConformerEncoder(config)
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.ss_codebook_size, bias=False)
                for _ in range(config.ss_num_quantizers)
            ]
        )
        self.masking_scheme = MaskingScheme(
            uniform_sample=config.uniform_sample, mask_id=config.ss_codebook_size
        )
        self.mha = MHA(config)
        self.config = config

    def decode(
        self,
        acoustic_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
        selected_qs: torch.Tensor,
    ):
        acoustic_embeds = []
        for q in range(self.config.ss_num_quantizers):
            acoustic_embeds.append(self.acoustic_embedding[q](acoustic_tokens[:, :, q]))
        acoustic_embeds = torch.stack(acoustic_embeds, dim=2)  # [B, T, Q, D]
        semantic_embeds = self.semantic_embedding(semantic_tokens)  # [B, T, 1, D]
        b, t, q, d = acoustic_embeds.size()
        for i in range(acoustic_embeds.size(0)):
            acoustic_embeds[i, :, selected_qs[i] + 1 :] = 0
        query = [
            acoustic_embeds[i, :, selected_qs[i] : selected_qs[i] + 1] for i in range(b)
        ]
        query = torch.stack(query, dim=0)
        query = rearrange(query, "b t q d -> (b t) q d")  # (B * T, 1, D)
        key = torch.cat((semantic_embeds, acoustic_embeds), dim=2)
        key = rearrange(key, "b t q d -> (b t) q d")  # (B, T, 1 + Q, D)
        value = key
        conformer_input = self.mha(q=query, k=key, v=value)  # (B * T, 1, D)
        conformer_input = rearrange(conformer_input, "(b t) 1 d -> b t d", b=b)
        last_hidden_state = self.conformer(conformer_input)["last_hidden_state"]
        logits = [self.heads[selected_qs[i]](last_hidden_state[i]) for i in range(b)]
        logits = torch.stack(logits, dim=0)
        return logits

    def forward(
        self, acoustic_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> torch.Tensor:
        masked_acoustic_tokens, rand_qs = self.masking_scheme(acoustic_tokens)
        logits = self.decode(masked_acoustic_tokens, semantic_tokens, rand_qs)
        return {
            "logits": logits,
            "masked_acoustic_tokens": masked_acoustic_tokens,
            "rand_qs": rand_qs,
        }

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        if x.size(-1) % (self.config.hop_length * 4) > 0:
            return F.pad(
                x,
                (
                    0,
                    self.config.hop_length * 4
                    - (x.size(-1) % (self.config.hop_length * 4)),
                ),
                "constant",
                0,
            )
        else:
            return x

    @torch.no_grad()
    def sample(
        self,
        semantic_tokens: torch.Tensor,
        num_iterations: List[int],
        score_strategies: List[str],
        temperatures: Optional[List[float]] = None,
    ) -> torch.Tensor:
        b, t, _ = semantic_tokens.size()
        device = semantic_tokens.device
        acoustic_tokens = torch.full(
            (b, t, self.config.ss_num_quantizers),
            self.masking_scheme.mask_id,
            dtype=torch.long,
            device=semantic_tokens.device,
        )

        if temperatures is None:
            temperatures = [1.0] * self.config.ss_num_quantizers

        for q in tqdm(range(self.config.ss_num_quantizers), desc="[SoundStorm]"):
            q_num_iteration = num_iterations[q]
            q_score_strategy = score_strategies[q]
            q_temperature = temperatures[q]
            ratios = torch.linspace(0, 1.0, q_num_iteration + 1)[1:]
            cos_ratios = self.masking_scheme.cosine_schedule(ratios)
            selected_qs = torch.LongTensor([q] * b).to(device)
            for step_idx, ratio in enumerate(cos_ratios):
                logits = self.decode(acoustic_tokens, semantic_tokens, selected_qs)
                probs = logits.softmax(dim=-1)
                masked_positions = (
                    acoustic_tokens[:, :, q] == self.masking_scheme.mask_id
                )

                if masked_positions.sum() == 0:
                    continue

                if step_idx == q_num_iteration - 1:
                    # greedy decoding for the last iteration
                    sampled_tokens = probs.argmax(dim=-1)
                    acoustic_tokens[:, :, q] = torch.where(
                        masked_positions, sampled_tokens, acoustic_tokens[:, :, q]
                    )
                else:
                    # sample candidates first
                    probs_scaled = (logits / q_temperature).softmax(dim=-1)
                    sampled_tokens = torch.distributions.categorical.Categorical(
                        probs_scaled
                    ).sample()

                    # gather the probabilities of each of the candidates
                    if q_score_strategy == "maskgit":
                        scores = probs.gather(
                            2, rearrange(sampled_tokens, "b n -> b n 1")
                        )
                        scores = rearrange(scores, "b n 1 -> b n")
                    elif q_score_strategy == "max_prob":
                        scores, _ = probs.max(dim=-1)
                    elif q_score_strategy == "max_entropy":
                        scores = torch.distributions.categorical.Categorical(
                            probs
                        ).entropy()
                    elif q_score_strategy == "min_entropy":
                        scores = (
                            torch.distributions.categorical.Categorical(probs).entropy()
                            * -1
                        )
                    elif q_score_strategy == "random":
                        scores = torch.rand(probs.shape[:2]).to(probs.device)
                    elif q_score_strategy == "sequential":
                        m, n = probs.shape[:2]
                        scores = torch.arange(n).repeat(m, 1).to(probs.device) * -1.0
                    else:
                        raise ValueError(f"Unknown score strategy: {q_score_strategy}")

                    # keep only the top k scores
                    # we assume an equal unmasking schedule, so we simply take the number
                    # of masked positions of the first batch element as our reference
                    tokens_left = masked_positions[0].sum()
                    tokens_unmasked = masked_positions.shape[-1] - tokens_left
                    topk_tokens = ((1 - ratio) * masked_positions.shape[-1]).long()
                    topk_tokens = topk_tokens - tokens_unmasked

                    # always select at least 1
                    topk_tokens = max(topk_tokens, 1)
                    # select the topk, otherwise the remainder
                    topk_tokens = min(topk_tokens, tokens_left)

                    # don't select topk of previously sampled tokens
                    scores = torch.where(
                        masked_positions, scores, -torch.finfo(scores.dtype).max
                    )
                    # batched topk
                    topk_probs, topk_indices = scores.topk(topk_tokens, dim=-1)

                    # create a mask that is True for all scores that meet the confidence criterium
                    confidence_mask = torch.zeros_like(
                        scores, dtype=torch.bool
                    ).scatter(dim=-1, index=topk_indices, value=True)

                    # only fill positions that are currently masked and meet the confidence criterium
                    # otherwise fill with original token
                    fill_positions = masked_positions & confidence_mask
                    acoustic_tokens[:, :, q] = torch.where(
                        fill_positions, sampled_tokens, acoustic_tokens[:, :, q]
                    )
        return acoustic_tokens
