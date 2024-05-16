import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from recipes.umm_062.models.vocoder import BigVGAN
from recipes.umm_062.models.vq import EMAVectorQuantizer, VectorQuantize
from recipes.umm_062.transforms.chroma import ChromaSpectrogram
from recipes.umm_062.transforms.speech import SpeechTransform
from samantha.models.flash_llama import LlamaForCausalLM


@dataclass
class ConformerEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vq_states: Optional[torch.FloatTensor] = None
    vq_ids: Optional[torch.IntTensor] = None
    vq_loss: Optional[torch.FloatTensor] = None
    vq_emb: Optional[torch.FloatTensor] = None


class ConformerRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
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

    def forward(self, hidden_states, vq=None, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        hidden_states = self.dropout(hidden_states)
        position_embeddings = self.embed_positions(hidden_states)
        vq_states = None
        vq_ids = None
        vq_loss = None
        vq_emb = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if vq is not None and self.config.vq_layer_idx == i:
                vq_states, vq_ids, vq_loss, vq_emb = vq(hidden_states)
                hidden_states = vq_states

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
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            vq_states=vq_states,
            vq_ids=vq_ids,
            vq_loss=vq_loss,
            vq_emb=vq_emb,
        )

    def forward_from_vq(self, hidden_states):
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.layers):

            if self.config.vq_layer_idx < i:
                continue

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

        return ConformerEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            vq_states=None,
            vq_ids=None,
            vq_loss=None,
        )

    def forward_to_vq(self, hidden_states, vq):
        hidden_states = self.dropout(hidden_states)
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.layers):

            if self.config.vq_layer_idx == i:
                vq_states, vq_ids, vq_loss, vq_emb = vq(hidden_states)
                hidden_states = vq_states
                return ConformerEncoderOutput(
                    last_hidden_state=None,
                    hidden_states=None,
                    vq_states=vq_states,
                    vq_ids=vq_ids,
                    vq_loss=vq_loss,
                    vq_emb=vq_emb,
                )

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

        raise ValueError("VQ layer index is out of range")


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
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel, 2, padding),
            nn.ReLU(),
        )
        self.linear = nn.Linear(input_dim * 64, output_dim)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["feature"]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x


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


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.shared_encoder = ConformerEncoder(config)
        if config.rq_input_layernorm:
            self.rq_input_layernorm = nn.LayerNorm(
                config.num_channels
                * pow(config.feature_encoder_kernel, config.feature_encoder_padding),
                elementwise_affine=False,
            )
        self.rq_head = nn.Linear(
            config.hidden_size,
            config.rq_codebook_size * config.rq_codebook_num,
            bias=False,
        )
        self.audio_transform = SpeechTransform(
            sample_rate=config.sample_rate,
            n_mels=config.num_channels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=0,
            f_max=config.sample_rate // 2,
        )
        if config.feature_cmvn is not None:
            self.audio_transform.load_from_checkpoint(config.feature_cmvn)

        self.unfolder = nn.Unfold(
            kernel_size=(config.feature_encoder_kernel, 1),
            dilation=1,
            padding=(config.feature_encoder_padding, 0),
            stride=(2, 1),
        )
        self.rq = RandomProjectionQuantizer(config)

        if config.add_vq:
            if config.use_ema_vq:
                self.vq = EMAVectorQuantizer(
                    dim=config.hidden_size,
                    codebook_size=config.vq_codebook_size,
                    codebook_dim=config.vq_codebook_dim,
                )
            else:
                self.vq = VectorQuantize(
                    dim=config.hidden_size,
                    codebook_size=config.vq_codebook_size,
                    codebook_dim=config.vq_codebook_dim,
                    threshold_ema_dead_code=config.vq_threshold_ema_dead_code,
                    kmeans_init=config.vq_kmeans_init,
                    kmeans_iters=config.vq_kmeans_iters,
                    sync_codebook=config.vq_sync_codebook,
                )
        self.config = config

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

    def forward(self, input_dict):
        raise NotImplementedError()

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
        return self.audio_transform(x, normalize=normalize)

    def get_latent(self, x, layer_idx: Optional[int] = None):
        x = self.audio_encoder(x)
        if layer_idx is None:
            emb = self.shared_encoder(x, vq=self.vq)["last_hidden_state"]
            return emb
        else:
            emb = self.shared_encoder(x, output_hidden_states=True, vq=self.vq)[
                "hidden_states"
            ]
            return emb[layer_idx]


class PreTrainedModel(BaseModel):
    def __init__(self, config):
        super().__init__(config=config)

    def forward(self, input_dict):
        masked_feature = input_dict["masked_feature"]
        masked_indices = input_dict["masked_indices"]
        feature = input_dict["feature"]
        encoded_masked_feature = self.audio_encoder(masked_feature)
        vq = self.vq if self.config.add_vq else None
        shared_encoder_output = self.shared_encoder(encoded_masked_feature, vq=vq)
        hidden_state = shared_encoder_output["last_hidden_state"]

        logits = self.rq_head(hidden_state)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.config.rq_codebook_num
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.config.rq_codebook_num
        )
        target = self.get_rq_target(feature)
        masked_target = target[tuple(masked_indices.t())]
        masked_target = rearrange(masked_target, "b c -> (b c)")
        output_dict = {
            "rq_logits": logits,
            "rq_masked_logits": masked_logits,
            "rq_target": target,
            "rq_masked_target": masked_target,
        }
        if self.config.add_vq:
            vq_states = shared_encoder_output["vq_states"]
            vq_ids = shared_encoder_output["vq_ids"]
            vq_loss = shared_encoder_output["vq_loss"]
            output_dict.update(vq_states=vq_states, vq_ids=vq_ids, vq_loss=vq_loss)
        return output_dict


class FineTunedModel(BaseModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.melrecon_head = Conv2dUpsampling(
            self.config.hidden_size, self.config.num_channels
        )
        self.ctc_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        if config.add_mulan:
            self.mulan_head = nn.Linear(
                self.config.hidden_size, self.config.mulan_hidden_size, bias=False
            )
        if config.add_vocoder:
            self.vocoder = BigVGAN(config)
        if config.get("add_chroma", False):
            self.chroma_transform = ChromaSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                n_chroma=config.n_chroma,
                normalized=False,
            )
            self.chromarecon_head = Conv2dUpsampling(
                self.config.hidden_size, config.n_chroma
            )
        del self.unfolder
        del self.rq
        del self.rq_head

    def forward(self, input_dict):
        feature = input_dict["mel"]
        encoded_feature = self.audio_encoder(feature)
        vq = self.vq if self.config.add_vq else None
        shared_encoder_output = self.shared_encoder(encoded_feature, vq=vq)
        hidden_state = shared_encoder_output["last_hidden_state"]

        logits = self.ctc_head(hidden_state)
        recon_feature = self.melrecon_head(hidden_state)
        if self.config.get("add_chroma", False):
            recon_chroma = self.chromarecon_head(hidden_state)
        if self.config.add_vocoder:
            recon_wav = self.vocoder(hidden_state.transpose(1, 2)).squeeze(1)

        output_dict = {
            "ctc_out": logits,
            "mel_out": recon_feature,
            "chroma_out": recon_chroma
            if self.config.get("add_chroma", False)
            else None,
            "recon_wav": recon_wav if self.config.add_vocoder else None,
        }
        if self.config.add_vq:
            vq_states = shared_encoder_output["vq_states"]
            vq_ids = shared_encoder_output["vq_ids"]
            vq_loss = shared_encoder_output["vq_loss"]
            # TODO: could also try mean first
            if self.config.add_mulan:
                vq_embeds = self.mulan_head(vq_states).mean(dim=1)
                vq_embeds = F.normalize(vq_embeds, p=2, dim=1)
            output_dict.update(
                vq_states=vq_states,
                vq_ids=vq_ids,
                vq_loss=vq_loss,
                vq_embeds=vq_embeds if self.config.add_mulan else None,
            )
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        if self.config.get("add_chroma", False):
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        return input_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        encoded_feature = self.wav2embed(wav)
        shared_encoder_output = self.shared_encoder.forward_to_vq(
            encoded_feature, vq=self.vq
        )
        vq_ids = shared_encoder_output["vq_ids"]
        return vq_ids

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2embed(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        wav = self.pad_audio(wav.float())
        feature = self.preprocessing(wav)["mel"]
        encoded_feature = self.audio_encoder(feature)
        return encoded_feature


class FineTunedVocoder(BaseModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.melrecon_head = Conv2dUpsampling(
            self.config.hidden_size, self.config.num_channels
        )
        if config.add_mulan:
            self.mulan_head = nn.Linear(
                self.config.hidden_size, self.config.mulan_hidden_size, bias=False
            )
        if config.add_vocoder:
            self.vocoder = BigVGAN(config)
        del self.unfolder
        del self.rq
        del self.rq_head

    def forward(self, input_dict):
        feature = input_dict["feature"]
        encoded_feature = self.audio_encoder(feature)
        vq = self.vq if self.config.add_vq else None
        shared_encoder_output = self.shared_encoder(encoded_feature, vq=vq)
        hidden_state = shared_encoder_output["last_hidden_state"]

        recon_feature = self.melrecon_head(hidden_state)
        if self.config.add_vocoder:
            recon_wav = self.vocoder(hidden_state.transpose(1, 2)).squeeze(1)

        output_dict = {
            "recon_feature": recon_feature,
            "recon_wav": recon_wav if self.config.add_vocoder else None,
        }
        if self.config.add_vq:
            vq_states = shared_encoder_output["vq_states"]
            vq_ids = shared_encoder_output["vq_ids"]
            vq_loss = shared_encoder_output["vq_loss"]
            # TODO: could also try mean first
            if self.config.add_mulan:
                vq_embeds = self.mulan_head(vq_states).mean(dim=1)
                vq_embeds = F.normalize(vq_embeds, p=2, dim=1)
            output_dict.update(
                vq_states=vq_states,
                vq_ids=vq_ids,
                vq_loss=vq_loss,
                vq_embeds=vq_embeds if self.config.add_mulan else None,
            )
        return output_dict


class UMMBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.shared_encoder = ConformerEncoder(config)
        self.audio_transform = SpeechTransform(
            sample_rate=config.sample_rate,
            n_mels=config.num_channels,
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
        if config.add_vq:
            self.vq = EMAVectorQuantizer(
                dim=config.hidden_size,
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
            )
        self.mel_head = Conv2dUpsampling(config.hidden_size, config.n_mels)
        if not config.disable_chroma:
            self.chroma_head = Conv2dUpsampling(config.hidden_size, config.n_chroma)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_dict):
        feature = input_dict["mel"]
        encoded_feature = self.audio_encoder(feature)
        shared_encoder_output = self.shared_encoder(
            encoded_feature, vq=self.vq if self.config.add_vq else None
        )
        hidden_state = shared_encoder_output["last_hidden_state"]

        ctc_out = self.ctc_head(hidden_state)
        mel_out = self.mel_head(hidden_state)
        chroma_out = (
            None if self.config.disable_chroma else self.chroma_head(hidden_state)
        )
        output_dict = {
            "ctc_out": ctc_out,
            "mel_out": mel_out,
            "chroma_out": chroma_out,
            "vq_states": shared_encoder_output.get("vq_states"),
            "vq_ids": shared_encoder_output.get("vq_ids"),
            "vq_loss": shared_encoder_output.get("vq_loss"),
        }
        return output_dict

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
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        if not self.config.disable_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
        else:
            chroma = None
        input_dict = {"mel": mel, "chroma": chroma}
        return input_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        wav = self.pad_audio(wav.float())
        feature = self.preprocessing(wav)["mel"]
        encoded_feature = self.audio_encoder(feature)
        shared_encoder_output = self.shared_encoder.forward_to_vq(
            encoded_feature, vq=self.vq
        )
        vq_ids = shared_encoder_output["vq_ids"]
        return vq_ids


class BaseUMM(UMMBase):
    def __init__(self, config):
        super().__init__(config)


class UMMAR(UMMBase):
    def __init__(self, config):
        super().__init__(config)
        self.ar_config = LlamaConfig(
            vocab_size=1025,
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 3,
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_act="silu",
            max_position_embeddings=1500,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=False,
            num_logits=1024,
        )
        self.ar_head = LlamaForCausalLM(self.ar_config)

    def forward(self, input_dict):
        feature = input_dict["mel"]
        ar_ids = input_dict["ar_ids"]
        encoded_feature = self.audio_encoder(feature)
        shared_encoder_output = self.shared_encoder(
            encoded_feature, vq=self.vq if self.config.add_vq else None
        )
        hidden_state = shared_encoder_output["last_hidden_state"]

        ctc_out = self.ctc_head(hidden_state)
        mel_out = self.mel_head(hidden_state)
        chroma_out = self.chroma_head(hidden_state)

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
        ar_inputs_embeds = self.ar_head.model.embed_tokens(ar_inputs_ids)
        ar_inputs_embeds = torch.cat(
            [shared_encoder_output["vq_states"], ar_inputs_embeds], dim=1
        )
        ar_out = self.ar_head(inputs_embeds=ar_inputs_embeds)["logits"][
            :, -ar_ids.size(1) :, :
        ]

        output_dict = {
            "ctc_out": ctc_out,
            "mel_out": mel_out,
            "chroma_out": chroma_out,
            "ar_out": ar_out,
            "vq_states": shared_encoder_output["vq_states"],
            "vq_ids": shared_encoder_output["vq_ids"],
            "vq_loss": shared_encoder_output["vq_loss"],
        }
        return output_dict


class UMMASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.shared_encoder = ConformerEncoder(config)
        self.audio_transform = SpeechTransform(
            sample_rate=config.sample_rate,
            n_mels=config.num_channels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=0,
            f_max=config.sample_rate // 2,
        )
        if config.feature_cmvn is not None:
            self.audio_transform.load_from_checkpoint(config.feature_cmvn)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_dict):
        feature = input_dict["mel"]
        encoded_feature = self.audio_encoder(feature)
        shared_encoder_output = self.shared_encoder(
            encoded_feature, vq=self.vq if self.config.add_vq else None
        )
        hidden_state = shared_encoder_output["last_hidden_state"]
        ctc_out = self.ctc_head(hidden_state)
        output_dict = {
            "ctc_out": ctc_out,
            "vq_states": shared_encoder_output.get("vq_states"),
            "vq_ids": shared_encoder_output.get("vq_ids"),
            "vq_loss": shared_encoder_output.get("vq_loss"),
        }
        return output_dict

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
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        return input_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        wav = self.pad_audio(wav.float())
        feature = self.preprocessing(wav)["mel"]
        encoded_feature = self.audio_encoder(feature)
        shared_encoder_output = self.shared_encoder.forward_to_vq(
            encoded_feature, vq=self.vq
        )
        vq_ids = shared_encoder_output["vq_ids"]
        return vq_ids
