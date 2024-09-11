# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from einops import rearrange, repeat
from transformers import GPT2Config
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from samantha.utils.triton.blocksparse import matmul as sparse_matmul
from samantha.utils.triton.blocksparse import softmax as sparse_softmax

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


#### sparse layout design ####
def create_mask(l_len, location, stride=0, loc_offset=0, unmask_len=0):
    x = torch.arange(l_len).unsqueeze(1)  # [tq, 1]
    y = torch.arange(l_len).unsqueeze(0)  # [1, tk]
    xy = x - y
    # location mask
    location_mask = torch.logical_and(xy.abs() <= location + loc_offset,
                                      xy.abs() >= loc_offset)
    total_mask = torch.logical_or(location_mask, xy.abs() <= 1)

    # stride mask
    if stride > 0:
        stride_mask = xy.abs() % stride == 0
        total_mask = torch.logical_or(total_mask, stride_mask)

    # unmask mask
    if unmask_len > 0:
        total_mask[:, 0:unmask_len] = True

    # causal mask
    causal_mask = xy >= 0
    total_mask = torch.logical_and(causal_mask, total_mask)
    return total_mask  # [tq, tk]


def semantic_10s_layout_fn(train_len=288, num_heads=16, block=32):
    tq = torch.arange(train_len // block).unsqueeze(1)
    tk = torch.arange(train_len // block).unsqueeze(0)
    mask = (tq - tk) >= 0
    mask = mask.unsqueeze(0).repeat(num_heads, 1, 1)
    return mask, block


def semantic_1280_layout_fn(train_len=768, num_heads=16, block=32):
    tq = torch.arange(train_len // block).unsqueeze(1)
    tk = torch.arange(train_len // block).unsqueeze(0)
    mask = (tq - tk) >= 0
    mask = mask.unsqueeze(0).repeat(num_heads, 1, 1)
    return mask, block


def coarse_30s_layout_fn(train_len=5600, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=32,
            stride=0,
            loc_offset=4 * max(0, i + 1 - num_heads // 2),
        ) for i in range(num_heads)
    ]

    for mask in layout:
        print(mask.sum() / (-mask.sum() + (mask + 1).sum()))
    layout = torch.stack(layout, dim=0)
    return layout, block


def fine_10s_layout_fn(train_len=4832, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=28,
            stride=0,
            unmask_len=1600 // block + 1,
            loc_offset=0,
        ) for l in range(num_heads)
    ]

    layout = torch.stack(layout, dim=0)
    return layout, block


def coarse_10s_layout_fn_v3(train_len=2272, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=32,
            stride=0,
            unmask_len=288 // block,
            loc_offset=8 * max(0, l + 1 - num_heads // 2),
        ) for l in range(num_heads)
    ]

    for l in layout:
        print(l.sum() / (-l.sum() + (l + 1).sum()))
    layout = torch.stack(layout, dim=0)
    return layout, block


def coarse_10s_layout_fn_v4(train_len=2784, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=32,
            stride=0,
            unmask_len=768 // block,
            loc_offset=8 * max(0, l + 1 - num_heads // 2),
        ) for l in range(num_heads)
    ]

    for l in layout:
        print(l.sum() / (-l.sum() + (l + 1).sum()))
    layout = torch.stack(layout, dim=0)
    return layout, block


def coarse_10s_layout_fn_v4_3(train_len=2272, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=32,
            stride=0,
            unmask_len=256 // block,
            loc_offset=8 * max(0, l + 1 - num_heads // 2),
        ) for l in range(num_heads)
    ]

    for l in layout:
        print(l.sum() / (-l.sum() + (l + 1).sum()))
    layout = torch.stack(layout, dim=0)
    return layout, block


def coarse_30s_layout_fn_v3(train_len=6784, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=32,
            stride=0,
            unmask_len=768 // block,
            loc_offset=8 * max(0, l + 1 - num_heads // 2),
        ) for l in range(num_heads)
    ]

    for l in layout:
        print(l.sum() / (-l.sum() + (l + 1).sum()))
    layout = torch.stack(layout, dim=0)
    return layout, block


def coarse_30s_layout_fn_v6(train_len=6784, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=64,
            stride=0,
            unmask_len=768 // block,
            loc_offset=0,
        ) for l in range(num_heads)
    ]

    for l in layout:
        print(l.sum() / (-l.sum() + (l + 1).sum()))
    layout = torch.stack(layout, dim=0)
    return layout, block


def fine_10s_layout_fn_v3(train_len=6032, num_heads=16, block=32):
    assert train_len % block == 0
    layout = [
        create_mask(
            l_len=train_len // block,
            location=24,
            stride=0,
            unmask_len=2000 // block + 1,
            loc_offset=4 * max(0, l + 1 - num_heads // 2),
        ) for l in range(num_heads)
    ]

    layout = torch.stack(layout, dim=0)
    return layout, block


layout_fn_dict = {
    "semantic_10s_layout": semantic_10s_layout_fn,
    "semantic_1280_layout": semantic_1280_layout_fn,
    "coarse_30s_layout": coarse_30s_layout_fn,
    "fine_10s_layout": fine_10s_layout_fn,
    "coarse_10s_layout_v3": coarse_10s_layout_fn_v3,
    "coarse_10s_layout_v4": coarse_10s_layout_fn_v4,
    "coarse_10s_layout_fn_v4_3": coarse_10s_layout_fn_v4_3,
    "coarse_30s_layout_v3": coarse_30s_layout_fn_v3,
    "coarse_30s_layout_v6": coarse_30s_layout_fn_v6,
    "fine_10s_layout_v3": fine_10s_layout_fn_v3,
}
#--- sparse layout design ----


def config_get(config, name, default):
    if hasattr(config, name):
        return config.__getattribute__(name)
    else:
        return default


class FixedAbsolutePositionEmbedding(nn.Module):

    def __init__(self, max_pos, hidden_size, pos_type="rope", ffn_fp32=False):
        super().__init__()
        assert pos_type in ["rope", "fixed", "rope_attn"]
        self.pos_type = pos_type
        self.ffn_fp32 = ffn_fp32

        inv_freq = 1. / (10000**(
            torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_pos, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()),
                               dim=-1)
        self.register_buffer('wpe', embeddings)

    def forward_fixed(self, x, position_ids):
        return x.float() + F.embedding(position_ids, self.wpe)

    def forward_rope(self, x, position_ids):
        with autocast(enabled=not self.ffn_fp32):
            embeddings = F.embedding(position_ids, self.wpe)  # [b, t, d]
            embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
            sin, cos = embeddings.unbind(dim=-2)  # b l d//2
            sin = repeat(sin, '... d -> ... (d 2)')
            cos = repeat(cos, '... d -> ... (d 2)')
            x = x.float()
            return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x, position_ids):
        if self.pos_type == 'fixed':
            return self.forward_fixed(x, position_ids)

        elif self.pos_type == 'rope' or self.pos_type == 'rope_attn':
            return self.forward_rope(x, position_ids)

    def forward(self, x, position_ids):
        if x.dim() == 3:
            return self._forward(x, position_ids)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x, position_ids)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # TODO: Explicit full/mixed precision?
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nf = nf
        self.use_bias = bias
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf)) if self.use_bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if self.use_bias:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        else:
            x = torch.matmul(x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class GPT2Attention(nn.Module):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.sparse = config_get(config, "sparse_attn", False)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions),
                           dtype=torch.uint8)).view(1, 1, max_positions,
                                                    max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        qk_bias = config_get(config, "qk_bias", False)
        self.c_q = Conv1D(self.embed_dim, self.embed_dim, bias=qk_bias)
        self.c_k = Conv1D(self.embed_dim, self.embed_dim, bias=qk_bias)
        self.c_v = Conv1D(self.embed_dim, self.embed_dim, bias=False)
        proj_bias = config_get(config, "proj_bias", False)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim, bias=proj_bias)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        # RoPE position embedding
        self.pos_type = config_get(config, "pos_type", "learnable")
        ffn_fp32 = config_get(config, "ffn_fp32", False)
        if self.pos_type == "rope_attn":
            self.wpe = FixedAbsolutePositionEmbedding(max_positions,
                                                      self.head_dim,
                                                      self.pos_type,
                                                      ffn_fp32)
            self.register_buffer("pos_ids",
                                 torch.arange(max_positions).unsqueeze(0))

        if self.sparse:
            layout_fn = layout_fn_dict[config.layout_fn]
            layout, block = layout_fn(train_len=config.train_len,
                                    num_heads=config.n_head)
            self.register_buffer("layout", layout)
            self.block = block
            print("self.layout.shape: ", self.layout.shape)

    def _get_sparse_fn(self, device):
        print("Prepare sparse function...")
        self.qk_matmul = sparse_matmul(
            self.layout,
            self.block,
            mode="sdd",
            trans_a=False,
            trans_b=True,
            device=device,
        )
        self.softmax_fn = sparse_softmax(self.layout,
                                         self.block,
                                         device,
                                         is_dense=False)
        self.wv_matmul = sparse_matmul(
            self.layout,
            self.block,
            mode="dsd",
            trans_a=False,
            trans_b=False,
            device=device,
        )

    def _sparse_attn(self,
                     query,
                     key,
                     value,
                     attention_mask=None,
                     head_mask=None):
        assert (attention_mask is None) and (head_mask is None), \
            "Block sparse dont support attention_mask and head_mask"
        if not hasattr(self, "qk_matmul"):
            self._get_sparse_fn(device=query.device)

        scale = 1.0
        if self.scale_attn_weights:
            scale = scale / (value.size(-1)**0.5)

        attn_weights = self.qk_matmul(query, key)
        attn_weights = self.softmax_fn(attn_weights,
                                       scale=scale,
                                       is_causal=True)
        if self.config.attn_pdrop > 0:
            attn_weights = self.attn_dropout(attn_weights)  # TODO
        attn_output = self.wv_matmul(attn_weights, value)  # [b, head, t_q, d]
        return attn_output, attn_weights

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if self.training and self.sparse:
            dtype = value.dtype
            return self._sparse_attn(query.to(dtype),
                                     key.to(dtype),
                                     value,
                                     attention_mask=None,
                                     head_mask=None)
        else:
            if self.sparse:
                # prepare inference mask
                mask_file = config_get(self.config, "mask_file",
                                       "full_mask.npy")
                if not hasattr(self, "full_mask"):

                    def dense_mask(layout, block):
                        print("Getting full mask...")
                        # create dense mask same as layout
                        h, t, d = layout.shape
                        mask = torch.zeros([h, t * block, d * block],
                                           device=layout.device)
                        # TODO: it is inefficent for GPU
                        for j, h_ in enumerate(layout):
                            for k, t_ in enumerate(h_):
                                for l, d_ in enumerate(t_):
                                    mask[j, k * block:(k + 1) * block,
                                         l * block:(l + 1) *
                                         block, ] = d_  # type: ignore
                        return mask.to(self.bias.dtype)

                    if os.path.exists(mask_file):
                        full_mask = np.load(mask_file)
                        self.full_mask = torch.from_numpy(full_mask).to(
                            query.device)
                    else:
                        self.full_mask = dense_mask(
                            self.layout, self.block).to(query.device)
                        np.save(mask_file,
                                self.full_mask.detach().cpu().numpy())
            # [b, head, t_q, d] x [b, head, d, t_k] = [b, head, t_q, t_k]
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            if self.scale_attn_weights:
                attn_weights = attn_weights / torch.full(
                    [],
                    value.size(-1)**0.5,
                    dtype=attn_weights.dtype,
                    device=attn_weights.device,
                )
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length -
                                    query_length:key_length, :key_length].to(
                                        torch.bool)  # [1, 1, tq, tk]
            if self.sparse:
                # sparse mask
                sparse_mask = self.full_mask[:, key_length -
                                             query_length:key_length, :
                                             key_length]  # [h, tq, tk]
                causal_mask = causal_mask * sparse_mask
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value,
                                    dtype=attn_weights.dtype).to(
                                        attn_weights.device)
            attn_weights = torch.where(causal_mask,
                                       attn_weights.to(attn_weights.dtype),
                                       mask_value)

            if attention_mask is not None:
                # Apply the attention mask
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)

            # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
            attn_weights = attn_weights.type(value.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            # [b, head, t_q, t_k] x [b, head, t_v, d] = [b, head, t_q, d]
            attn_output = torch.matmul(attn_weights, value)
            return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        # (batch, head, seq_length, head_features)
        return tensor.permute(0, 2, 1,3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size, )
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        query = self.c_q(hidden_states)
        key = self.c_k(hidden_states)
        value = self.c_v(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # RoPE with cache
        if self.pos_type == "rope_attn":
            query_length, key_length = query.size(-2), key.size(-2)
            past_len = 0 if layer_past is None else layer_past[0].size(-2)
            query = self.wpe(query,
                             self.pos_ids[:, past_len:past_len + query_length])
            key = self.wpe(key, self.pos_ids[:,
                                             past_len:past_len + key_length])

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value,
                                               attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads,
                                        self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights, )

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Module):

    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim, bias=False)
        proj_bias = config_get(config, "proj_bias", False)
        self.c_proj = Conv1D(embed_dim, intermediate_size, bias=proj_bias)
        if config.activation_function == 'gelu_new':
            self.act = ACT2FN[config.activation_function]
        else:
            self.act = F.silu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        norm_fn = config_get(config, "norm_fn", "layer_norm")
        if norm_fn == "layer_norm":
            norm_fn = nn.LayerNorm
        elif norm_fn == "rms_norm":
            norm_fn = RMSNorm
        else:
            raise NotImplementedError(
                "Norm function must be \"layer_norm\" or \"rms_norm\", but get {}"
                .format(norm_fn))

        self.ln_1 = norm_fn(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = norm_fn(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config,
                                                is_cross_attention=True,
                                                layer_idx=layer_idx)
            self.ln_cross_attn = norm_fn(hidden_size,
                                         eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)
        self.ffn_fp32 = config_get(config, "ffn_fp32", False)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[
            torch.FloatTensor, ...]]], ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (outputs + cross_attn_outputs[2:]
                       )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        with autocast(enabled=not self.ffn_fp32):
            feed_forward_hidden_states = self.mlp(hidden_states)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(
                    mean=0.0,
                    std=(self.config.initializer_range /
                         math.sqrt(2 * self.config.n_layer)),
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPT2Model):
            module.gradient_checkpointing = value


GPT2_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`GPT2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with gpt2-large:
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.pos_type = config_get(config, "pos_type", "learnable")
        if self.pos_type == "learnable":
            self.wpe = nn.Embedding(config.max_position_embeddings,
                                    self.embed_dim)
        elif self.pos_type == "fixed" or self.pos_type == "rope":
            self.wpe = FixedAbsolutePositionEmbedding(
                config.max_position_embeddings, self.embed_dim, self.pos_type)
        elif self.pos_type == "rope_attn":
            self.wpe = None
        else:
            raise NotImplementedError(
                "pos_type must be learnable, fixed or rope")

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        norm_fn = config_get(config, "norm_fn", "layer_norm")
        if norm_fn == "layer_norm":
            norm_fn = nn.LayerNorm
        elif norm_fn == "rms_norm":
            norm_fn = RMSNorm
        else:
            raise NotImplementedError(
                "Norm function must be \"layer_norm\" or \"rms_norm\", but get {}"
                .format(norm_fn))
        self.ln_f = norm_fn(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape,
                                                    device=device)
            encoder_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if self.pos_type == "learnable":
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        elif self.pos_type == "fixed" or self.pos_type == "rope":
            hidden_states = self.wpe(inputs_embeds, position_ids)
        elif self.pos_type == "rope_attn":
            hidden_states = inputs_embeds
        else:
            raise NotImplementedError(
                "pos_type must be learnable, fixed or rope")

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1), )

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (() if output_attentions
                                and self.config.add_cross_attention else None)
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device)
                        for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # outputs: hidden_states, present, (attentions, cross_attentions)
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1], )

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1], )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2], )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                presents,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadSparseModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(
        self,
        config,
        logit_num,
        use_cond=True,
        cond_dim=128,
        cond_type="linear",
        flash_attn=None,
    ):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, logit_num, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Conditioning
        self.use_cond = use_cond
        if use_cond:
            if cond_type == "tanh":
                self.cond_layer = nn.Sequential(
                    nn.Linear(cond_dim, config.n_embd, bias=False),
                    nn.Tanh(),
                    nn.Linear(config.n_embd, config.n_embd, bias=False),
                )
            elif cond_type == "linear":
                self.cond_layer = nn.Sequential(nn.Linear(cond_dim, config.n_embd, bias=False))
            else:
                raise KeyError(f"Invalid cond_type. {cond_type}")

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()

    def wte(self, input_ids):
        return self.transformer.wte(input_ids)

    def condition(self, input_embed):
        return self.cond_layer(input_embed)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        if self.use_cond:
            inputs_embeds = self.cond_layer(inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, self.wte(input_ids)], dim=1)
            transformer_outputs = self.transformer(
                input_ids=None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            transformer_outputs = self.transformer(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    # flake8: noqa
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]],
                       beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past) for layer_past in past)
