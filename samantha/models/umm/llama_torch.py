# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from samantha.utils.flops_profiler import conv_flops
from samantha.utils.triton.sparse_fn import _get_sparse_fn

__all__ = ["ModelArgs", "LLaMa", "LLaMaEncoder", "ResidualBlock", "WeightedSum"]


@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    vocab_size: int = 16384  # defined later by tokenizer
    out_dim: int = 1024  # maybe not same as vocab_size
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6

    max_batch_size: int = 32
    max_seq_len: int = 8000
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    sparse: bool = True
    checkpointing: bool = False


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@torch.cuda.amp.autocast(enabled=False)
def precompute_freqs_cis(dim: int, end: int, theta: float = 20000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


@torch.cuda.amp.autocast(enabled=False)
def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # reshape rotary embeeding reshape
    if freqs_cis.dim() == 2:  # [t, d]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    elif freqs_cis.dim() == 3:
        freqs_cis = freqs_cis.unsqueeze(2)  # [b, t, d] -> [b, t, h=1, d]
    else:
        raise NotImplementedError

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.sparse = args.sparse
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)
        # for block sparse
        self.block = 32
        # for inference
        self.args.use_cache = False

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.args.use_cache:
            if not hasattr(self, "cache_k") or not hasattr(self, "cache_v"):
                self.cache_k = torch.zeros(
                    (
                        self.args.max_batch_size,
                        self.args.max_seq_len,
                        self.n_local_heads,
                        self.head_dim,
                    )
                ).to(xq)
                self.cache_v = torch.zeros(
                    (
                        self.args.max_batch_size,
                        self.args.max_seq_len,
                        self.n_local_heads,
                        self.head_dim,
                    )
                ).to(xq)
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            # cache k and v
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)  # [b, t, head, dim] -> [b, head, t, dim]
        keys = keys.transpose(1, 2)  # [b, t, head, dim] -> [b, head, t, dim]
        values = values.transpose(1, 2)  # [b, t, head, dim] -> [b, head, t, dim]

        if not self.sparse or not self.training:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            size = keys.size()[1:]  # [head, t, dim]
            fns, pad_len = _get_sparse_fn(size=size, device=xq.device)
            if pad_len != 0:
                xq = F.pad(xq, (0, 0, 0, pad_len))
                keys = F.pad(keys, (0, 0, 0, pad_len))
                values = F.pad(values, (0, 0, 0, pad_len))
            qk_matmul, softmax_fn, wv_matmul = fns
            scores = qk_matmul(xq, keys)
            scores = softmax_fn(
                scores, scale=1 / math.sqrt(self.head_dim), is_causal=True
            )
            scores = self.attn_dropout(scores)
            output = wv_matmul(scores, values)  # [b, head, tq, d]
            output = (
                output[:, :, 0 : size[1], :]
                .transpose(1, 2)
                .contiguous()
                .view(bsz, seqlen, -1)
            )

        wo_output = self.wo(output)
        wo_output = self.resid_dropout(wo_output)

        return wo_output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm.weight.data.mul_(1 / math.sqrt(2))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLaMa(nn.Module):
    def __init__(self, params=ModelArgs(), token_input=True):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.token_input = token_input

        if self.token_input:
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.out_dim, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.apply(self._init_weights)

    def forward(
        self, inputs: torch.Tensor, start_pos: int = 0, pos_ids=None, token_input=None
    ):
        token_input = self.token_input if token_input is None else token_input
        if token_input:
            _bsz, seqlen = inputs.shape
            h = self.tok_embeddings(inputs)
        else:
            _bsz, seqlen, _ = inputs.shape
            h = inputs

        self.freqs_cis = self.freqs_cis.to(h.device)
        if pos_ids is None:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        else:
            freqs_cis = torch.nn.functional.embedding(pos_ids, self.freqs_cis)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=inputs.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            if self.params.checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*_inputs):
                        return module(*_inputs)

                    return custom_forward

                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), h, start_pos, freqs_cis, mask
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask)  # causal mask

        h = self.norm(h)
        output = self.output(h)
        return output

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True


class LLaMaEncoder(LLaMa):
    def __init__(self, params=ModelArgs(), token_input=True):
        super().__init__(params, token_input)

    def forward(
        self,
        inputs: torch.Tensor,
        start_pos: int = 0,
        pos_ids=None,
        token_input=None,
        mask=None,
    ):
        token_input = self.token_input if token_input is None else token_input
        if token_input:
            _bsz, seqlen = inputs.shape
            h = self.tok_embeddings(inputs)
        else:
            _bsz, seqlen, _ = inputs.shape
            h = inputs

        self.freqs_cis = self.freqs_cis.to(h.device)
        if pos_ids is None:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        else:
            freqs_cis = torch.nn.functional.embedding(pos_ids, self.freqs_cis)

        for layer in self.layers:
            if self.params.checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*_inputs):
                        return module(*_inputs)

                    return custom_forward

                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), h, start_pos, freqs_cis, mask
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask)  # causal mask

        h = self.norm(h)
        output = self.output(h)
        return output


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-5):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.pad_layer = nn.ConstantPad1d([2 * padding, 0], 0.0)
        self.padding = padding

    def forward(self, x):
        x = self.conv(self.pad_layer(x))
        return x

    def get_flops(self, b, c, t):
        return conv_flops(self.conv, [b, c, t + self.padding * 2])


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=device,
        )
        if isinstance(padding, int):
            # [N, C, H, W] = [N, C=1, T, D] in UMM
            self.pad_layer = nn.ConstantPad1d([padding, padding, 2 * padding, 0], 0.0)
        else:
            raise Exception("Not supported padding {}".format(padding))

        self.padding = padding

    def forward(self, x):
        x = self.conv(self.pad_layer(x))
        return x

    def get_flops(self, b, c, t, d):
        return conv_flops(self.conv, [b, c, t + self.padding * 2, d + self.padding * 2])


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel_size=3, is_causal=False, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.is_causal = is_causal
        conv_cls = CausalConv1d if is_causal else nn.Conv1d
        self.conv_block = nn.Sequential(
            LayerNorm(channel_num, dim=1),
            conv_cls(
                channel_num, channel_num, kernel_size, padding=(kernel_size - 1) // 2
            ),
            LambdaLayer(lambda x: x * (kernel_size**-0.5)),
            nn.GELU(),
            conv_cls(
                channel_num, channel_num, kernel_size, padding=(kernel_size - 1) // 2
            ),
        )

    def forward(self, x, position_embeddings=None):
        if self.transpose:
            x = x.transpose(1, 2)
        residual = x
        x = self.conv_block(x)
        x = x + residual
        if self.transpose:
            x = x.transpose(1, 2)
        return x

    def get_flops(self, b, t):

        if self.is_causal:
            flops1, output_shape1 = self.conv_block[1].get_flops(b, 1, t)
            flops2, _ = self.conv_block[4].get_flops(*output_shape1)
        else:
            flops1, output_shape1 = conv_flops(self.conv_block[1], [b, 1, t])
            flops2, _ = conv_flops(self.conv_block[4], output_shape1)

        return flops1 + flops2


class WeightedSum(nn.Module):
    def __init__(self, n_weights):
        super().__init__()
        self.weights = nn.Parameters(torch.zeros(size=[n_weights]))
        self.weights.zero_()

    def forward(self, inputs):
        ret = (self.weights.softmax() * torch.stack(inputs, dim=-1)).sum(dim=-1)
        return ret
