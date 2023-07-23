# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from re import X
from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, repeat
from triton.ops.blocksparse import matmul as sparse_matmul
from triton.ops.blocksparse import softmax as sparse_softmax


__all__ = [
    "LLaMa",
    "LLaMaNAR",
]

sparse_fns = {}

def _get_sparse_fn(size, device):
    global sparse_fns
    num_heads, train_len, _ = size

    stride = 32
    pad_len = train_len % stride
    if pad_len != 0:
        pad_len = stride - pad_len
    train_len = train_len + pad_len

    name = "head:{}_len:{}_device:{}".format(num_heads, train_len, device)
    if name not in sparse_fns:
        print("Prepare sparse function \"{}\"".format(name))
        layout, block = default_layout(train_len, num_heads)
        qk_matmul = sparse_matmul(
            layout,
            block,
            mode="sdd",
            trans_a=False,
            trans_b=True,
            device=device,
        )
        softmax_fn = sparse_softmax(
            layout,
            block,
            device,
            is_dense=False,
        )
        wv_matmul = sparse_matmul(
            layout,
            block,
            mode="dsd",
            trans_a=False,
            trans_b=False,
            device=device,
        )
        sparse_fns[name] = [qk_matmul, softmax_fn, wv_matmul]
    return sparse_fns[name], pad_len


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 1024  # defined later by tokenizer
    out_dim: int = 1024     # maybe not same as vocab_size
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6

    max_batch_size: int = 32
    max_seq_len: int = 2048
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    sparse: bool = False
    checkpointing: bool = False

    # for codec
    num_res: int = -1
    num_coarse: int = -1
    num_fine: int = -1

    audio_tokens_num: int = 1024
    phone_tokens_num: int = 200


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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, (x.shape[1], x.shape[-1]))
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class PositionEmbedding(nn.Module):
    def __init__(self, max_pos, dim, pos_type="rope"):
        super().__init__()
        assert pos_type in ["rope", "fixed", "rope_attn"]
        self.pos_type = pos_type

        inv_freq = 1. / (10000**(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        position = torch.arange(max_pos, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('wpe', embeddings)

    def forward_fixed(self, x, position_ids):
        dtype = x.dtype
        with autocast(enabled=False):
            out = x.float() + F.embedding(position_ids, self.wpe)
        return out.to(dtype)

    def forward_rope(self, x, position_ids):
        dtype = x.dtype
        with autocast(enabled=False):
            embeddings = F.embedding(position_ids, self.wpe) # [b, t, d]
            embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
            sin, cos = embeddings.unbind(dim=-2)  # b l d//2
            sin = repeat(sin, '... d -> ... (d 2)').float()
            cos = repeat(cos, '... d -> ... (d 2)').float()
            x = x.float()
            out = x * cos + self.rotate_every_two(x) * sin
        return out.to(dtype)

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
            b = x.size(0)
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            if position_ids.size(0) != 1:
                position_ids = position_ids.unsqueeze(1).repeat(1, h, 1).view([b*h, -1]) # [b, t] -> [b*h, t]
            x = self._forward(x, position_ids)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x


def default_layout(train_len, num_heads, block=32):
    if train_len % block != 0:
        train_len = train_len + (block - train_len % block)
    assert train_len % block == 0
    tq = torch.arange(train_len // block).unsqueeze(1)
    tk = torch.arange(train_len // block).unsqueeze(0)
    mask = (tq - tk) >= 0
    mask = mask.unsqueeze(0).repeat(num_heads, 1, 1)
    return mask, block


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        #self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.sparse = args.sparse
        '''
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        '''
        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )
        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)
        # for block sparse
        self.block = 32
        # for inference
        self.args.use_cache = False

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.args.use_cache:
            if not hasattr(self, "cache_k") or not hasattr(self, "cache_v"):
                self.cache_k = torch.zeros(
                    (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).to(xq)
                self.cache_v = torch.zeros(
                    (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
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

        xq = xq.transpose(1, 2)         # [b, t, head, dim] -> [b, head, t, dim]
        keys = keys.transpose(1, 2)     # [b, t, head, dim] -> [b, head, t, dim]
        values = values.transpose(1, 2) # [b, t, head, dim] -> [b, head, t, dim]

        if not self.sparse or not self.training:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            size = keys.size()[1:] # [head, t, dim]
            fns, pad_len = _get_sparse_fn(size=size, device=xq.device)
            if pad_len != 0:
                xq = F.pad(xq, (0, 0, 0, pad_len))
                keys = F.pad(keys, (0, 0, 0, pad_len))
                values = F.pad(values, (0, 0, 0, pad_len))
            qk_matmul, softmax_fn, wv_matmul = fns
            scores = qk_matmul(xq, keys)
            scores = softmax_fn(
                scores,
                scale=1 / math.sqrt(self.head_dim),
                is_causal=True,
            )
            scores = self.attn_dropout(scores)
            output = wv_matmul(scores, values) # [b, head, tq, d]
            output = output[:, :, 0:size[1], :].transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        wo_output = self.wo(output)
        wo_output = self.resid_dropout(wo_output)

        return wo_output


class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of)
        '''
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        '''
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
        self.feed_forward = FeedForward(dim=args.dim,
                                        hidden_dim=4 * args.dim,
                                        multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos,
                                       freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class DropoutV1(nn.Module):
    def __init__(self, p, activated):
        super().__init__()
        self.p = p
        self.activated = activated

    def forward(self, x):
        if self.activated:
            return F.dropout(x, self.p, training=True)
        else:
            return F.dropout(x, self.p, training=self.training)


class LLaMa(nn.Module):
    def __init__(self, params: ModelArgs, token_input=True):
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
        '''
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        '''
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.h_output = nn.Linear(params.dim, params.out_dim, bias=False)

        # self.prenet = nn.Sequential(
        #     nn.Linear(params.out_dim, params.dim, bias=False),
        #     nn.LeakyReLU(0,1),
        #     DropoutV1(0.5, True)
        # )

        self.prenet = nn.Sequential(
            LinearNorm(params.out_dim, 256, bias=False),
            nn.ReLU(),
            DropoutV1(0.5, True),
            LinearNorm(256, 256, bias=False),
            nn.ReLU(),
            DropoutV1(0.5, True),
            LinearNorm(256, params.dim, bias=False),
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2)

        self.apply(self._init_weights)

    def _norm(self, x):
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        norm = torch.norm(x, dim=-1, keepdim=True) * 16**-0.5
        return x / norm.clamp(min=1e-8)

    def forward(self, text_ids: torch.Tensor, text_id_lens: torch.Tensor, 
                    bns: torch.Tensor, bn_lens: torch.Tensor, 
                    inputs: torch.Tensor, start_pos: int = 0, use_cache=False):

        bsz, seqlen = inputs.shape
        token_in_h = self.tok_embeddings(inputs)
        bn_in_h = self.prenet(bns)        

        if use_cache:
            assert bsz == 1
            h=bn_in_h
        else:
            h = []
            for i in range(bsz):
                h.append(
                    torch.cat(
                        # bos_text_sep  + bn + eos_pad0
                        (token_in_h[i, :text_id_lens[i]+2, :], bn_in_h[i, :bn_lens[i], :], token_in_h[i, bn_lens[i]+text_id_lens[i]+2:, :]),
                        dim=-2
                    )
                )
            h = torch.stack(h)

        # print('inputs shape: ', inputs.shape)
        # print('inputs: ', inputs[0])
        # print('h: ', h[0])

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=inputs.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            if self.params.checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*_inputs):
                        return module(*_inputs)
                    return custom_forward
                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h, start_pos, freqs_cis, mask
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask) # causal mask

        h = self.norm(h)
        output = self.output(h)
        h_output = self.h_output(h)
        h_output = self._norm(h_output)
        output_dict = {"logits": output.float(), "dense": h_output.float()}
        return output_dict

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True




if __name__ == '__main__':
    params = ModelArgs()
    model = LLaMa(params).eval()
    params.use_cache = True
    input_tokens = torch.randint(low=0, high=1024, size=[2, 100])

    with torch.no_grad():
        out1 = model(input_tokens)['logits']

        out2 = []
        for i in range(100):
            out = model(input_tokens[:, i:i+1], start_pos=i)
            out2.append(out['logits'])
        out2 = torch.cat(out2, dim=1)

    err = out1 - out2
    print(out1.abs().mean(), err.abs().max(), out2.abs().mean())
