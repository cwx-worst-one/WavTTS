# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from samantha.utils.triton.blocksparse import matmul as sparse_matmul
from samantha.utils.triton.blocksparse import softmax as sparse_softmax


__all__ = [
    "ModelArgs",
    "LLaMa",
    "LLaMaEncoder",
    "LLaMaNAR",
    "LLaMaSoundStormUp",
    "LLaMaCLVP",
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
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
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

    # reshape rotary embeeding reshape
    if freqs_cis.dim() == 2: # [t, d]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    elif freqs_cis.dim() == 3:
        freqs_cis = freqs_cis.unsqueeze(2) # [b, t, d] -> [b, t, h=1, d]
    else:
        raise NotImplementedError

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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
        self.output = nn.Linear(params.dim, params.out_dim, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2)

        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor, start_pos: int = 0, pos_ids=None):
        if self.token_input:
            _bsz, seqlen = inputs.shape
            h = self.tok_embeddings(inputs)
        else:
            _bsz, seqlen, _ = inputs.shape
            h = inputs

        self.freqs_cis = self.freqs_cis.to(h.device)
        if pos_ids is None:
            freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        else:
            freqs_cis = torch.nn.functional.embedding(pos_ids, self.freqs_cis)

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
        output_dict = {"logits": output.float()}
        return output_dict

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True


class LLaMaEncoder(LLaMa):

    def __init__(self, params: ModelArgs, token_input=False):
        super().__init__(params=params, token_input=token_input)

    def forward(self, inputs: torch.Tensor, start_pos: int = 0, pos_ids=None, mask=None):
        if self.token_input:
            _bsz, seqlen = inputs.shape
            h = self.tok_embeddings(inputs)
        else:
            _bsz, seqlen, _ = inputs.shape
            h = inputs

        self.freqs_cis = self.freqs_cis.to(h.device)
        if pos_ids is None:
            freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        else:
            freqs_cis = torch.nn.functional.embedding(pos_ids, self.freqs_cis)

        for layer in self.layers:
            if self.params.checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h, start_pos, freqs_cis, mask
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        output_dict = {"logits": output.float()}
        return output_dict


class LLaMaNAR(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.res_embedding = nn.Embedding(params.num_res - 1, params.dim)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(params.vocab_size, embedding_dim=params.dim, padding_idx=0) for i in range(params.num_res)]
        )
        self.dense_layer = nn.Linear(params.dim * params.num_res, params.dim, bias=False)
        self.transformer = LLaMaEncoder(params)

        self.apply(self._init_weights)

    def forward(self, x, seq_len, layer_index=None):
        # x: [B, t, n_code]
        device = x.device
        b, t, n_code = x.size()

        # predicted layer index
        if layer_index is None:
            layer_index = torch.randint(low=1, high=n_code, size=[b,], device=device)
        # [1, n_code] < [b, 1] = [b, n_code]
        codebook_mask = torch.arange(n_code, device=device).unsqueeze(0) < layer_index.unsqueeze(1)
        # seqlen mask: [1, t] < [b, 1] = [b, t]
        seq_mask = torch.arange(t, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        codec_mask = torch.logical_and(codebook_mask.unsqueeze(1), seq_mask.unsqueeze(2))
        # TODO: add T partial mask and condition mask

        mask_x = torch.where(codec_mask, x + 1, torch.zeros_like(x)) # offset: pad 1
        embeddings = []
        for i in range(n_code):
            embeddings.append(self.embeddings[i](mask_x[:, :, i]))
        embeddings = torch.cat(embeddings, dim=-1) # [b, t, d*num_res]
        embeddings = self.dense_layer(embeddings)

        ### transformers ###
        res_embeddings = self.res_embedding(layer_index - 1).unsqueeze(1) # [b, 1, d]
        outputs = embeddings + res_embeddings
        # attention mask
        attn_mask = torch.full((b, 1, t, t), float("-inf"), device=x.device) # [b, head, tq, tk]
        attn_mask = torch.where(seq_mask.unsqueeze(1).unsqueeze(2), torch.zeros_like(attn_mask), attn_mask)
        model_outputs = self.transformer(outputs, mask=attn_mask)
        # get targets and loss mask
        targets = []
        x_ = torch.where(seq_mask.unsqueeze(2), x + 1, torch.zeros_like(x)) # offset: pad 1
        for i, ind in enumerate(layer_index):
            targets.append(x_[i, :, ind])
        targets = torch.stack(targets, dim=0)
        loss_mask = seq_mask
        assert (targets == 0).sum() == (loss_mask == 0).sum() # 确保loss_mask与输入的一致性
        model_outputs.update(
            {
                "targets": (targets - 1).clamp(0), # remove offset: pad 1
                "loss_mask": loss_mask,
            }
        )

        return model_outputs

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True


class PaddingEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=0,
        std=1.0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        weight = torch.randn(size=[num_embeddings - 1, embedding_dim]) * std
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_buffer("padding_weight", torch.zeros_like(self.weight[0:1, :]))

    def forward(self, ids):
        if self.padding_idx == 0:
            lut = torch.cat([self.padding_weight, self.weight], dim=0)
        elif self.padding_idx == self.weight.size(1) - 1:
            lut = torch.cat([self.weight, self.padding_weight], dim=0)
        else:
            lut = torch.cat([self.weight[0:self.padding_idx], self.padding_weight, self.weight[self.padding_idx:]])
        embeddings = torch.nn.functional.embedding(ids, lut)
        return embeddings


class LLaMaSoundStormUp(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.res_embedding = nn.Embedding(params.num_res, params.dim)
        self.embeddings = nn.ModuleList(
            [
                PaddingEmbedding(
                    params.vocab_size,
                    embedding_dim=params.dim,
                    padding_idx=0,
                ) for i in range(params.num_res)
            ]
        )
        self.cond_embedding = nn.Embedding(1024, params.dim) # n_semantic
        self.cond_conv = nn.Conv1d(params.dim, params.dim, kernel_size=3, padding=1)
        self.dense_layer = nn.Linear(params.dim * (params.num_res + 1), params.dim, bias=False)
        self.transformer = LLaMaEncoder(params)

        self.apply(self._init_weights)

    def forward(self, x, seq_len, cond, cond_len, layer_index=None, prompt_mask=None, cosine_mask=None):
        # x: [B, t, n_code]
        device = x.device
        b, t, n_code = x.size()

        ### mask strategy ###
        # predicted layer index
        if layer_index is None:
            layer_index = torch.randint(low=0, high=n_code, size=[b,], device=device)
        # [1, n_code] < [b, 1] = [b, n_code]
        codebook_mask = torch.arange(n_code, device=device).unsqueeze(0) < layer_index.unsqueeze(1)
        # seqlen mask: [1, t] < [b, 1] = [b, t]
        seq_mask = torch.arange(t, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        # prompt mask strategy
        if prompt_mask is None:
            prompt_mask = self._time_mask(seq_len, max_len=t) # [b, t]
        # cosine mask strategy
        if cosine_mask is None:
            cosine_mask = self._cosine_mask(torch.logical_and(~prompt_mask, seq_mask))
        # merge all mask to determine how codes are selected
        codec_mask = torch.logical_or(cosine_mask.unsqueeze(2), codebook_mask.unsqueeze(1)) # [b, t, n_codebook]
        codebook_mask2 = torch.arange(n_code, device=device).unsqueeze(0) <= layer_index.unsqueeze(1)
        codec_mask = torch.logical_and(codec_mask, codebook_mask2.unsqueeze(1))
        codec_mask = torch.logical_or(codec_mask, prompt_mask.unsqueeze(2)) # [b, t, n_codebook]
        codec_mask = torch.logical_and(codec_mask, seq_mask.unsqueeze(2))

        mask_x = torch.where(codec_mask, x + 1, torch.zeros_like(x)) # offset: pad 1

        # condition embeddings
        cond_embeddings = self.cond_embedding(cond).transpose(1, 2) # [b, t, d] -> [b, d, t]
        cond_embeddings = torch.nn.functional.interpolate(cond_embeddings, scale_factor=1.6) # [b, t, d] -> [b, d, t]
        cond_embeddings = self.cond_conv(cond_embeddings).transpose(1, 2) # [b, d, t] -> [b, t, d]
        if cond_embeddings.size(1) < t:
            cond_embeddings = torch.nn.functional.pad(cond_embeddings, (0, 0, 0, t - cond_embeddings.size(1)))
        else:
            cond_embeddings = cond_embeddings[:, 0:t, :]
        cond_embeddings = torch.where(seq_mask.unsqueeze(2), cond_embeddings, torch.zeros_like(cond_embeddings))

        embeddings = []
        for i in range(n_code):
            embeddings.append(self.embeddings[i](mask_x[:, :, i]))
        embeddings = torch.cat([cond_embeddings] + embeddings, dim=-1) # [b, t, d*num_res]
        embeddings = self.dense_layer(embeddings)

        ### transformers ###
        res_embeddings = self.res_embedding(layer_index).unsqueeze(1) # [b, 1, d]
        outputs = embeddings + res_embeddings
        # attention mask
        attn_mask = torch.full((b, 1, t, t), float("-inf"), device=x.device)
        attn_mask = torch.where(seq_mask.unsqueeze(1).unsqueeze(2), torch.zeros_like(attn_mask), attn_mask)
        model_outputs = self.transformer(outputs, mask=attn_mask)
        # get targets and loss mask
        targets = []
        target_mask = torch.logical_and(seq_mask, ~prompt_mask) # [b, t]
        target_mask = torch.logical_and(target_mask, ~cosine_mask) # [b, t]
        x_ = torch.where(target_mask.unsqueeze(2), x + 1, torch.zeros_like(x)) # offset: pad 1
        for i, ind in enumerate(layer_index):
            targets.append(x_[i, :, ind])
        targets = torch.stack(targets, dim=0)
        loss_mask = target_mask.float()
        model_outputs.update(
            {
                "targets": (targets - 1).clamp(0), # remove offset: pad 1
                "loss_mask": loss_mask,
            }
        )

        return model_outputs

    @torch.no_grad()
    def _cosine_mask(self, exsit_mask):
        device = exsit_mask.device
        with autocast(enabled=False):
            bsz = exsit_mask.size(0)
            rho = 1 - (torch.rand(size=[bsz,], device=device) * math.pi / 2).cos() # [b,]
            mask = torch.rand(size=[bsz, exsit_mask.size(1)], device=device) < rho.unsqueeze(1)
            mask = torch.logical_and(mask, exsit_mask)
        return mask

    @torch.no_grad()
    def _time_mask(self, lengths, max_len=None):
        max_len = lengths.max() if max_len is None else max_len
        device = lengths.device
        with autocast(enabled=False):
            bsz = lengths.size(0)
            rho = torch.rand(size=[bsz,], device=device)
            # avoid all prompted
            mask = torch.arange(max_len, device=device) < (rho * lengths - 1).unsqueeze(1)
            mask = mask.to(device)
        return mask

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Embedding, PaddingEmbedding)):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.n_layers),
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True


class LLaMaCLVP(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.transformer = LLaMaEncoder(params, token_input=True)
        self.dense = nn.Linear(params.out_dim, params.out_dim, bias=False)

    def forward(self, inputs, input_lengths):
        b, t = inputs.shape
        device = inputs.device
        inputs = torch.nn.functional.pad(inputs + 1, (1, 0))

        attn_mask = torch.full((b, 1, t + 1, t + 1), float("-inf"), device=inputs.device)
        seq_mask = torch.arange(t + 1).unsqueeze(0).to(device) < (input_lengths + 1).unsqueeze(1) # [b=1, t] < [b, t=1] = [b, t]
        attn_mask = torch.where(seq_mask.unsqueeze(1).unsqueeze(2), torch.zeros_like(attn_mask), attn_mask)
        outputs = self.transformer(inputs, mask=attn_mask)['logits']

        outputs = torch.where(seq_mask.unsqueeze(2), outputs, torch.zeros_like(outputs))
        outputs = outputs.sum(dim=1) / seq_mask.sum(dim=1).unsqueeze(1)
        outputs = self.dense(outputs)
        outputs = torch.nn.functional.normalize(outputs, dim=1, eps=1e-8)

        return outputs

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True


if __name__ == '__main__':
    params = ModelArgs()
    params.num_res = 12
    model = LLaMaCLVP(params)

    model.eval()
    inputs = torch.randint(low=0, high=1023, size=[2, 100])
    input_lengths = torch.LongTensor([50, 100])

    y = model(inputs, input_lengths)
    inputs = torch.nn.functional.pad(inputs, (0, 100))
    y2 = model(inputs, input_lengths)

    err = y - y2
    print((y**2).sum())
    print(err.abs().max())
