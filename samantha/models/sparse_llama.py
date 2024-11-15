# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU
# General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.cuda.amp import autocast

from samantha.utils.hparams import DotDict
from samantha.utils.triton.sparse_fn import _get_sparse_fn

__all__ = ["LLaMa", "LLaMaNAR"]


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 1024  # defined later by tokenizer
    out_dim: int = 1024  # maybe not same as vocab_size
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


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
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

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        position = torch.arange(max_pos, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j -> ij", position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("wpe", embeddings)

    def forward_fixed(self, x, position_ids):
        dtype = x.dtype
        with autocast(enabled=False):
            out = x.float() + F.embedding(position_ids, self.wpe)
        return out.to(dtype)

    def forward_rope(self, x, position_ids):
        dtype = x.dtype
        with autocast(enabled=False):
            embeddings = F.embedding(position_ids, self.wpe)  # [b, t, d]
            embeddings = rearrange(embeddings, "b l (j d) -> b l j d", j=2)
            sin, cos = embeddings.unbind(dim=-2)  # b l d//2
            sin = repeat(sin, "... d -> ... (d 2)").float()
            cos = repeat(cos, "... d -> ... (d 2)").float()
            x = x.float()
            out = x * cos + self.rotate_every_two(x) * sin
        return out.to(dtype)

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d j -> ... (d j)")

    def _forward(self, x, position_ids):
        if self.pos_type == "fixed":
            return self.forward_fixed(x, position_ids)

        elif self.pos_type == "rope" or self.pos_type == "rope_attn":
            return self.forward_rope(x, position_ids)

    def forward(self, x, position_ids):
        if x.dim() == 3:
            return self._forward(x, position_ids)

        elif x.dim() == 4:
            b = x.size(0)
            h = x.size(1)
            x = rearrange(x, "b h l d -> (b h) l d")
            if position_ids.size(0) != 1:
                position_ids = (
                    position_ids.unsqueeze(1).repeat(1, h, 1).view([b * h, -1])
                )  # [b, t] -> [b*h, t]
            x = self._forward(x, position_ids)
            x = rearrange(x, "(b h) l d -> b h l d", h=h)
            return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.sparse = args.sparse
        """
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
        """
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
        """
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        """
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
        """
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        """
        self.output = nn.Linear(params.dim, params.out_dim, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor, start_pos: int = 0):
        if self.token_input:
            _bsz, seqlen = inputs.shape
            h = self.tok_embeddings(inputs)
        else:
            _bsz, seqlen, _ = inputs.shape
            h = inputs

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

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
        output_dict = DotDict({"logits": output.float()})
        return output_dict

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True


class LLaMaEncoder(LLaMa):
    def __init__(self, params: ModelArgs):
        super().__init__(params=params, token_input=False)

    def forward(self, inputs: torch.Tensor, start_pos: int = 0, mask=None):
        if self.token_input:
            _bsz, seqlen = inputs.shape
            h = self.tok_embeddings(inputs)
        else:
            _bsz, seqlen, _ = inputs.shape
            h = inputs

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        for layer in self.layers:
            if self.params.checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), h, start_pos, freqs_cis, mask
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        output_dict = DotDict({"logits": output.float()})
        return output_dict


class LLaMaNAR(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.res_embedding = nn.Embedding(params.num_res - 1, params.dim)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(params.vocab_size, embedding_dim=params.dim, padding_idx=0)
                for i in range(params.num_res)
            ]
        )
        self.dense_layer = nn.Linear(
            params.dim * params.num_res, params.dim, bias=False
        )
        self.transformer = LLaMaEncoder(params)

        self.apply(self._init_weights)

    def forward(self, x, seq_len, layer_index=None):
        # x: [B, t, n_code]
        device = x.device
        b, t, n_code = x.size()

        # predicted layer index
        if layer_index is None:
            layer_index = torch.randint(low=1, high=n_code, size=[b], device=device)
        # [1, n_code] < [b, 1] = [b, n_code]
        codebook_mask = torch.arange(n_code, device=device).unsqueeze(
            0
        ) < layer_index.unsqueeze(1)
        # seqlen mask: [1, t] < [b, 1] = [b, t]
        seq_mask = torch.arange(t, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        codec_mask = torch.logical_and(
            codebook_mask.unsqueeze(1), seq_mask.unsqueeze(2)
        )
        # TODO: add T partial mask and condition mask

        mask_x = torch.where(codec_mask, x + 1, torch.zeros_like(x))  # offset: pad 1
        embeddings = []
        for i in range(n_code):
            embeddings.append(self.embeddings[i](mask_x[:, :, i]))
        embeddings = torch.cat(embeddings, dim=-1)  # [b, t, d*num_res]
        embeddings = self.dense_layer(embeddings)

        # transformers
        res_embeddings = self.res_embedding(layer_index - 1).unsqueeze(1)  # [b, 1, d]
        outputs = embeddings + res_embeddings
        # attention mask
        attn_mask = torch.full((b, 1, t, t), float("-inf"), device=x.device)
        attn_mask = torch.where(
            seq_mask.unsqueeze(1).unsqueeze(2), torch.zeros_like(attn_mask), attn_mask
        )
        model_outputs = self.transformer(outputs, mask=attn_mask)
        # get targets and loss mask
        targets = []
        x_ = torch.where(
            seq_mask.unsqueeze(2), x + 1, torch.zeros_like(x)
        )  # offset: pad 1
        for i, ind in enumerate(layer_index):
            targets.append(x_[i, :, ind])
        targets = torch.stack(targets, dim=0)
        loss_mask = seq_mask
        assert (targets == 0).sum() == (
            loss_mask == 0
        ).sum()  # 确保loss_mask与输入的一致性
        model_outputs.update(
            {
                "targets": (targets - 1).clamp(0),  # remove offset: pad 1
                "loss_mask": loss_mask,
            }
        )

        return model_outputs

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True


class LLaMaNAR2(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.res_embedding = nn.Embedding(params.num_res - 1, params.dim)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(params.vocab_size, embedding_dim=params.dim, padding_idx=0)
                for i in range(params.num_res)
            ]
        )
        self.dense_layer = nn.Linear(
            params.dim * params.num_res, params.dim, bias=False
        )
        self.transformer = LLaMaEncoder(params)

        self.apply(self._init_weights)

    def forward(self, x, seq_len, seq_sen_id, layer_index=None):
        # x: [B, t, n_code]
        device = x.device
        b, t, n_code = x.size()

        text_len = (seq_sen_id == 1).sum(dim=1)
        wav_len = (seq_sen_id == 2).sum(dim=1)

        # predicted layer index
        if layer_index is None:
            layer_index = torch.randint(low=1, high=n_code, size=[b], device=device)
        # [1, n_code] < [b, 1] = [b, n_code]
        codebook_mask = torch.arange(n_code, device=device).unsqueeze(
            0
        ) < layer_index.unsqueeze(1)
        unmask_len = torch.randint(
            low=1, high=320, size=[b], device=device
        )  # 320 frames means 4 seconds
        rand_len = torch.randint(low=1, high=10000, size=[b], device=device)
        unmask_len = torch.where(
            wav_len < unmask_len, rand_len % wav_len, unmask_len
        ).clamp(
            1
        )  # [b,]
        overall_mask = (unmask_len + text_len).unsqueeze(1) > torch.arange(
            t, device=device
        ).unsqueeze(
            0
        )  # [b, 1] > [1, t] = [b, t]

        # seqlen mask: [1, t] < [b, 1] = [b, t]
        seq_mask = (text_len + wav_len).unsqueeze(1) > torch.arange(
            t, device=device
        ).unsqueeze(
            0
        )  # [b, 1] > [1, t] = [b, t]
        mask = (
            codebook_mask.unsqueeze(1) + overall_mask.unsqueeze(2)
        ) * seq_mask.unsqueeze(
            2
        )  # ([3, 1, 6] + [3, 1034, 1]) * [3, 1034, 1] => [3, 1034, 6]

        # TODO: add T partial mask and condition mask
        mask_x = torch.where(mask, x, torch.zeros_like(x))

        embeddings = []
        for i in range(n_code):
            embeddings.append(self.embeddings[i](mask_x[:, :, i]))
        embeddings = torch.cat(embeddings, dim=-1)  # [b, t, d*num_res]
        embeddings = self.dense_layer(embeddings)

        # transformers
        res_embeddings = self.res_embedding(layer_index - 1).unsqueeze(1)  # [b, 1, d]
        outputs = embeddings + res_embeddings
        # attention mask
        attn_mask = torch.full((b, 1, t, t), float("-inf"), device=x.device)
        attn_mask = torch.where(
            seq_mask.unsqueeze(1).unsqueeze(2), torch.zeros_like(attn_mask), attn_mask
        )
        model_outputs = self.transformer(outputs, mask=attn_mask)

        # get targets and loss mask
        targets = []
        x_ = torch.where(seq_mask.unsqueeze(2), x, torch.zeros_like(x))
        for i, ind in enumerate(layer_index):
            targets.append(x_[i, :, ind])
        targets = torch.stack(targets, dim=0)
        loss_mask = seq_mask * (~overall_mask)
        # assert (targets == 0).sum() == (loss_mask == 0).sum() # 确保loss_mask与输入的一致性
        model_outputs.update(
            {
                "targets": (targets - 1 - self.params.phone_tokens_num).clamp(
                    0
                ),  # remove offset: pad 1
                "loss_mask": loss_mask,
            }
        )

        return model_outputs

    def predict(self, x, seq_len, seq_sen_id, layer_index):
        # x: [B, t, n_code]
        device = x.device
        b, t, n_code = x.size()

        text_len = (seq_sen_id == 1).sum(dim=1)
        wav_len = (seq_sen_id == 2).sum(dim=1)

        # [1, n_code] < [b, 1] = [b, n_code]
        codebook_mask = torch.arange(n_code, device=device).unsqueeze(
            0
        ) < layer_index.unsqueeze(1)
        overall_mask = (seq_len + text_len).unsqueeze(1) > torch.arange(
            t, device=device
        ).unsqueeze(
            0
        )  # [b, 1] > [1, t] = [b, t]

        # seqlen mask: [1, t] < [b, 1] = [b, t]
        seq_mask = (text_len + wav_len).unsqueeze(1) > torch.arange(
            t, device=device
        ).unsqueeze(
            0
        )  # [b, 1] > [1, t] = [b, t]
        mask = (
            codebook_mask.unsqueeze(1) + overall_mask.unsqueeze(2)
        ) * seq_mask.unsqueeze(
            2
        )  # ([3, 6, 1] + [3, 1, 1034]) * [3, 1, 1034] => [3, 6, 1034]

        # TODO: add T partial mask and condition mask
        mask_x = torch.where(mask, x, torch.zeros_like(x))

        embeddings = []
        for i in range(n_code):
            embeddings.append(self.embeddings[i](mask_x[:, :, i]))
        embeddings = torch.cat(embeddings, dim=-1)  # [b, t, d*num_res]
        embeddings = self.dense_layer(embeddings)

        # transformers
        res_embeddings = self.res_embedding(layer_index - 1).unsqueeze(1)  # [b, 1, d]
        outputs = embeddings + res_embeddings
        # attention mask
        attn_mask = torch.full((b, 1, t, t), float("-inf"), device=x.device)
        attn_mask = torch.where(
            seq_mask.unsqueeze(1).unsqueeze(2), torch.zeros_like(attn_mask), attn_mask
        )
        model_outputs = self.transformer(outputs, mask=attn_mask)

        return model_outputs

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )

    def gradient_checkpointing_enable(self):
        self.params.checkpointing = True
