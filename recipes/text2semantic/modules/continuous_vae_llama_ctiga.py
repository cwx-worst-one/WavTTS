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
from samantha.utils.triton.blocksparse import matmul as sparse_matmul
from samantha.utils.triton.blocksparse import softmax as sparse_softmax

from s3a.providers.ctiga.models.gpt import GPTModel
from transformers import GPT2Config


__all__ = ["LLaMa", "LLaMaNAR"]

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
        print('Prepare sparse function "{}"'.format(name))
        layout, block = default_layout(train_len, num_heads)
        qk_matmul = sparse_matmul(
            layout, block, mode="sdd", trans_a=False, trans_b=True, device=device
        )
        softmax_fn = sparse_softmax(layout, block, device, is_dense=False)
        wv_matmul = sparse_matmul(
            layout, block, mode="dsd", trans_a=False, trans_b=False, device=device
        )
        sparse_fns[name] = [qk_matmul, softmax_fn, wv_matmul]
    return sparse_fns[name], pad_len


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
    lang_vocab_size: int = 200


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
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
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


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

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
    def __init__(
        self,
        params: ModelArgs,
        use_speaker_id=False,
        token_input=True,
        provider="default",
        state_dict_path=None,
    ):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.token_input = token_input
        self.use_speaker_id = use_speaker_id
        print('use_speaker_id: ', self.use_speaker_id)

        if self.token_input:
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        assert provider in ("default", "ctiga")
        self.provider = provider
        self.layers = self.create_layers(params=params, provider=provider)

        if self.provider == "default":
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        """
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        """
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.h_output = nn.Linear(params.dim, params.out_dim * 2, bias=False)

        self.prenet = nn.Linear(params.out_dim, params.dim, bias=False)

        # print('modify prenet...!!!')
        # self.prenet = nn.Sequential(
        #     LinearNorm(params.out_dim, 256, bias=False),
        #     nn.ReLU(),
        #     DropoutV1(0.5, True),
        #     LinearNorm(256, 256, bias=False),
        #     nn.ReLU(),
        #     DropoutV1(0.5, True),
        #     LinearNorm(256, params.dim, bias=False),
        # )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.apply(self._init_weights)
        if state_dict_path is not None:
            print(f"restore model state dict from {state_dict_path}")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def _repara(self, stats):
        m, logs = torch.split(stats, self.params.out_dim, dim=-1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z

    def forward(
        self,
        text_ids: torch.Tensor,
        text_id_lens: torch.Tensor,
        bns: torch.Tensor,
        bn_lens: torch.Tensor,
        inputs: torch.Tensor,
        start_pos: int = 0,
        use_cache=False,
        inference_params=None,
    ):

        if self.use_speaker_id:
            extra_shift_num = 3
        else:
            extra_shift_num = 2

        bsz, seqlen = inputs.shape
        token_in_h = self.tok_embeddings(inputs)

        if bns.shape[1] > 0:  # 正常训练和带prompt推理都应该走这个
            bn_in_z = self._repara(bns)
            bn_in_h = self.prenet(bn_in_z)
        else:
            bn_in_z = torch.zeros([1, 0, self.params.out_dim]).to(bns.device)
            bn_in_h = torch.zeros([1, 0, self.params.dim]).to(
                bns.device
            )  # noprompt 推理用的占位符, T为0所以等于没有任何内容

        if use_cache:
            assert bsz == 1
            h = bn_in_h
        else:
            h = []
            for i in range(bsz):
                h.append(
                    torch.cat(
                        # bos_text_sep  + bn + eos_pad0
                        (
                            token_in_h[i, : text_id_lens[i] + extra_shift_num, :],
                            bn_in_h[i, : bn_lens[i], :],
                            token_in_h[i, bn_lens[i] + text_id_lens[i] + extra_shift_num :, :],
                        ),
                        dim=-2,
                    )
                )
            h = torch.stack(h)

        # print('inputs shape: ', inputs.shape)
        # print('inputs: ', inputs[0])
        # print('h: ', h[0])

        h = self.forward_layers(
            h, start_pos=start_pos, seqlen=seqlen, inference_params=inference_params
        )

        if self.provider == "default":
            h = self.norm(h)
        output = self.output(h)
        h_output = self.h_output(h)
        output_dict = {"logits": output.float(), "dense": h_output.float()}
        return output_dict, bn_in_z

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

        if self.provider == "ctiga":
            self.layers.gradient_checkpointing_enable()

    def create_layers(self, params, provider):
        if provider == "ctiga":

            def get_n_inner_dim(n_embd, multiple_of):
                n_inner = 4 * n_embd
                n_inner = int(2 * n_inner / 3)
                N = multiple_of
                return ((n_inner - 1) // N) * N + N

            ctiga_config = GPT2Config(
                n_positions=0,
                vocab_size=params.vocab_size,
                num_logits=params.out_dim,
                n_embd=params.dim,
                n_head=params.n_heads,
                n_layer=params.n_layers,
                layer_norm_epsilon=params.norm_eps,
                attn_pdrop=params.attn_pdrop,
                resid_pdrop=params.resid_pdrop,
                embd_pdrop=0.0,
                n_inner=get_n_inner_dim(params.dim, params.multiple_of),
                # workaround rope forward issue with attention mask
                activation_function="swiglu",
                rotary_emb_fraction=1.0,
                rotary_emb_interleaved=True,
                rotary_emb_compat="default",
                tie_word_embeddings=False,
                initializer_range=0.02,
                rms_norm=True,
                qkv_proj_bias=False,
                out_proj_bias=False,
                mlp_fc1_bias=False,
                mlp_fc2_bias=False,
                use_flash_attn=True,
                fused_bias_fc=True,
                fused_mlp=False,
                fused_dropout_add_ln=True,
                residual_in_fp32=True,
                checkpointing=params.checkpointing,
            )
            layers = GPTModel(ctiga_config)
            del layers.embeddings.word_embeddings
            return layers
        else:
            layers = torch.nn.ModuleList()
            for layer_id in range(params.n_layers):
                layers.append(TransformerBlock(layer_id, params))
            return layers

    def forward_layers(self, h, start_pos, seqlen, inference_params=None):
        if self.provider == "ctiga":
            return self.layers(inputs_embeds=h, inference_params=inference_params)
        else:
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (1, 1, seqlen, seqlen), float("-inf"), device=h.device
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
            return h

    def flops_fn(self, batch_size, seqlen):
        params = self.params
        L, H, V, O, M = params.n_layers, params.dim, params.vocab_size, params.out_dim, params.multiple_of
        prenet = O * H
        h_output = H * O * 2
        output = H * V
        n_inner = int(2 * 4 * H / 3)
        inner_size = (n_inner - 1) // M * M + M
        layers = 4 * H * H * L + 3 * inner_size * H * L

        N = prenet + h_output + output + layers
        num_tokens = batch_size * seqlen
        flops_per_token = 3 * 2 * (N + 2 * L * H * seqlen + 3 * L * H)
        return num_tokens * flops_per_token


def checkpoint_converter(origin_ckpt_path, ctiga_ckpt_path, only_weights=False):
    import copy

    def map_to_hf(origin_state_dict, params):
        hidden_size, n_head, n_layer = params.dim, params.n_heads, params.n_layers
        hf_state_dict = copy.deepcopy(origin_state_dict)

        head_dim = hidden_size // n_head
        for i in range(n_layer):
            attn_layer = f"model.layers.{i}.attention"
            mlp = f"model.layers.{i}.feed_forward"
            input_norm = f"model.layers.{i}.attention_norm"
            ffn_norm = f"model.layers.{i}.ffn_norm"

            qw = hf_state_dict.pop(f"{attn_layer}.wq.weight")
            qw = (
                qw.reshape(n_head, head_dim // 2, 2, hidden_size)
                .transpose(1, 2)
                .reshape(hidden_size, hidden_size)
            )
            hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = qw

            kw = hf_state_dict.pop(f"{attn_layer}.wk.weight")
            kw = (
                kw.reshape(n_head, head_dim // 2, 2, hidden_size)
                .transpose(1, 2)
                .reshape(hidden_size, hidden_size)
            )
            hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = kw

            hf_state_dict[
                f"model.layers.{i}.self_attn.v_proj.weight"
            ] = hf_state_dict.pop(f"{attn_layer}.wv.weight")
            hf_state_dict[
                f"model.layers.{i}.self_attn.o_proj.weight"
            ] = hf_state_dict.pop(f"{attn_layer}.wo.weight")

            hf_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = hf_state_dict.pop(
                f"{mlp}.w1.weight"
            )
            hf_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = hf_state_dict.pop(
                f"{mlp}.w2.weight"
            )
            hf_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = hf_state_dict.pop(
                f"{mlp}.w3.weight"
            )

            hf_state_dict[
                f"model.layers.{i}.input_layernorm.weight"
            ] = hf_state_dict.pop(f"{input_norm}.weight")
            hf_state_dict[
                f"model.layers.{i}.post_attention_layernorm.weight"
            ] = hf_state_dict.pop(f"{ffn_norm}.weight")
        return hf_state_dict

    def map_to_ctiga(hf_state_dict, params):
        hidden_size, n_head, n_layer = params.dim, params.n_heads, params.n_layers

        def permute(w):
            return (
                w.view(n_head, 2, hidden_size // n_head // 2, hidden_size)
                .transpose(1, 2)
                .reshape(hidden_size, hidden_size)
            )

        ctiga_state_dict = copy.deepcopy(hf_state_dict)
        if only_weights:
            ctiga_state_dict["tok_embeddings.weight"] = ctiga_state_dict.pop(
                "model.tok_embeddings.weight"
            )
            ctiga_state_dict["layers.ln_f.weight"] = ctiga_state_dict.pop(
                "model.norm.weight"
            )
            ctiga_state_dict["output.weight"] = ctiga_state_dict.pop(
                "model.output.weight"
            )
            ctiga_state_dict["prenet.weight"] = ctiga_state_dict.pop(
                "model.prenet.weight"
            )
            ctiga_state_dict["h_output.weight"] = ctiga_state_dict.pop(
                "model.h_output.weight"
            )
        else:
            ctiga_state_dict["model.layers.ln_f.weight"] = ctiga_state_dict.pop(
                "model.norm.weight"
            )
        from_prefix = "model.layers.{}.{}"
        to_prefix = (
            "layers.layers.{}.{}" if only_weights else "model.layers.layers.{}.{}"
        )
        for layer_idx in range(n_layer):
            # RMS-Norm
            # input_layernorm -> norm1
            ctiga_state_dict[
                f"{to_prefix.format(layer_idx, 'norm1')}.weight"
            ] = ctiga_state_dict.pop(
                f"{from_prefix.format(layer_idx, 'input_layernorm')}.weight"
            )
            # post_attention_layernorm -> norm2
            ctiga_state_dict[
                f"{to_prefix.format(layer_idx, 'norm2')}.weight"
            ] = ctiga_state_dict.pop(
                f"{from_prefix.format(layer_idx, 'post_attention_layernorm')}.weight"
            )

            # attention block
            from_ = from_prefix.format(layer_idx, "self_attn")
            to_ = to_prefix.format(layer_idx, "mixer")

            # [self_attn.q_proj,self_attn.k_proj,self_attn.v_proj] -> mixer.Wqkv
            q, k, v, o = [
                ctiga_state_dict.pop(f"{from_}.{k}_proj.weight")
                for k in ["q", "k", "v", "o"]
            ]
            q = permute(q)
            k = permute(k)
            ctiga_state_dict[f"{to_}.Wqkv.weight"] = torch.cat([q, k, v], dim=0)
            # self_attn.o_proj -> mixer.out_proj
            ctiga_state_dict[f"{to_}.out_proj.weight"] = o

            # mlp
            # [mlp.gate_proj,mlp.up_proj] -> mlp.fc1
            from_ = from_prefix.format(layer_idx, "mlp")
            to_ = to_prefix.format(layer_idx, "mlp")
            ctiga_state_dict[f"{to_}.fc1.weight"] = torch.cat(
                [
                    ctiga_state_dict.pop(f"{from_}.{k}_proj.weight")
                    for k in ["up", "gate"]
                ],
                dim=0,
            )
            ctiga_state_dict[f"{to_}.fc2.weight"] = ctiga_state_dict.pop(
                f"{from_}.down_proj.weight"
            )
        return ctiga_state_dict

    ckpt = torch.load(origin_ckpt_path, map_location="cpu")
    params = ckpt["hyper_parameters"]["model_cls"].keywords["params"]
    hf_sd = map_to_hf(ckpt["state_dict"], params)
    ctiga_sd = map_to_ctiga(hf_sd, params)
    ckpt["state_dict"] = ctiga_sd
    ckpt["optimizer_states"] = {}
    if only_weights:
        torch.save(ctiga_sd, ctiga_ckpt_path)
    else:
        torch.save(ckpt, ctiga_ckpt_path)


if __name__ == "__main__":
    # pass
    origin_ckpt_path = "/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/SFT_from_51k_16A00/0.0.1/checkpoints/last.ckpt"
    ctiga_state_dict_path = "/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/SFT_from_51k_16A00/0.0.1/checkpoints/ctiga_only_weights.ckpt"
    checkpoint_converter(origin_ckpt_path, ctiga_state_dict_path, only_weights=True)
    # when resume model from converted ckpt
    #
    # if only weights:
    # bash recipes/valle/custom_scripts/run_vae_t2s.sh --model_cls.provider ctiga --model_cls.state_dict_path ctiga_only_weights.ckpt
    #
    # else:
    # bash recipes/valle/custom_scripts/run_vae_t2s.sh --model_cls.provider ctiga --ckpt_path ctiga.ckpt
