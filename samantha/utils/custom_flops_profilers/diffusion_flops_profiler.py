"""
    FLOPS(Floating Point Operations Per Second):每秒浮点运算次数，是一个衡量硬件速度的指标，维基百科介绍如下
    FLOPs(Floating Point Operations):浮点运算次数，用来衡量模型计算复杂度
"""
import collections
import functools
import time

import torch

enable_tf32 = False


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def get_inner_dim(dim, is_gated=False, multiple_of=256):
    if is_gated:
        return ((8 * dim // 3) + multiple_of - 1) // multiple_of * multiple_of
    else:
        return 4 * dim


def calc_gelu_FLOPs(b, t, h):
    return b * t * h


def calc_silu_FLOPs(b, t, h):
    return b * t * h


def calc_attention_FLOPs(
    b,
    q_length,
    kv_length,
    query_dim,
    key_dim=None,
    inner_dim=None,
    out_dim=None,
    qkv_bias=False,
    o_bias=False,
    use_cache=False,
    use_rope=False,
):
    key_dim = default(key_dim, query_dim)
    inner_dim = default(inner_dim, query_dim)
    out_dim = default(out_dim, query_dim)
    n_FLOPs = 0
    n_FLOPs += calc_fc_FLOPs(b, q_length, query_dim, inner_dim, qkv_bias)  # q
    if use_cache:
        n_FLOPs += 2 * calc_fc_FLOPs(b, q_length, key_dim, inner_dim, qkv_bias)  # kv
    else:
        n_FLOPs += 2 * calc_fc_FLOPs(b, kv_length, key_dim, inner_dim, qkv_bias)  # kv
    if use_rope:
        n_FLOPs += calc_rotary_emb_FLOPs(b, q_length, inner_dim)
        n_FLOPs += calc_rotary_emb_FLOPs(b, kv_length, inner_dim)
    n_FLOPs += b * calc_mm_FLOPs(m=q_length, n=kv_length, k=inner_dim)  # q*k
    n_FLOPs += b * calc_mm_FLOPs(m=q_length, n=inner_dim, k=kv_length)  # qk*v
    n_FLOPs += calc_fc_FLOPs(b, q_length, inner_dim, out_dim, o_bias)  # o
    # print("[attn]",n_FLOPs/1e12,q_length,kv_length,query_dim,key_dim,inner_dim,out_dim)
    return n_FLOPs


def calc_elemops_FLOPs(b, t, h):
    return b * t * h


def calc_rmsnorm_FLOPs(b, t, h, training=False):
    return calc_elemops_FLOPs(b, t, h) * (3 if training else 2)


def calc_layernorm_FLOPs(b, t, h, training=False):
    return calc_elemops_FLOPs(b, t, h) * (5 if training else 4)


def calc_rotary_emb_FLOPs(b, t, h):
    return b * t * h * 2


def calc_ffn_FLOPs(
    b, t, in_dim, out_dim=None, inner_dim=None, bias=False, is_gated=False
):
    n_FLOPs = 0
    out_dim = default(out_dim, in_dim)
    inner_dim = default(inner_dim, get_inner_dim(in_dim, is_gated))
    if is_gated:
        n_FLOPs += calc_fc_FLOPs(b, t, in_dim, 2 * inner_dim, bias)  # up+gate
        n_FLOPs += calc_silu_FLOPs(b, t, inner_dim) + calc_elemops_FLOPs(
            b, t, inner_dim
        )  # mul+silu
        n_FLOPs += calc_fc_FLOPs(b, t, inner_dim, out_dim, bias)  # down
    else:
        n_FLOPs += calc_fc_FLOPs(b, t, in_dim, inner_dim, bias)  # fc1
        n_FLOPs += calc_gelu_FLOPs(b, t, inner_dim)  # gelu
        n_FLOPs += calc_fc_FLOPs(b, t, inner_dim, out_dim, bias)  # fc2
    # print("[ffn]",n_FLOPs/1e12, b,t,in_dim,out_dim,inner_dim)
    return n_FLOPs


def calc_mm_FLOPs(m, n, k):
    # (m,k) * (k,n) -> (m,n)
    return 2 * m * k * n


def calc_fc_FLOPs(b, t, in_dim, out_dim, bias=False):
    n_FLOPs = 0
    n_FLOPs += 2 * (b * t * in_dim * out_dim)
    if bias:
        n_FLOPs += b * t * out_dim
    return n_FLOPs


class TransformerFLOPsCounter:
    def __init__(
        self,
        n_layer,
        n_head,
        query_dim,
        key_dim=None,
        attn_inner_dim=None,
        out_dim=None,
        ffn_inner_dim=None,
        vocab_size=None,
        gated_mlp=False,
        qkv_bias=False,
        o_bias=False,
        mlp_bias=False,
        use_rope=False,
        cross_attn=False,
        norm_type="layer_norm",
        training=False,
    ):
        # MHA
        self.training = training
        assert norm_type in ["layer_norm", "rms_norm"]
        self.norm_type = norm_type
        if self.norm_type == "layer_norm":
            self.norm_factor = 4
        else:
            self.norm_factor = 2
        if self.training:
            self.norm_factor += 1

        self.cross_attn = cross_attn
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.query_dim = query_dim
        self.key_dim = default(key_dim, self.query_dim)
        self.attn_inner_dim = default(attn_inner_dim, self.query_dim)
        self.out_dim = default(out_dim, self.query_dim)
        self.qkv_bias = qkv_bias
        self.o_bias = o_bias
        self.use_rope = use_rope
        # self.head_dim = out_dim // n_head
        # self.model_type = model_type

        # FFN
        self.gated_mlp = gated_mlp
        self.ffn_inner_dim = default(
            ffn_inner_dim, get_inner_dim(self.out_dim, self.gated_mlp)
        )
        self.mlp_bias = mlp_bias

        # calculators
        self.attn_calc = functools.partial(
            calc_attention_FLOPs,
            query_dim=self.query_dim,
            key_dim=self.key_dim,
            inner_dim=self.attn_inner_dim,
            out_dim=self.out_dim,
            qkv_bias=self.qkv_bias,
            o_bias=self.o_bias,
            use_rope=use_rope,
        )
        # print("[attn_calc]", self.attn_calc)
        self.ffn_calc = functools.partial(
            calc_ffn_FLOPs,
            in_dim=self.out_dim,
            out_dim=self.out_dim,
            inner_dim=self.ffn_inner_dim,
            is_gated=self.gated_mlp,
            bias=self.mlp_bias,
        )
        # print("[ffn_calc]", self.ffn_calc)
        if exists(self.vocab_size):
            self.lmhead_calc = functools.partial(
                calc_fc_FLOPs, in_dim=self.out_dim, out_dim=self.vocab_size, bias=False
            )
        else:
            self.lmhead_calc = None

    def calc_generation_context_FLOPs(self, bs, context_len, last_token_only=False):
        return self.calc_FLOPs(
            bs,
            context_len,
            context_len,
            use_cache=False,
            last_token_only=last_token_only,
        )

    def calc_generation_decode_FLOPs(self, bs, kv_length, use_cache=False):
        return self.calc_FLOPs(
            bs, 1, kv_length, use_cache=use_cache, last_token_only=True
        )

    def calc_generation(
        self, bs, seqlen, steps, use_cache=False, last_token_only=False
    ):
        assert not self.training
        n_FLOPs = self.calc_generation_context_FLOPs(bs, seqlen, last_token_only)
        # print("[context_FLOPs]",n_FLOPs/1e12)
        for i in range(1, steps):
            step_FLOPs = self.calc_generation_decode_FLOPs(
                bs, seqlen + i, use_cache=use_cache
            )
            # print(f"[decode_FLOPs:{i}]",step_FLOPs/1e12)
            n_FLOPs += step_FLOPs
        return n_FLOPs

    def calc_FLOPs(
        self, bs, q_length, kv_length, use_cache=False, last_token_only=False
    ):
        # print(f"q,k={q_length,kv_length}")
        n_FLOPs = 0
        # pre layer FLOPs
        per_layer_FLOPs = 0
        # norm
        if self.cross_attn:
            per_layer_FLOPs += (
                calc_elemops_FLOPs(bs, q_length, self.query_dim) * self.norm_factor
            )
            per_layer_FLOPs += (
                calc_elemops_FLOPs(bs, kv_length, self.key_dim) * self.norm_factor
            )
        else:
            per_layer_FLOPs += (
                calc_elemops_FLOPs(bs, q_length, self.query_dim) * self.norm_factor
            )
        # attn
        per_layer_FLOPs += self.attn_calc(bs, q_length, kv_length, use_cache=use_cache)
        # resid+norm
        per_layer_FLOPs += calc_elemops_FLOPs(bs, q_length, self.out_dim)
        per_layer_FLOPs += (
            calc_elemops_FLOPs(bs, q_length, self.out_dim) * self.norm_factor
        )
        # ffn
        per_layer_FLOPs += self.ffn_calc(bs, q_length)
        # resid
        per_layer_FLOPs += calc_elemops_FLOPs(bs, q_length, self.out_dim)

        n_FLOPs += self.n_layer * per_layer_FLOPs
        if self.lmhead_calc is not None:
            n_FLOPs += self.lmhead_calc(bs, 1 if last_token_only else q_length)
        return n_FLOPs


LlamaFlopsCounter = functools.partial(
    TransformerFLOPsCounter,
    gated_mlp=True,
    key_dim=None,
    attn_inner_dim=None,
    out_dim=None,
    ffn_inner_dim=None,
    qkv_bias=False,
    o_bias=False,
    mlp_bias=False,
    use_rope=True,
    cross_attn=False,
    norm_type="rms_norm",
)

MusicLlamaFlopsCounter = functools.partial(
    TransformerFLOPsCounter,
    gated_mlp=True,
    key_dim=None,
    attn_inner_dim=None,
    out_dim=None,
    qkv_bias=False,
    o_bias=False,
    mlp_bias=False,
    use_rope=True,
    cross_attn=False,
    norm_type="rms_norm",
)

GPTFlopsCounter = functools.partial(
    TransformerFLOPsCounter,
    gated_mlp=False,
    key_dim=None,
    attn_inner_dim=None,
    out_dim=None,
    ffn_inner_dim=None,
    qkv_bias=True,
    o_bias=True,
    mlp_bias=True,
    use_rope=False,
    cross_attn=False,
    norm_type="layer_norm",
)

DiffusionTransformerFLOPsCounter = functools.partial(
    TransformerFLOPsCounter,
    gated_mlp=False,
    qkv_bias=False,
    o_bias=True,
    mlp_bias=True,
    use_rope=True,
    norm_type="rms_norm",
)


class TNTBlocksFLOPsCounter:
    def __init__(
        self,
        depth,
        input_dim,
        context_dim,
        fine_dim,
        fine_heads,
        fine_head_dim,
        coarse_dim,
        coarse_heads,
        coarse_head_dim,
        training=False,
    ) -> None:
        self.training = training
        self.depth = depth
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.fine_dim = fine_dim
        self.fine_heads = fine_heads
        self.fine_head_dim = fine_head_dim
        self.coarse_dim = coarse_dim
        self.coarse_heads = coarse_heads
        self.coarse_head_dim = coarse_head_dim

        self.coarse_transformer_calc = DiffusionTransformerFLOPsCounter(
            n_layer=1,
            n_head=self.coarse_heads,
            query_dim=self.coarse_dim,
            key_dim=self.coarse_dim,
            attn_inner_dim=self.coarse_heads * self.coarse_head_dim,
            out_dim=self.coarse_dim,
            ffn_inner_dim=self.coarse_dim * 4,
            training=self.training,
            cross_attn=False,
        )

        self.fine_transformer_calc = DiffusionTransformerFLOPsCounter(
            n_layer=1,
            n_head=self.fine_heads,
            query_dim=self.fine_dim,
            key_dim=self.fine_dim,
            attn_inner_dim=self.fine_heads * self.fine_head_dim,
            out_dim=self.fine_dim,
            ffn_inner_dim=self.fine_dim * 4,
            training=self.training,
            cross_attn=False,
        )

        self.context_cross_attn_calc = DiffusionTransformerFLOPsCounter(
            n_layer=1,
            n_head=self.fine_heads,
            query_dim=self.fine_dim,
            key_dim=self.fine_dim,
            attn_inner_dim=self.fine_heads * self.fine_head_dim,
            out_dim=self.fine_dim,
            ffn_inner_dim=self.fine_dim * 4,
            training=self.training,
            cross_attn=True,
        )

    def __call__(
        self,
        bs,  # b
        semantic_length,  # t'
        fine_length,  # lf
        coarse_length,  # lc
        vc_context_length=0,
    ):
        """
        x: b,d,fine_len,coarse_len -> b*coarse_len,fine_len,d->b*coarse_len,fine_len+1,d(fine_emb)
        time_emb: b,d,fine_len,coarse_len-> b*coarse_len,fine_len,d-> b*coarse_len,1,d -> b*coarse_len,1+fine_len,d
        semantic_context_emb: b,semantic_len,context_dim
        fine_emb : b*coarse_len, 1+fine_len, d
        """
        n_FLOPs = 0
        # fine_to_coarse: (b,lc,df) -> (b,lc,dc)
        perlayer_FLOPs = calc_rmsnorm_FLOPs(
            bs * coarse_length, 1, self.fine_dim, self.training
        ) + calc_fc_FLOPs(
            bs * coarse_length, 1, self.fine_dim, self.coarse_dim, bias=True
        )
        # coarse_transformer: q=(b,lc,dc)
        perlayer_FLOPs += self.coarse_transformer_calc.calc_FLOPs(
            bs, coarse_length, coarse_length, last_token_only=False, use_cache=False
        )
        # crossattn: q=(b,lc,dc),kv=(b,ls,ds)
        perlayer_FLOPs += self.context_cross_attn_calc.calc_FLOPs(
            bs,
            coarse_length,
            semantic_length + vc_context_length + (1 if vc_context_length else 0),
            last_token_only=False,
            use_cache=False,
        )
        # getats
        perlayer_FLOPs += calc_rmsnorm_FLOPs(
            bs, coarse_length, self.coarse_dim, self.training
        ) + calc_fc_FLOPs(bs, coarse_length, self.coarse_dim, self.fine_dim)
        perlayer_FLOPs += self.fine_transformer_calc.calc_FLOPs(
            bs * coarse_length,
            fine_length + 1,
            fine_length + 1,
            last_token_only=False,
            use_cache=False,
        )

        n_FLOPs += self.depth * perlayer_FLOPs
        return n_FLOPs


class TNTDiffusionNetworkFLOPsCounter:
    def __init__(
        self,
        depth,
        input_dim,
        feature_dim,
        context_dim,
        segment_size,
        segment_stride,
        training=False,
    ):
        self.training = training
        self.depth = depth
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.segment_size = segment_size
        self.segment_stride = segment_stride

        self.tnt_block_calc = TNTBlocksFLOPsCounter(
            depth,
            input_dim=feature_dim,
            context_dim=context_dim,
            fine_dim=feature_dim,
            fine_heads=8,
            fine_head_dim=int(feature_dim / 8),
            coarse_dim=feature_dim,
            coarse_heads=8,
            coarse_head_dim=int(feature_dim / 8),
            training=self.training,
        )

        self.reset()

    @staticmethod
    def calc_unfold_len(c, t, k, s, p=0, d=1):
        rounded_t = (t + k - 1) // k * k
        unfold_t = (rounded_t + 2 * p - d * (k - 1) - 1) // s + 1
        unfold_c = c * k
        return (unfold_c, unfold_t)

    def calc_time_embed_FLOPs(self, bs, t):
        n_FLOPs = 0
        n_FLOPs += calc_rmsnorm_FLOPs(bs, t, 256, self.training)  # rmsnorm
        n_FLOPs += calc_fc_FLOPs(bs, t, 256, self.feature_dim, bias=True)  # fc
        n_FLOPs += calc_gelu_FLOPs(bs, t, self.feature_dim)  # gelu
        n_FLOPs += calc_fc_FLOPs(
            bs, t, self.feature_dim, self.feature_dim, bias=True
        )  # fc
        return n_FLOPs

    def calc_vc_FLOPs(self, bs, t):
        n_FLOPs = 0
        n_FLOPs += calc_rmsnorm_FLOPs(bs, t, 32, self.training)
        n_FLOPs += calc_fc_FLOPs(bs, t, 32, self.feature_dim, bias=True)
        n_FLOPs += calc_gelu_FLOPs(bs, t, self.feature_dim)
        n_FLOPs += calc_fc_FLOPs(bs, t, self.feature_dim, self.feature_dim, bias=True)
        return n_FLOPs

    def calc_semantic_context_embed_FLOPs(self, bs, t):
        n_FLOPs = 0
        n_FLOPs += calc_rmsnorm_FLOPs(bs, t, self.context_dim, self.training)
        if self.context_dim == 1:
            pass
        else:
            n_FLOPs += calc_fc_FLOPs(
                bs, t, self.context_dim, self.feature_dim, bias=True
            )
        n_FLOPs += calc_gelu_FLOPs(bs, t, self.feature_dim)
        n_FLOPs += calc_fc_FLOPs(bs, t, self.feature_dim, self.feature_dim, bias=True)
        return n_FLOPs

    def get_total_flops(self):
        return self._flops

    def reset(self):
        self.n_calls = 0
        self._flops = 0

    def __call__(self, bs, seqlen, semantic_seqlen, vc_context_length=0):
        # x: b,input_dim,seqlen (16,128,1176)
        # timesteps: b (16,1,1176)
        # semantic_context: b, semantic_seqlen (16,200)
        # print(f"[bs={bs},seqlen={seqlen},semantic_seqlen={semantic_seqlen},vc_context_length={vc_context_length}]")
        n_FLOPs = 0
        n_FLOPs += self.calc_time_embed_FLOPs(
            bs, seqlen
        )  # time_embed (timesteps)->[b,seqlen,h]->[b,h,seqlen]->[b,h,seg_size,n_segs]
        n_FLOPs += self.calc_semantic_context_embed_FLOPs(
            bs, semantic_seqlen
        )  # semantic_context_embed(semantic_context)->[b,semantic_seqlen,h]
        if vc_context_length:
            n_FLOPs += self.calc_vc_FLOPs(bs, vc_context_length)
        n_FLOPs += calc_fc_FLOPs(
            bs, seqlen, self.input_dim, self.feature_dim, bias=False
        )  # input_map(x)-> [b,h,seqlen]
        _, unflod_seqlen = self.calc_unfold_len(
            self.feature_dim, seqlen, self.segment_size, self.segment_stride
        )  # dpp.unfold(x)->>[b,h,seg_size,n_segs]
        # print(unflod_seqlen)
        n_FLOPs += self.tnt_block_calc(
            bs,
            semantic_length=semantic_seqlen,
            coarse_length=unflod_seqlen,
            fine_length=self.segment_size,
            vc_context_length=vc_context_length,
        )  # blocks->b,d,lf,lc

        n_FLOPs += calc_fc_FLOPs(
            bs, seqlen, self.feature_dim, self.input_dim, bias=False
        )  # output()->b,n,t
        self.n_calls += 1
        self._flops += n_FLOPs
        return n_FLOPs


class SamplerFLOPsCounter:
    def __init__(
        self,
        in_channels=32,
        length=3750,
        num_splits=1,
        depth=16,
        input_dim=32,
        feature_dim=1024,
        context_dim=1,
        segment_size=32,
        segment_stride=32,
    ) -> None:
        self.in_channels = in_channels
        self.length = length
        self.num_splits = num_splits
        self.semantic_hop_size = 125
        self.diffusion_hop_size = 625
        self.diffusion_calc = TNTDiffusionNetworkFLOPsCounter(
            depth, input_dim, feature_dim, context_dim, segment_size, segment_stride
        )

    def __call__(
        self,
        bs: int,
        semantic_seqlen: int = 750,
        num_items: int = None,
        num_chunks: int = 1,
        num_steps: int = 25,
        classifier_free_guidance=1,
    ):
        """
        tmp_emb: bs, self.in_channels, self.length + self.diffusion_hop_size* (num_chunks-1)
        avg_cnt: bs, self.in_channels, self.length + self.diffusion_hop_size* (num_chunks-1)
        prev_noise: bs, self.in_channels, self.length
        """
        n_FLOPs = 0
        num_items = default(num_items, bs)
        for i in range(num_chunks):
            n_FLOPs += self.calc_loop_FLOPs(
                bs=bs,
                seqlen=self.length if i == 0 else self.diffusion_hop_size * 2,
                semantic_seqlen=semantic_seqlen,
                num_steps=num_steps,
                classifier_free_guidance=classifier_free_guidance,
            )
        return n_FLOPs

    def calc_loop_FLOPs(
        self, bs, seqlen, semantic_seqlen, num_steps, classifier_free_guidance=1
    ):
        """
        current: B,C,T
        sigma_i: B,1,T
        """
        n_FLOPs = 0
        per_step_FLOPS = 0
        per_step_FLOPS += self.diffusion_calc(bs, seqlen, semantic_seqlen)
        if classifier_free_guidance != 1:
            per_step_FLOPS += self.diffusion_calc(bs, seqlen, semantic_seqlen)
        n_FLOPs += per_step_FLOPS * num_steps
        return n_FLOPs


def init_diffusion_FLOPs_calculator(diffusion_model, sampler, provider):
    return SamplerFLOPsCounter(
        in_channels=sampler.in_channels,
        length=sampler.length,
        num_splits=sampler.num_splits,
        depth=len(diffusion_model.blocks.layers),
        input_dim=diffusion_model.input_dim,
        feature_dim=diffusion_model.feature_dim,
        context_dim=diffusion_model.context_dim,
        segment_size=diffusion_model.segment_size,
        segment_stride=diffusion_model.segment_stride,
    )


def init_semantic_FLOPs_calculator(model, provider):
    SUPPORT_PROVIDRES = ["flash-llama", "xperf", "ctiga"]
    assert provider in SUPPORT_PROVIDRES

    if provider == "flash-llama":
        print(model.config)
        calc = MusicLlamaFlopsCounter(
            n_layer=model.config.num_hidden_layers,
            n_head=model.config.num_attention_heads,
            query_dim=model.config.hidden_size,
            ffn_inner_dim=model.config.intermediate_size,
            vocab_size=model.config.num_logits,
        )
    elif provider == "ctiga":
        calc = MusicLlamaFlopsCounter(
            n_layer=model.config.n_layer,
            n_head=model.config.n_head,
            query_dim=model.config.hidden_size,
            ffn_inner_dim=model.config.n_inner,
            vocab_size=model.config.num_logits,
        )
    elif provider == "xperf":
        calc = MusicLlamaFlopsCounter(
            n_layer=model.model.config.num_hidden_layers,
            n_head=model.model.config.num_attention_heads,
            query_dim=model.model.config.hidden_size,
            ffn_inner_dim=model.model.config.intermediate_size,
            vocab_size=model.model.config.vocab_size,
        )
    else:
        raise TypeError(
            f"no support provider '{provider}'(support {SUPPORT_PROVIDRES})"
        )

    setattr(calc, "provider", provider)
    return calc


PEAK_TFLOPS_LIST = {
    "T4": {
        torch.float32: 8.1,
        torch.float16: 65,
        torch.bfloat16: 65,
        torch.int8: 130,
        # torch.int4:260,
    },
    "A100": {
        torch.float32: 156 if enable_tf32 else 19.5,  # tf32:156
        torch.float16: 312,
        torch.bfloat16: 312,
        torch.int8: 624,
        torch.double: 9.7,
    },  # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf  # noqa: E501
    "V100": {
        torch.float32: 15.7,
        torch.float16: 125,
        torch.bfloat16: 125,
        torch.double: 7.8,
    },  # https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "A10": {
        torch.float32: 62.5 if enable_tf32 else 31.2,  # tf32:62.5
        torch.float16: 125,
        torch.bfloat16: 125,
        torch.int8: 250,
        # torch.int4:500,
        torch.double: None,
    },  # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/datasheet-new/nvidia-a10-datasheet.pdf
    "A30": {
        torch.float32: 82 if enable_tf32 else 10.3,  # tf32: 82
        torch.float16: 165,
        torch.bfloat16: 165,
        torch.int8: 330,
        # torch.int4:661,
        torch.double: 10.3,  # 5.2/10.3
    },  # https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf
}  # 1e12


def ar_FLOPs_expect(b, t, steps, n=24, h=1536, hh=4128):
    context_FLOPs = n * (8 * b * t * h * h + 4 * b * t * t * h + 6 * b * t * h * hh)
    decode_FLOPs = [
        n * (8 * b * 1 * h * h + 4 * b * 1 * h * (t + i) + 6 * b * 1 * h * hh)
        for i in range(1, steps)
    ]
    all_FLOPs = context_FLOPs + sum(decode_FLOPs)
    return all_FLOPs / 1e12


print(ar_FLOPs_expect(1, 404, 1))
print(ar_FLOPs_expect(4, 404, 1))
print(ar_FLOPs_expect(8, 404, 1))
print(ar_FLOPs_expect(16, 404, 1))
print(ar_FLOPs_expect(32, 404, 1))


def compute_mfu(TFLOPs, ts, dtype, device):
    device_name = torch.cuda.get_device_name(device).split(" ")[-1].split("-")[0]
    assert (
        device_name in PEAK_TFLOPS_LIST
    ), f"{device_name} not in {list(PEAK_TFLOPS_LIST.keys())}"
    peak_TFLOPS = PEAK_TFLOPS_LIST[device_name][dtype]
    return TFLOPs / (ts * peak_TFLOPS)


def ar_mfu_recoder(
    func,
    input_embeds,
    total_tokens,
    temperature,
    num_return,
    calc: MusicLlamaFlopsCounter,
    title="Semantic Profile",
):
    last_token_only = True
    if calc.provider == "flash-llama":
        num_return = 1
        last_token_only = False
        use_cache = True
    profile_dict = collections.OrderedDict()

    @functools.wraps(func)
    def wrapper(func):
        dtype = input_embeds.dtype
        device = input_embeds.device
        bs, seqlen = input_embeds.shape[0] * num_return, input_embeds.shape[1]
        TFLOPs = (
            calc.calc_generation(
                bs,
                seqlen,
                total_tokens,
                use_cache=use_cache,
                last_token_only=last_token_only,
            )
            / 1e12
        )

        st = time.perf_counter()
        samples = func(input_embeds, total_tokens, temperature)
        et = time.perf_counter()

        ts = et - st

        profile_dict.update(
            dict(
                batch=bs,
                prompt_len=seqlen,
                gen_len=total_tokens,
                dtype=dtype,
                provider=calc.provider,
                latency=ts,
                TFLOPs=TFLOPs,
                MFU=compute_mfu(TFLOPs, ts, dtype, device),
            )
        )
        verbose_str = ""
        verbose_str += "=" * 120 + "\n" + title + "\n"
        for k, v in profile_dict.items():
            verbose_str += f"{k}:{round(v,3) if isinstance(v,float) else v}\n"
        verbose_str += "=" * 120 + "\n"
        print(verbose_str)

        return samples, profile_dict

    return wrapper(func)


def diffusion_mfu_recoder(
    func,
    diffusion_model,
    samples,
    params,
    calc: SamplerFLOPsCounter,
    title="Diffusion Profile",
):
    profile_dict = collections.OrderedDict()

    num_chunks = params.get("num_chunks", 1)
    diffusion_steps = params.get("diffusion_steps", 25)
    schedule_slope = params.get("schedule_slope", 2.5)
    guidance_scale = params.get("guidance_scale", 3)
    bf16_portion = params.get("bf16_portion", 1.0)

    @functools.wraps(func)
    def wrapper(func):
        device = samples.device
        bs, semantic_seqlen = samples.shape[:2]
        TFLOPs = (
            calc(bs, semantic_seqlen, None, num_chunks, diffusion_steps, guidance_scale)
            / 1e12
        )

        st = time.perf_counter()
        pred_emb = func(
            model=diffusion_model,
            semantic_context=samples,
            num_items=bs,
            num_chunks=num_chunks,
            num_steps=diffusion_steps,
            bf16_portion=bf16_portion,
            start=None,
            show_progress=True,
            angle_schedule="linear",
            schdeule_slope=schedule_slope,
            classifier_free_guidance=guidance_scale,
        )
        et = time.perf_counter()

        ts = et - st

        profile_dict.update(
            dict(
                batch=bs,
                semantic_len=semantic_seqlen,
                output_len=pred_emb.shape[-1],
                dtype=pred_emb.dtype,
                provider=getattr(calc, "provider", "torch"),
                latency=ts,
                TFLOPs=TFLOPs,
                MFU=compute_mfu(TFLOPs, ts, pred_emb.dtype, device),
            )
        )
        verbose_str = ""
        verbose_str += "=" * 120 + "\n" + title + "\n"
        for k, v in profile_dict.items():
            verbose_str += f"{k}:{round(v,3) if isinstance(v,float) else v}\n"
        verbose_str += "=" * 120 + "\n"
        print(verbose_str)

        return samples, profile_dict

    return wrapper(func)


if __name__ == "__main__":
    dtype = torch.float32
    device = "cuda:0"
    seqlen = 404
    dim = 1536
    total_tokens = 1

    """
    ======== compute semantic mfu =======
    """
    print("********* TEST SEMANTIC AR MODEL **********")
    from recipes.bigmusic.lightning.semantic_modules import SemanticModule
    from samantha.models.flash_llama import LlamaRMSNorm, LlamaRotaryEmbedding

    # semantic_model_dir = (
    #     "/mnt/bn/yyf-merlin-nfs/assets/bigmusic/bigmusic_lyrics2song/0.7B"
    # )

    semantic_module = (
        SemanticModule.load_from_checkpoint(
            # f"{model_dir}/last_semantic_ar.ckpt"
            "/mnt/bn/lyrics-to-song/qq/logs/semantic_model_mulan_text_07B/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=094000-val_accu_0=20.53.ckpt"  # noqa: E501
        )
        .to(device)
        .eval()
    )
    print(semantic_module)

    # # use deepspeed
    # from deepspeed.profiling.flops_profiler import get_model_profile
    # from deepspeed.accelerator import get_accelerator
    # from transformers.activations import *
    # def deepspeed_profile(bs):
    #     ignore_modules=[LlamaRotaryEmbedding,SiLUActivation,LlamaRMSNorm]
    #     get_model_profile(
    #         semantic_module.model,
    #             kwargs=dict(
    #                 inputs_embeds=torch.randn(bs,seqlen,dim,device=device,dtype=dtype),
    #                 past_key_values=None,
    #                 use_cache=True
    #             ),ignore_modules=ignore_modules,warm_up=1,detailed=False
    #     )

    #     for step in range(1,750):
    #         get_model_profile(
    #             semantic_module.model,
    #             kwargs=dict(
    #                 inputs_embeds=torch.randn(bs,1,dim,device=device,dtype=dtype),
    #                 past_key_values=[(torch.randn(bs,12,seqlen+step,128,device=device,dtype=dtype),torch.randn(bs,12,seqlen+step,128,device=device,dtype=dtype)) for _ in range(24)],  # noqa: E501
    #                 use_cache=True
    #             ),ignore_modules=ignore_modules,warm_up=1,detailed=False
    #         )
    # deepspeed_profile(1)
    # deepspeed_profile(4)
    # deepspeed_profile(8)
    # deepspeed_profile(16)

    ar_calc = init_semantic_FLOPs_calculator(semantic_module.model, "flash-llama")
    # warmpup
    for i in range(1):
        print(f"warmup {i} ...")
        semantic_module.super_predict(
            torch.randn(1, seqlen, dim, device=device, dtype=dtype), total_tokens, 1.0
        )

    print("compute mfu ...")
    for bs in [1, 4, 8, 16, 32]:
        ar_mfu_recoder(
            semantic_module.super_predict,
            torch.randn(bs, seqlen, dim, device=device, dtype=dtype),
            total_tokens,
            1.0,
            1,
            ar_calc,
        )

    """
    ======== compute diffusion mfu =======
    """
    print("********* TEST DIFFUSION MODEL **********")
    from bigmusic.lyrics2song import init_diffusion, init_sampler

    enable_tf32 = False
    diffusion_model = init_diffusion(
        "/mnt/bn/audio-diffusion/wtl/diffusion/model_14_30s_finetune/checkpoints/last.ckpt",
        device,
    )["diffusion"]
    print(diffusion_model)
    sampler = init_sampler(None, device)["sampler"]

    diffusion_calc = init_diffusion_FLOPs_calculator(diffusion_model, sampler, None)
    # warmup
    for i in range(4):
        print(f"warmup {i} ...")
        sampler(
            model=diffusion_model,
            semantic_context=torch.randint(
                0, 100, (1, total_tokens), device=device, dtype=torch.long
            ),
            num_items=1,
            num_chunks=1,
            num_steps=25,
            bf16_portion=1.0,
            start=None,
            show_progress=True,
            angle_schedule="linear",
            schdeule_slope=2.5,
            classifier_free_guidance=3,
        )

    print("compute mfu ...")
    for bs in [1, 4, 8, 16, 32]:
        diffusion_mfu_recoder(
            sampler,
            diffusion_model,
            torch.randint(0, 100, (bs, total_tokens), device=device, dtype=torch.long),
            calc=diffusion_calc,
            params={},
        )
