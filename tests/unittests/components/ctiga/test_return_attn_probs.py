from cgitb import enable
import functools
import pytest
import random
import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from samantha.models.ctiga.gpt import GPTLMHeadModel, GPTModel,GPT2Config
from samantha.components.ctiga.block import Block
from samantha.components.ctiga.mha import MHA
from samantha.components.ctiga.mlp import GatedMlp
from samantha.components.ctiga.ops.rms_norm import RMSNorm
from samantha.utils.ctiga.padding import unpad_input, pad_input
from samantha.utils.ctiga.reconstruct_attn_probs import reconstruct_attention_probs

pl.seed_everything(0)

device = "cuda:0"
dtype = torch.float16

llama_mha_cls = functools.partial(
    MHA,
    cross_attn=False,
    qkv_proj_bias=False,
    out_proj_bias=False,
    dwconv=False,
    rotary_emb_interleaved=True,
    use_flash_attn=True,
    return_residual=False,
    version=2,
    device=device,
    dtype=dtype,
)

llama_block_cls = functools.partial(
    Block,
    version=2,
    fused_dropout_add_ln=True,
    return_residual=False,
    residual_in_fp32=True,
)

llama_mlp_cls = functools.partial(
    GatedMlp,
    activation=F.silu,
    bias1=False,
    bias2=False,
    return_residual=False,
    device=device,
    dtype=dtype,
)
llama_norm_cls = functools.partial(RMSNorm, eps=1e-6, device=device, dtype=dtype)

llama_config_cls= functools.partial(GPT2Config,
        vocab_size=1227,
        n_positions=0,  # No absolute position embedding
        # n_embd=llama_config.hidden_size,
        n_layer=24,
        # n_head=llama_config.num_attention_heads,
        # n_inner=llama_config.intermediate_size,
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        bos_token_id=1,
        eos_token_id=2,
        # These are new arguments not in the original GPT2Config
        pad_token_id=0,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,  # align with byteformer
        rotary_emb_compat="default",
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        use_flash_attn=True,
        fused_bias_fc=True,
        fused_mlp=False,
        fused_dropout_add_ln=True,
        residual_in_fp32=True,
        flashattn_version=2,

)

def random_seqlen(bs, max_seqlen, max_diff_len=20):
    max_seqlen_idx = random.randint(0, bs - 1)
    return [
        max_seqlen
        if i == max_seqlen_idx
        else max_seqlen - random.randint(1, min(max_diff_len, max_seqlen))
        for i in range(bs)
    ]


@pytest.mark.skip
@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("seqlen", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("nheads", [1, 4, 16, 24])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
@pytest.mark.parametrize("use_padding_mask", [True, False])
@pytest.mark.parametrize("return_attn_probs", [True, False])
@pytest.mark.parametrize("return_log_attn_probs", [True, False])
def test_llama_mha(
    nheads,
    headdim,
    bs,
    seqlen,
    causal,
    dropout_p,
    use_padding_mask,
    return_attn_probs,
    return_log_attn_probs,
):
    if return_log_attn_probs:
        if not return_attn_probs:
            pytest.skip()

    if dropout_p == 0:
        pytest.skip()

    mha = llama_mha_cls(
        embed_dim=nheads * headdim,
        num_heads=nheads,
        rotary_emb_dim=headdim,
        dropout=dropout_p,
    )

    h = torch.randn(
        bs, seqlen, nheads * headdim, dtype=dtype, device=device, requires_grad=True
    )

    if use_padding_mask:
        key_padding_mask = torch.stack(
            [
                torch.arange(seqlen, device=device) < tmp_seqlen
                for tmp_seqlen in random_seqlen(bs, seqlen)
            ],
            dim=0,
        ).to(device)
        unpad_h, indices, cu_seqlens, max_seqlen = unpad_input(h, key_padding_mask)
        outs = mha.forward(
            unpad_h,
            None,
            key_padding_mask,
            cu_seqlens,
            max_seqlen,
            indices,
            return_attn_probs=return_attn_probs,
        )
        if return_attn_probs:
            unpad_h_out, (lse, score_cummax, dmask) = outs
            attn_probs, dropout_mask = reconstruct_attention_probs(
                dmask,
                lse,
                score_cummax,
                headdim,
                seqlen,
                seqlen,
                key_padding_mask,
                key_padding_mask,
                dropout_p,
                causal,
                return_log_attn_probs,
            )
        else:
            unpad_h_out = outs[0]
        h_out = pad_input(unpad_h_out, indices, bs, seqlen)

    else:
        outs = mha.forward(h, return_attn_probs=return_attn_probs)
        if return_attn_probs:
            h_out, (lse, score_cummax, dmask) = outs
            attn_probs, dropout_mask = reconstruct_attention_probs(
                dmask,
                lse,
                score_cummax,
                headdim,
                seqlen,
                seqlen,
                None,
                None,
                dropout_p,
                causal,
                return_log_attn_probs,
            )
        else:
            h_out = outs[0]

    # backward
    dh = torch.autograd.grad(h_out, h, torch.randn_like(h_out))


@pytest.mark.skip
@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("seqlen", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("nheads", [1, 4, 16, 24])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("use_padding_mask", [True, False])
@pytest.mark.parametrize("return_attn_probs", [True, False])
@pytest.mark.parametrize("return_log_attn_probs", [True, False])
def test_llama_block(
    nheads,
    headdim,
    bs,
    seqlen,
    causal,
    dropout_p,
    use_residual,
    use_padding_mask,
    return_attn_probs,
    return_log_attn_probs,
):
    if return_log_attn_probs:
        if not return_attn_probs:
            pytest.skip()

    if dropout_p == 0:
        pytest.skip()

    mixer_cls = copy.deepcopy(llama_mha_cls)
    mixer_cls.keywords["rotary_emb_dim"] = headdim
    mixer_cls.keywords["dropout"] = dropout_p
    mixer_cls.keywords["num_heads"] = nheads

    block = llama_block_cls(
        dim=nheads * headdim,
        mixer_cls=mixer_cls,
        mlp_cls=llama_mlp_cls,
        norm_cls=llama_norm_cls,
    )

    h = torch.randn(
        bs, seqlen, nheads * headdim, dtype=dtype, device=device, requires_grad=True
    )
    if use_residual:
        residual = torch.randn(
            bs, seqlen, nheads * headdim, dtype=dtype, device=device, requires_grad=True
        )
    else:
        residual = None

    if use_padding_mask:
        key_padding_mask = torch.stack(
            [
                torch.arange(seqlen, device=device) < tmp_seqlen
                for tmp_seqlen in random_seqlen(bs, seqlen)
            ],
            dim=0,
        ).to(device)
        unpad_h, indices, cu_seqlens, max_seqlen = unpad_input(h, key_padding_mask)
        if use_residual:
            unpad_residual, _, _, _ = unpad_input(residual, key_padding_mask)
        else:
            unpad_residual = None
        outs = block.forward(
            unpad_h,
            unpad_residual,
            mixer_kwargs={
                "key_padding_mask": key_padding_mask,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "indices": indices,
            },
            return_attn_probs=return_attn_probs,
        )
        if return_attn_probs:
            unpad_h_out, unpad_residual_out, (lse, score_cummax, dmask) = outs
            attn_probs, dropout_mask = reconstruct_attention_probs(
                dmask,
                lse,
                score_cummax,
                headdim,
                seqlen,
                seqlen,
                key_padding_mask,
                key_padding_mask,
                dropout_p,
                causal,
                return_log_attn_probs,
            )
        else:
            unpad_h_out, unpad_residual_out = outs
        h_out = pad_input(unpad_h_out, indices, bs, seqlen)
        residual_out = pad_input(unpad_residual_out, indices, bs, seqlen)

    else:
        outs = block.forward(h, residual=residual, return_attn_probs=return_attn_probs)
        if return_attn_probs:
            h_out, residual_out, (lse, score_cummax, dmask) = outs
            attn_probs, dropout_mask = reconstruct_attention_probs(
                dmask,
                lse,
                score_cummax,
                headdim,
                seqlen,
                seqlen,
                None,
                None,
                dropout_p,
                causal,
                return_log_attn_probs,
            )
        else:
            h_out, residual_out = outs

    # backward
    dh = torch.autograd.grad(h_out, h, torch.randn_like(h_out))


@pytest.mark.skip
@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("seqlen", [32, 56, 77, 128, 131, 270, 577, 1024])
@pytest.mark.parametrize("nheads", [1, 4, 16, 24])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
@pytest.mark.parametrize("use_padding_mask", [True, False])
@pytest.mark.parametrize("return_attn_probs", [True, False])
@pytest.mark.parametrize("return_log_attn_probs", [True, False])
@pytest.mark.parametrize("enable_lm", [True, False])
def test_llama_model(
    nheads,
    headdim,
    bs,
    seqlen,
    causal,
    dropout_p,
    use_padding_mask,
    return_attn_probs,
    return_log_attn_probs,
    enable_lm,
):
    h = torch.randn(
            bs, seqlen, nheads * headdim, dtype=dtype, device=device, requires_grad=True
        )
    if use_padding_mask:
        key_padding_mask = torch.stack(
            [
                torch.arange(seqlen, device=device) < tmp_seqlen
                for tmp_seqlen in random_seqlen(bs, seqlen)
            ],
            dim=0,
        ).to(device)
    else:
        key_padding_mask=None 
     

    
    model = (GPTLMHeadModel if enable_lm else GPTModel)(config=llama_config_cls(n_embd=nheads*headdim,n_head=nheads),device=device,dtype=dtype)
    outs = model.forward(
        inputs_embeds=h,attention_mask=key_padding_mask,
        return_attn_probs=return_attn_probs,
    )
    if return_attn_probs:
        logits, all_attn_probs = outs.logits,outs.attn_probs
        for (lse,score_cummax,dmask) in all_attn_probs:
            attn_probs, dropout_mask = reconstruct_attention_probs(
                dmask,
                lse,
                score_cummax,
                headdim,
                seqlen,
                seqlen,
                key_padding_mask,
                key_padding_mask,
                dropout_p,
                causal,
                return_log_attn_probs,
            )
    else:
        logits = outs.logits

        # backward
        dh = torch.autograd.grad(logits, h, torch.randn_like(logits))