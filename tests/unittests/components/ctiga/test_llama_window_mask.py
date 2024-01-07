import pytest
import random
import torch

skip_test = False
try:
    from samantha.models.ctiga.gpt import (
        create_mixer_cls,
        create_block,
        GPT2Config,
        GPTModel,
    )
    from samantha.utils.ctiga.localmask import *
    from samantha.utils.ctiga.padding import pad_input, unpad_input
except Exception:
    skip_test = True


has_cuda = torch.cuda.is_available()
device = "cuda:0"


def random_seqlen(bs, max_seqlen, max_diff_len=8):
    max_seqlen_idx = random.randint(0, bs - 1)
    return torch.tensor(
        [
            max_seqlen
            if i == max_seqlen_idx
            else max_seqlen - random.randint(1, min(max_diff_len, max_seqlen-1))
            for i in range(bs)
        ]
    )


@pytest.mark.skip
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [5, 12, 16])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("seqlen", [17, 32, 178, 244, 388, 512, 768, 1024])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("window_type", ["elemwise","blockwise"])
@pytest.mark.parametrize("varlen", [True, False])
def test_llama(seqlen, causal, local, window_type, varlen, n, d, dtype):
    bs = 3
    if causal and local:
        pytest.skip()
    if causal:
        if window_type == BLOCKWISE_WINDOW_MASK:
            pytest.skip()
        window_size = [-1, 0]
    else:
        window_size = [random.randint(-1, 8) for _ in range(2)]
        if local:
            if window_type == BLOCKWISE_WINDOW_MASK:
                window_size = [-1, random.randint(1, 8)]
        else:
            if window_type == BLOCKWISE_WINDOW_MASK:
                pytest.skip()
            window_size = [-1, -1]
    print(f"window_type={window_type}, window_size={window_size}")
    seqlens = random_seqlen(bs, seqlen)
    key_padding_mask = query_padding_mask = (
        torch.arange(0, seqlen, dtype=torch.int).unsqueeze(0).repeat(bs, 1)
        < seqlens.unsqueeze(1)
    ).to(device)
    window_mask = construct_local_mask(
        seqlen, seqlen, window_size.copy(), window_type, None if not varlen else query_padding_mask, None if not varlen else key_padding_mask, device
    )
    print(
        f"varlen={varlen} causal={causal}, window_size={window_size}, window_type={window_type}, seqlens={seqlens} \n{window_mask.int()}"
    )

    config = GPT2Config(
        vocab_size=1227,
        n_positions=0,
        n_embd=n * d,
        n_layer=5,
        n_head=n,
        layer_norm_epsilon=1e-6,
        n_inner=2816,
        rms_norm=True,
        activation_function="swiglu",
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
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
        flashattn_version=2.3,
        # to add
        causal = causal,
        use_window_mask= local,
        window_size=window_size,
        window_type=window_type,
    )
    # MHA
    attn = create_mixer_cls(config, 0, None, device, dtype)(n * d)
    hidden_states = torch.randn(
        bs, seqlen, n * d, requires_grad=True, dtype=dtype, device=device
    )
    if varlen:
        unpad_hidden_states, indices, cu_seqlens, max_seqlen = unpad_input(
            hidden_states, key_padding_mask
        )
        unpad_out = attn.forward(
            unpad_hidden_states,
            key_padding_mask=key_padding_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            indices=indices,
        )[0]
        out = pad_input(unpad_out, indices, bs, seqlen)

        g = torch.rand_like(out)
        out.backward(g)

        assert not hidden_states.grad.isnan().any()
    else:
        out = attn.forward(hidden_states)[0]
        g = torch.rand_like(out)
        out.backward(g)
        assert not hidden_states.grad.isnan().any()

    # LlamaBlock
    block = create_block(config, 0, None, device, dtype)
    hidden_states = torch.randn(
        bs, seqlen, n * d, requires_grad=True, dtype=dtype, device=device
    )
    if varlen:
        unpad_hidden_states, indices, cu_seqlens, max_seqlen = unpad_input(
            hidden_states, key_padding_mask
        )
        unpad_out = block.forward(
            unpad_hidden_states,
            mixer_kwargs=dict(
                key_padding_mask=key_padding_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                indices=indices,
            ),
        )[0]
        out = pad_input(unpad_out, indices, bs, seqlen)

        g = torch.rand_like(out)
        out.backward(g)
        assert not hidden_states.grad.isnan().any()
    else:
        out = block.forward(hidden_states)[0]
        g = torch.rand_like(out)
        out.backward(g)
        assert not hidden_states.grad.isnan().any()

    # Llama
    model = GPTModel(config, None, device, dtype)
    hidden_states = torch.randn(
        bs, seqlen, n * d, requires_grad=True, dtype=dtype, device=device
    )

    out = model.forward(
        inputs_embeds=hidden_states,
        attention_mask=query_padding_mask if varlen else None,
    )
    g = torch.rand_like(out)
    out.backward(g)
    assert not hidden_states.grad.isnan().any()
