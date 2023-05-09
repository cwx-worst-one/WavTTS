import numpy as np
import pytest
import torch

from samantha.byteformers.components.attention import MultiHeadAttention


def test_flash_attention():

    batch_size = 4
    seq_len = 7
    d_model = 32
    n_heads = 4

    test_input = torch.randn(batch_size, seq_len, d_model)

    bf_mha = MultiHeadAttention(
        d_model, n_heads, dropout=0, scale=None, causal=False, enable_flash=False
    ).eval()
    bf_mha_flash = MultiHeadAttention(
        d_model, n_heads, dropout=0, scale=None, causal=False, enable_flash=True
    ).eval()

    bf_mha_flash.load_state_dict(bf_mha.state_dict())

    with torch.no_grad():
        out = bf_mha(test_input)
        out_flash = bf_mha_flash(test_input)
    assert torch.allclose(out, out_flash, atol=1e-7)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("use_rotary_embeddings", [False, True])
@pytest.mark.parametrize("enable_flash", [False, True])
@torch.no_grad()
def test_qkv_cached(causal, use_rotary_embeddings, enable_flash):

    batch_size = 1
    seq_len = 7
    d_model = 32
    n_heads = 1

    test_input = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(
        d_model,
        n_heads,
        causal=causal,
        use_rotary_embeddings=use_rotary_embeddings,
        enable_flash=enable_flash,
        enable_mem_efficient=False,
    ).eval()

    q, k, v = mha.qkv(test_input)
    score = mha.attention(q, k, v, return_attention=False)

    kv_cache = {}
    mha._use_cache = True
    for idx in range(seq_len):
        q_c, k_c, v_c = mha.qkv(test_input[:, idx : idx + 1], kv_cache=kv_cache)
        np.testing.assert_allclose(q[:, :, idx : idx + 1], q_c, atol=1e-6)
        np.testing.assert_allclose(
            k[:, :, idx : idx + 1], k_c[:, :, idx : idx + 1], atol=1e-6
        )
        np.testing.assert_allclose(
            v[:, :, idx : idx + 1], v_c[:, :, idx : idx + 1], atol=1e-6
        )

        score_c = mha.attention(q_c, k_c, v_c, return_attention=False)

    # only the last attention score will match the original one
    np.testing.assert_allclose(score[:, -1:], score_c, atol=1e-6)

    # use a prefix
    prefix_len = 2
    kv_cache = {}
    mha._use_cache = True
    q_prefix, k_prefix, v_prefix = mha.qkv(
        test_input[:, :prefix_len], kv_cache=kv_cache
    )
    np.testing.assert_allclose(q[:, :, :prefix_len], q_prefix, atol=1e-6)
    np.testing.assert_allclose(k[:, :, :prefix_len], k_prefix, atol=1e-6)
    np.testing.assert_allclose(v[:, :, :prefix_len], v_prefix, atol=1e-6)

    for idx in range(prefix_len, seq_len):
        q_c, k_c, v_c = mha.qkv(test_input[:, idx : idx + 1], kv_cache=kv_cache)
        np.testing.assert_allclose(q[:, :, idx : idx + 1], q_c, atol=1e-6)
        np.testing.assert_allclose(
            k[:, :, idx : idx + 1], k_c[:, :, idx : idx + 1], atol=1e-6
        )
        np.testing.assert_allclose(
            v[:, :, idx : idx + 1], v_c[:, :, idx : idx + 1], atol=1e-6
        )

        score_c = mha.attention(q_c, k_c, v_c, return_attention=False)

    np.testing.assert_allclose(score[:, -1:], score_c, atol=1e-6)
