import pytest
import torch

from samantha.components.attention import MultiHeadAttention
from tests.helpers.testing_utils import torch_device


def test_flash_attention():

    batch_size = 4
    seq_len = 7
    d_model = 32
    n_heads = 4

    test_input = torch.randn(batch_size, seq_len, d_model)

    bf_mha = MultiHeadAttention(
        d_model, n_heads, dropout=0, scale=None, is_causal=False, enable_flash=False
    ).eval()
    bf_mha_flash = MultiHeadAttention(
        d_model, n_heads, dropout=0, scale=None, is_causal=False, enable_flash=True
    ).eval()

    bf_mha_flash.load_state_dict(bf_mha.state_dict())

    with torch.no_grad():
        out = bf_mha(test_input)
        out_flash = bf_mha_flash(test_input)
    assert torch.allclose(out, out_flash, atol=1e-7)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("use_rotary_embeddings", [False, True])
@pytest.mark.parametrize("enable_flash", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.no_grad()
def test_qkv_cached(is_causal, use_rotary_embeddings, enable_flash, dtype):
    torch.manual_seed(42)
    if dtype == torch.float16 and torch_device == "cpu":
        return

    batch_size = 1
    seq_len = 50
    d_model = 4096
    n_heads = 32

    test_input = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=torch_device
    )

    mha = MultiHeadAttention(
        d_model,
        n_heads,
        is_causal=is_causal,
        use_rotary_embeddings=use_rotary_embeddings,
        enable_flash=enable_flash,
        enable_mem_efficient=False,
    ).eval()

    mha = mha.to(torch_device, dtype=dtype)

    q, k, v = mha.qkv(test_input)
    score = mha.attention(q, k, v, return_attention=False)
    kv_cache = {}
    mha._use_cache = True
    for idx in range(seq_len):
        q_c, k_c, v_c = mha.qkv(test_input[:, idx : idx + 1], kv_cache=kv_cache)
        torch.testing.assert_close(q[:, :, idx : idx + 1], q_c, atol=1e-3, rtol=1e-7)
        torch.testing.assert_close(k[:, :, idx : idx + 1], k_c[:, :, idx : idx + 1], atol=2e-3, rtol=1e-7)
        torch.testing.assert_close(v[:, :, idx : idx + 1], v_c[:, :, idx : idx + 1])

        score_c = mha.attention(q_c, k_c, v_c, return_attention=False)

    # only the last attention score will match the original one
    torch.testing.assert_close(score[:, -1:], score_c, atol=2e-3, rtol=1e-7)

    # use a prefix
    prefix_len = 2
    kv_cache = {}
    mha._use_cache = True
    q_prefix, k_prefix, v_prefix = mha.qkv(
        test_input[:, :prefix_len], kv_cache=kv_cache
    )
    torch.testing.assert_close(q[:, :, :prefix_len], q_prefix, atol=1e-3, rtol=1e-7)
    torch.testing.assert_close(k[:, :, :prefix_len], k_prefix, atol=2e-3, rtol=1e-7)
    torch.testing.assert_close(v[:, :, :prefix_len], v_prefix, atol=1e-3, rtol=1e-7)

    for idx in range(prefix_len, seq_len):
        q_c, k_c, v_c = mha.qkv(test_input[:, idx : idx + 1], kv_cache=kv_cache)
        torch.testing.assert_close(q[:, :, idx : idx + 1], q_c, atol=1e-3, rtol=1e-7)
        torch.testing.assert_close(k[:, :, idx : idx + 1], k_c[:, :, idx : idx + 1], atol=2e-3, rtol=1e-7)
        torch.testing.assert_close(v[:, :, idx : idx + 1], v_c[:, :, idx : idx + 1])

        score_c = mha.attention(q_c, k_c, v_c, return_attention=False)

    torch.testing.assert_close(score[:, -1:], score_c, atol=2e-4, rtol=1e-7)
