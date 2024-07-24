import pytest
import torch
import torch.utils.benchmark as benchmark
from byteformers.components.attention import MultiHeadAttention
from byteformers.components.attention.base import (
    flash_scaled_dot_product,
    scaled_dot_product,
)
from byteformers.convert import convert_pytorch_to_byteformers_attention_weights
from torch.nn import MultiheadAttention as MultiheadAttentionPT

from tests.runif import RunIf
from tests.utils import torch_device

torch.set_float32_matmul_precision("high")


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("is_causal", [True, False])
def test_scaled_dot_product(dtype, is_causal):
    device = "cuda"
    batch_size = 2
    n_heads = 2
    seq_len = 7
    d_k = 128
    scale = d_k**-0.5
    q = torch.randn(batch_size, n_heads, seq_len, d_k, dtype=dtype, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, d_k, dtype=dtype, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, d_k, dtype=dtype, device=device)

    score, attention = scaled_dot_product(
        q,
        k,
        v,
        scale=scale,
        dropout_p=0,
        mask=None,
        attn_bias=None,
        is_causal=is_causal,
    )

    flash_score_flash = flash_scaled_dot_product(
        q,
        k,
        v,
        dropout_p=0,
        is_causal=is_causal,
        use_cache=False,
        enable_flash=True,
        enable_mem_efficient=False,
        enable_math=False,
    )

    flash_score_memefficient = flash_scaled_dot_product(
        q,
        k,
        v,
        dropout_p=0,
        is_causal=is_causal,
        use_cache=False,
        enable_flash=False,
        enable_mem_efficient=True,
        enable_math=False,
    )

    flash_score_math = flash_scaled_dot_product(
        q,
        k,
        v,
        dropout_p=0,
        is_causal=is_causal,
        use_cache=False,
        enable_flash=False,
        enable_mem_efficient=False,
        enable_math=True,
    )

    torch.testing.assert_close(
        flash_score_flash, flash_score_memefficient, atol=1e-5, rtol=1
    )
    torch.testing.assert_close(flash_score_flash, flash_score_math, atol=1e-5, rtol=1)

    torch.testing.assert_close(score, flash_score_flash, atol=1e-5, rtol=1)


@torch.no_grad()
@pytest.mark.parametrize("d", ["cpu", "cuda"])
def test_attention_parity(d):
    if d == "cuda" and torch_device != "cuda":
        return

    batch_size = 4
    seq_len = 7
    d_model = 32
    n_heads = 4

    test_input = torch.randn(batch_size, seq_len, d_model, device=d)

    pt_mha = (
        MultiheadAttentionPT(d_model, n_heads, dropout=0, batch_first=True).eval().to(d)
    )

    bf_mha = (
        MultiHeadAttention(
            d_model,
            n_heads,
            dropout=0,
            scale=None,
            causal=False,
            enable_flash=False,
            enable_mem_efficient=False,
        )
        .eval()
        .to(d)
    )

    bf_mha_flash = (
        MultiHeadAttention(
            d_model,
            n_heads,
            dropout=0,
            scale=None,
            causal=False,
            enable_flash=True,
            enable_mem_efficient=False,
        )
        .eval()
        .to(d)
    )

    bf_mha = convert_pytorch_to_byteformers_attention_weights(pt_mha, bf_mha)
    bf_mha_flash = convert_pytorch_to_byteformers_attention_weights(
        pt_mha, bf_mha_flash
    )

    out_pt, _ = pt_mha(test_input, test_input, test_input)
    out = bf_mha(test_input)
    out_flash = bf_mha_flash(test_input)

    torch.testing.assert_close(out_pt, out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_pt, out_flash, atol=1e-6, rtol=1e-6)


# @torch.no_grad()
# @pytest.mark.parametrize("causal", [True, False])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
# @RunIf(min_cuda_gpus=1)
# def test_flash_attention_v2(causal, dtype):
#     batch_size = 32
#     seq_len = 2000
#     d_model = 2048
#     n_heads = 16

#     test_input = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")

#     bf_mha = (
#         MultiHeadAttention(
#             d_model,
#             n_heads,
#             dropout=0,
#             scale=None,
#             causal=causal,
#             enable_flash=False,
#             enable_mem_efficient=False,
#             enable_flash_2=False,
#         )
#         .eval()
#         .to("cuda", dtype=dtype)
#     )

#     bf_mha_flash_v1 = (
#         MultiHeadAttention(
#             d_model,
#             n_heads,
#             dropout=0,
#             scale=None,
#             causal=causal,
#             enable_flash=True,
#             enable_mem_efficient=False,
#             enable_flash_2=False,
#         )
#         .eval()
#         .to("cuda", dtype=dtype)
#     )
#     bf_mha_flash_v1.load_state_dict(bf_mha.state_dict())

#     bf_mha_flash_v2 = (
#         MultiHeadAttention(
#             d_model,
#             n_heads,
#             dropout=0,
#             scale=None,
#             causal=causal,
#             enable_flash=False,
#             enable_mem_efficient=False,
#             enable_flash_2=True,
#         )
#         .eval()
#         .to("cuda", dtype=dtype)
#     )
#     bf_mha_flash_v2.load_state_dict(bf_mha_flash_v1.state_dict())

#     with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
#         out = bf_mha(test_input)
#         out_v1 = bf_mha_flash_v1(test_input)
#         out_v2 = bf_mha_flash_v2(test_input)

#         torch.testing.assert_close(out, out_v1, atol=1e-4, rtol=1e-7)
#         # assert torch.testing.assert_close(out_v1, out_v2, atol=1e-5, rtol=1e-7)

#         t = benchmark.Timer(
#             stmt="bf_mha(test_input)",
#             globals={"bf_mha": bf_mha, "test_input": test_input},
#         )
#         t.timeit(5)  # warmup
#         print(t.timeit(50))

#         t_v1 = benchmark.Timer(
#             stmt="bf_mha_v1(test_input)",
#             globals={"bf_mha_v1": bf_mha_flash_v1, "test_input": test_input},
#         )
#         t_v1.timeit(5)  # warmup
#         print(t_v1.timeit(50))

#         t_v2 = benchmark.Timer(
#             stmt="bf_mha_v2(test_input)",
#             globals={"bf_mha_v2": bf_mha_flash_v2, "test_input": test_input},
#         )
#         t_v2.timeit(5)  # warmup
#         print(t_v2.timeit(50))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("use_rotary_embeddings", [False, True])
@pytest.mark.parametrize("enable_flash", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_qkv_cached(causal, use_rotary_embeddings, enable_flash, dtype):
    MAX_ATOL = 1e-4
    MAX_RTOL = 1e-1

    torch.manual_seed(42)

    if dtype == torch.float32 and enable_flash:
        return

    if dtype == torch.float16 and torch_device == "cpu":
        return

    batch_size = 1
    seq_len = 50
    d_model = 256
    n_heads = 2

    test_input = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=torch_device
    )

    mha = MultiHeadAttention(
        d_model,
        n_heads,
        causal=causal,
        use_rotary_embeddings=use_rotary_embeddings,
        enable_flash=enable_flash,
        enable_mem_efficient=False,
        enable_math=False,
    ).eval()

    mha = mha.to(torch_device, dtype=dtype)

    q, k, v = mha.qkv(test_input)
    score = mha.attention(q, k, v, return_attention=False)
    kv_cache = {}
    mha._use_cache = True
    for idx in range(seq_len):
        q_c, k_c, v_c = mha.qkv(test_input[:, idx : idx + 1], kv_cache=kv_cache)
        torch.testing.assert_close(
            q[:, :, idx : idx + 1], q_c, atol=MAX_ATOL, rtol=1e-7
        )
        torch.testing.assert_close(
            k[:, :, idx : idx + 1],
            k_c[:, :, idx : idx + 1],
            atol=MAX_ATOL,
            rtol=MAX_RTOL,
        )
        torch.testing.assert_close(v[:, :, idx : idx + 1], v_c[:, :, idx : idx + 1])

        score_c = mha.attention(q_c, k_c, v_c, return_attention=False)

    # only the last attention score will match the original one
    torch.testing.assert_close(score[:, -1:], score_c, atol=MAX_ATOL, rtol=MAX_RTOL)

    # use a prefix
    prefix_len = 2
    kv_cache = {}
    mha._use_cache = True
    q_prefix, k_prefix, v_prefix = mha.qkv(
        test_input[:, :prefix_len], kv_cache=kv_cache
    )
    torch.testing.assert_close(
        q[:, :, :prefix_len], q_prefix, atol=MAX_ATOL, rtol=MAX_RTOL
    )
    torch.testing.assert_close(
        k[:, :, :prefix_len], k_prefix, atol=MAX_ATOL, rtol=MAX_RTOL
    )
    torch.testing.assert_close(
        v[:, :, :prefix_len], v_prefix, atol=MAX_ATOL, rtol=MAX_RTOL
    )

    for idx in range(prefix_len, seq_len):
        q_c, k_c, v_c = mha.qkv(test_input[:, idx : idx + 1], kv_cache=kv_cache)
        torch.testing.assert_close(
            q[:, :, idx : idx + 1], q_c, atol=MAX_ATOL, rtol=MAX_RTOL
        )
        torch.testing.assert_close(
            k[:, :, idx : idx + 1],
            k_c[:, :, idx : idx + 1],
            atol=MAX_ATOL,
            rtol=MAX_RTOL,
        )
        torch.testing.assert_close(v[:, :, idx : idx + 1], v_c[:, :, idx : idx + 1])

        score_c = mha.attention(q_c, k_c, v_c, return_attention=False)

    torch.testing.assert_close(score[:, -1:], score_c, atol=MAX_ATOL, rtol=MAX_RTOL)
