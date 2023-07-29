# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from samantha.components.attention import MultiHeadAttention
from samantha.byteformers.triton.utils import get_current_cuda_device

# CREDITS:
# Tests with some changes
# https://github.com/openai/triton/blob/master/python/test/unit/operators/test_blocksparse.py
# Initially copied here folowing a fork from the matmul kernel


def block_sparsify_tensor(x, mask, block_size):
    """
    Block sparsify a tensor, given a mask and block size
    """
    ret = torch.empty(
        (x.size(0), mask.sum(), block_size, block_size), dtype=x.dtype, device=x.device
    )

    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[:, idx, :, :] = x[
            :,
            h,
            i * block_size : (i + 1) * block_size,
            j * block_size : (j + 1) * block_size,
        ]
    return ret


_triton_available = torch.cuda.is_available()
_matmul_types = []

if _triton_available:
    try:
        import triton
        from triton.ops.blocksparse import matmul as blocksparse_matmul
        from triton.ops.blocksparse import softmax as blocksparse_softmax

        from samantha.components.attention import (
            BlockSparseAttention,
        )
        from samantha.byteformers.triton.utils import gpu_capabilities_older_than_70

        _triton_available = not gpu_capabilities_older_than_70()
        _matmul_types = ["sdd", "dsd", "dds"]
    except (ImportError, ModuleNotFoundError) as e:
        import logging

        logging.warning(f"Triton is not available: {e}. Some tests will be skipped")
        _triton_available = False


def mask_tensor(x, mask, block, value=0):
    ret = x.clone()
    for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
        ret[:, h, i * block : (i + 1) * block, j * block : (j + 1) * block] = value
    return ret


@pytest.mark.skipif(not _triton_available, reason="Triton requires a recent CUDA gpu")
@pytest.mark.skipif(
    not _triton_available or get_current_cuda_device() == "T4",
    reason="FIXME - blocksparse matmuls are slightly off on T4s",
)
@pytest.mark.parametrize("MODE", _matmul_types)
@pytest.mark.parametrize("TRANS_A", [False, True])
@pytest.mark.parametrize("TRANS_B", [False, True])
@pytest.mark.parametrize("BLOCK", [16, 32, 64])
@pytest.mark.parametrize("DTYPE", [torch.float16])
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=32, H=2, M=512, N=384, K=256):
    # set seed
    torch.random.manual_seed(0)

    # create inputs
    a = torch.randn(
        (Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device="cuda"
    )
    b = torch.randn(
        (Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device="cuda"
    )
    shape = {
        "sdd": (M, N),
        "dsd": (a.shape[2], a.shape[3]),
        "dds": (b.shape[2], b.shape[3]),
    }[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))

    # triton result
    op = blocksparse_matmul(
        layout,
        BLOCK,
        MODE,
        trans_a=TRANS_A,
        trans_b=TRANS_B,
        device=torch.device("cuda"),
    )
    ra = block_sparsify_tensor(a, layout, BLOCK) if MODE == "dsd" else a
    rb = block_sparsify_tensor(b, layout, BLOCK) if MODE == "dds" else b
    rc = triton.testing.catch_oor(lambda: op(ra, rb), pytest)

    # torch result
    ta = mask_tensor(a, layout, BLOCK) if MODE == "dsd" else a
    tb = mask_tensor(b, layout, BLOCK) if MODE == "dds" else b
    ta = ta.transpose(2, 3) if TRANS_A else ta
    tb = tb.transpose(2, 3) if TRANS_B else tb
    tc = torch.matmul(ta, tb)
    tc = mask_tensor(tc, layout, BLOCK) if MODE == "sdd" else tc
    tc = block_sparsify_tensor(tc, layout, BLOCK) if MODE == "sdd" else tc

    # compare
    torch.testing.assert_close(rc, tc)


@pytest.mark.skipif(not _triton_available, reason="Triton requires a recent CUDA gpu")
@pytest.mark.parametrize("BLOCK", [32, 128])
@pytest.mark.parametrize("WIDTH", [256, 576, 1024, 1792])
@pytest.mark.parametrize("DTYPE", [torch.float16, torch.float32])
def test_softmax(BLOCK, WIDTH, DTYPE):
    # set seed
    torch.random.manual_seed(0)
    Z, H, M, N = 2, 4, WIDTH, WIDTH
    scale = 0.4

    # create inputs
    layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
    x = torch.randn((Z, H, M, N), dtype=DTYPE, requires_grad=True, device="cuda")

    # triton result
    op = blocksparse_softmax(layout, BLOCK, device=torch.device("cuda"))
    tx = block_sparsify_tensor(x, layout, BLOCK)
    ty = op(tx, scale=scale)

    # torch result
    rx = mask_tensor(x, layout, BLOCK, value=float("-inf"))
    rx = rx[:, :, : (M // BLOCK) * BLOCK, : (M // BLOCK) * BLOCK]

    ry = torch.softmax(rx * scale, -1)
    ry = block_sparsify_tensor(ry, layout, BLOCK)

    # compare
    torch.testing.assert_close(ry, ty)


@pytest.mark.skipif(not _triton_available, reason="Triton requires a recent CUDA gpu")
@pytest.mark.parametrize("block", [32, 43, 128])  # 16, 32,
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attention_fwd_bwd(
    block, dtype, input_scale=1.0, scale=1 / 8.0, n_ctx=384, batch_size=2, n_heads=2
):
    # inputs
    head_dim = 64
    qkv_shape = (batch_size, n_heads, n_ctx, head_dim)
    qkvs = [
        torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True)
        .to(dtype)
        .cuda()
        for _ in range(3)
    ]

    def loss_fn(x):
        return (x**2).mean()

    # Triton:
    n_blocks = n_ctx // block
    layout = torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long)
    query, key, value = [x.clone() for x in qkvs]
    query.retain_grad()
    key.retain_grad()
    value.retain_grad()
    if block not in [16, 32, 64, 128]:
        # Check that unsupported dimensions are caught
        with pytest.raises(AssertionError):
            _ = BlockSparseAttention(
                layout, block, d_model=head_dim * n_heads, n_heads=n_heads
            )
    else:
        block_sparse_attention = BlockSparseAttention(
            layout, block, d_model=head_dim * n_heads, n_heads=n_heads
        ).to(dtype).cuda()
        attn_out = block_sparse_attention.attention(
            q=query, k=key, v=value, return_attention=False
        )

        # ad hoc loss
        loss = loss_fn(attn_out)
        loss.backward()
        grads = [query.grad, key.grad, value.grad]

        # Torch version:
        torch_q, torch_k, torch_v = [x.clone() for x in qkvs]
        torch_q = torch_q * scale
        torch_q.retain_grad()
        torch_k.retain_grad()
        torch_v.retain_grad()
        scores = scale * torch.einsum("bhsd,bhtd->bhst", torch_q, torch_k)
        probs = torch.softmax(scores, dim=-1)
        torch_attn_out = torch.einsum("bhst,bhtd->bhsd", probs, torch_v)

        # ad hoc loss
        torch_loss = loss_fn(torch_attn_out)
        torch_loss.backward()
        torch_grads = [torch_q.grad, torch_k.grad, torch_v.grad]

        # comparison
        torch.testing.assert_close(
            loss, torch_loss, msg=f"Triton loss {loss} and torch loss {torch_loss}", atol=2e-3, rtol=1e-7
        )

        for g1, g2 in zip(grads, torch_grads):
            torch.testing.assert_close(
                torch.norm(g1),
                torch.norm(g2),
                msg=(
                    f"Triton grad {torch.norm(g1).item()} and torch grad"
                    f" {torch.norm(g2).item()}"
                ),
                atol=2e-4,
                rtol=1e-7,
            )


@pytest.mark.skipif(not _triton_available, reason="Triton requires a recent CUDA gpu")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_blocksparse_attention_parity(dtype):
    def _reset_seeds():
        torch.manual_seed(0)

    seq = 64
    model = 128
    heads = 4
    block_size = 16
    batch_size = 2
    batched_dim = heads * batch_size

    block_size = block_size
    layout = torch.ones(seq // block_size, seq // block_size, dtype=torch.long)

    inputs = torch.rand(batched_dim, seq, model, device="cuda", dtype=dtype)

    _reset_seeds()
    multi_head_sdp = MultiHeadAttention(
        d_model=model, n_heads=heads, dropout=0, causal=False
    ).to(device=torch.device("cuda"), dtype=dtype)
    r_sdp = multi_head_sdp(inputs)

    _reset_seeds()
    multi_head_blocksparse = BlockSparseAttention(
        layout=layout,
        block_size=block_size,
        d_model=model,
        n_heads=heads,
        dropout=0,
        causal=False,
    ).to(device=torch.device("cuda"), dtype=dtype)
    r_blocksparse = multi_head_blocksparse(inputs)

    torch.testing.assert_close(r_sdp, r_blocksparse, atol=3e-3, rtol=1e-7)
