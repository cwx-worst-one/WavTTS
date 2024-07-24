import pytest
import torch
from byteformers.components.positional_embedding.rotary import (
    LlamaRotaryEmbedding,
    llama_apply_rotary_pos_emb,
)

from tests.models.test_llama import orig_llama  # noqa


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int8]
)
@pytest.mark.parametrize("use_complex", [False, True])
def test_rope(dtype, use_complex, orig_llama) -> None:  # noqa
    torch.manual_seed(1)
    batch_size = 1
    seq_len = 7
    n_head = 2
    n_embd = 8

    x = torch.randint(
        0, 10000, size=(batch_size, n_head, seq_len, n_embd // n_head)
    ).to(dtype)
    rotary = LlamaRotaryEmbedding(dim_model=n_embd // n_head, use_complex=use_complex)
    rotary._update_cos_sin_tables(x, seq_dimension=-2)

    freqs_cis = orig_llama.precompute_freqs_cis(n_embd // n_head, seq_len)

    torch.testing.assert_close(
        freqs_cis, torch.view_as_complex(rotary._cache.squeeze())
    )

    x_rope = llama_apply_rotary_pos_emb(x, rotary._cache)

    orig_llama_x_rope, _ = orig_llama.apply_rotary_emb(
        x.transpose(1, 2), x.transpose(1, 2), freqs_cis
    )
    torch.testing.assert_close(x_rope, orig_llama_x_rope.transpose(1, 2))
