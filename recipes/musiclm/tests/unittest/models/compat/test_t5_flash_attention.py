import torch
from pytorch_lightning import seed_everything
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Attention

from recipes.musiclm.models.compat.t5_flash_attention import T5FlashAttention


def test_flash_attention():
    batch_size = 1
    seq_len = 8
    n_embd = 32
    config = T5Config(d_model=n_embd, num_layers=1, num_heads=4)

    seed_everything(42)
    attn = T5Attention(config)

    seed_everything(42)
    flash_attn = T5FlashAttention(config)

    hidden_states = torch.randn(batch_size, seq_len, n_embd)

    seed_everything(42)
    outputs_attn = attn.forward(
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    )

    seed_everything(42)
    outputs_flash_attn = flash_attn.forward(
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    )

    attn_output, present_key_value_state, position_bias = outputs_attn
    fa_attn_output, fa_present_key_value_state, fa_position_bias = outputs_flash_attn

    assert torch.allclose(attn_output, fa_attn_output)
    assert torch.allclose(position_bias, fa_position_bias)

    if present_key_value_state is not None:
        assert torch.equal(present_key_value_state, fa_present_key_value_state)
