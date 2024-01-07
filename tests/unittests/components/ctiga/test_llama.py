import pytest
import torch
import pytorch_lightning as pl

skip_test = False
try:
    from samantha.models.ctiga.gpt import GPTLMHeadModel
    from samantha.models.flash_llama import LlamaForCausalLM
    from transformers import LlamaConfig
    from samantha.models.ctiga.llama import remap_state_dict_huggingface_llama, flash_llama_config_to_gpt_config
    from samantha.models.ctiga_llama import create_ctiga_from_flash_llama
except Exception:
    skip_test = True

pl.seed_everything(0)

has_cuda = torch.cuda.is_available()


@pytest.mark.skip
@pytest.mark.parametrize("num_hidden_layers", [1, 8, 24])
@pytest.mark.parametrize("intermediate_size", [256*17, 256*16+32])
def test_remap_flash_llama_to_ctiga_llama(
    num_hidden_layers,
    intermediate_size,
):
    config = LlamaConfig(
        vocab_size = 1,
        hidden_size = 1536,
        intermediate_size = intermediate_size,
        num_hidden_layers = num_hidden_layers,
        num_attention_heads = 12,
        hidden_act = "silu",
        max_position_embeddings = 4096,
        initializer_range = 0.02,
        rms_norm_eps = 1e-6,
        use_cache = False,
        num_logits = 32_768 + 2,
    )

    remap_ctiga_state_dict = remap_state_dict_huggingface_llama(
        LlamaForCausalLM(config).state_dict(), 
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size,
    )

    kwargs = {
        "multiple_of" : 32
        } 
    if hasattr(config, "num_logits"):
        kwargs["num_logits"] = getattr(config, "num_logits")
    ctiga_state_dict = GPTLMHeadModel(
        flash_llama_config_to_gpt_config(config, **kwargs)).state_dict()

    # print("remap_ctiga_state_dict")
    # for key in remap_ctiga_state_dict:
    #     print(f"{key} -> {remap_ctiga_state_dict[key].shape}")
    # print("ctiga_state_dict")
    # for key in ctiga_state_dict:
    #     if "rotary_emb" not in key:
    #         print(f"{key} -> {ctiga_state_dict[key].shape}")


    for key in remap_ctiga_state_dict:
        assert key in ctiga_state_dict
        assert remap_ctiga_state_dict[key].shape == ctiga_state_dict[key].shape
        # print(f"{key} -> {ctiga_state_dict[key].shape}")


@pytest.mark.skip
@pytest.mark.parametrize("num_hidden_layers", [1, 8, 24])
@pytest.mark.parametrize("intermediate_size", [256*17, 256*16+32])
def test_sanity_flash_llama_to_ctiga_llama(
    num_hidden_layers,
    intermediate_size,
    dtype = torch.float16,
    device = "cuda",
):
    config = LlamaConfig(
        vocab_size = 1,
        hidden_size = 1536,
        intermediate_size = intermediate_size,
        num_hidden_layers = num_hidden_layers,
        num_attention_heads = 12,
        hidden_act = "silu",
        max_position_embeddings = 4096,
        initializer_range = 0.02,
        rms_norm_eps = 1e-6,
        use_cache = False,
        num_logits = 32_768 + 2,
    )

    flash_model = LlamaForCausalLM(config).to(dtype)
    flash_model.to(device).eval()

    ctiga_model = create_ctiga_from_flash_llama(flash_model)
    ctiga_model.to(dtype).to(device).eval()


    batch_size = 2
    max_seqlen = 256
    seqlens = torch.randint(
        max_seqlen // 2, max_seqlen + 1, (batch_size,), device=device
    )
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    with torch.no_grad():
        flash_logits = flash_model(input_ids=input_ids).logits
        ctiga_logits = ctiga_model(input_ids=input_ids).logits
    
    diff = (flash_logits - ctiga_logits).abs()
    print(f"Output max diff: {diff.max().item()}")
    print(f"Output mean diff: {diff.mean().item()}")
    assert diff.mean() < 5e-3 and diff.max() < 1e-2





