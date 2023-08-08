# Copyright (c) 2023, Tri Dao.

import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2Config, LlamaConfig, LlamaForCausalLM


def update_state_dict(state_dict, name, data):
    state_dict.update({name: data})


def remap_state_dict_ctiga_llama_to_hf(state_dict, n_layers, n_heads, n_embd):
    def permute(w):
        return (
            w.view(n_heads, n_embd // n_heads // 2, 2, n_embd)
            .transpose(1, 2)
            .reshape(n_embd, n_embd)
        )

    mapped_state_dict = OrderedDict()
    # word_embeddings
    # embed_tokens -> embeddings.word_embeddings
    update_state_dict(
        mapped_state_dict,
        "model.embed_tokens.weight",
        state_dict.pop("transformer.embeddings.word_embeddings.weight"),
    )
    # norm -> ln_f
    update_state_dict(
        mapped_state_dict,
        "model.norm.weight",
        state_dict.pop("transformer.ln_f.weight"),
    )
    # lm_head -> lm_head
    update_state_dict(
        mapped_state_dict, "lm_head.weight", state_dict.pop("lm_head.weight")
    )

    to_prefix = "model.layers.{}.{}"
    from_prefix = "transformer.layers.{}.{}"
    for layer_idx in range(n_layers):
        # RMS-Norm
        # input_layernorm -> norm1
        update_state_dict(
            mapped_state_dict,
            f"{to_prefix.format(layer_idx,'input_layernorm')}.weight",
            state_dict.pop(f"{from_prefix.format(layer_idx,'norm1')}.weight"),
        )
        # post_attention_layernorm -> norm2
        update_state_dict(
            mapped_state_dict,
            f"{to_prefix.format(layer_idx,'post_attention_layernorm')}.weight",
            state_dict.pop(f"{from_prefix.format(layer_idx,'norm2')}.weight"),
        )

        # attention block
        from_ = from_prefix.format(layer_idx, "mixer")
        to_ = to_prefix.format(layer_idx, "self_attn")
        # self_attn.rotary_emb.inv_freq -> mixer.rotary_emb.inv_freq (only in fp32), use default
        # update_state_dict(
        #     mapped_state_dict, f"{to_}.rotary_emb.inv_freq", state_dict.pop(f"{from_}.rotary_emb.inv_freq").float()
        # )

        # [self_attn.q_proj,self_attn.k_proj,self_attn.v_proj] -> mixer.Wqkv
        for k, w in zip(
            ["q", "k", "v"], state_dict.pop(f"{from_}.Wqkv.weight").chunk(3, dim=0)
        ):
            if k in ["q", "k"]:
                w = permute(w)
            update_state_dict(mapped_state_dict, f"{to_}.{k}_proj.weight", w)
        # self_attn.o_proj -> mixer.out_proj
        update_state_dict(
            mapped_state_dict,
            f"{to_}.o_proj.weight",
            state_dict.pop(f"{from_}.out_proj.weight"),
        )

        # mlp
        # [mlp.gate_proj,mlp.up_proj] -> mlp.fc1
        from_ = from_prefix.format(layer_idx, "mlp")
        to_ = to_prefix.format(layer_idx, "mlp")
        for k, w in zip(
            ["up", "gate"], state_dict.pop(f"{from_}.fc1.weight").chunk(2, dim=0)
        ):
            update_state_dict(mapped_state_dict, f"{to_}.{k}_proj.weight", w)
        update_state_dict(
            mapped_state_dict,
            f"{to_}.down_proj.weight",
            state_dict.pop(f"{from_}.fc2.weight"),
        )
    return mapped_state_dict


def remap_state_dict_huggingface_llama(state_dict, n_layers, n_heads, n_embd):
    def permute(w):
        # return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)
        return (
            w.view(n_heads, 2, n_embd // n_heads // 2, n_embd)
            .transpose(1, 2)
            .reshape(n_embd, n_embd)
        )

    mapped_state_dict = OrderedDict()
    # word_embeddings
    # embed_tokens -> embeddings.word_embeddings
    update_state_dict(
        mapped_state_dict,
        "transformer.embeddings.word_embeddings.weight",
        state_dict.pop("model.embed_tokens.weight"),
    )
    # norm -> ln_f
    update_state_dict(
        mapped_state_dict,
        "transformer.ln_f.weight",
        state_dict.pop("model.norm.weight"),
    )
    # lm_head -> lm_head
    update_state_dict(
        mapped_state_dict, "lm_head.weight", state_dict.pop("lm_head.weight")
    )

    from_prefix = "model.layers.{}.{}"
    to_prefix = "transformer.layers.{}.{}"
    for layer_idx in range(n_layers):
        # RMS-Norm
        # input_layernorm -> norm1
        update_state_dict(
            mapped_state_dict,
            f"{to_prefix.format(layer_idx,'norm1')}.weight",
            state_dict.pop(f"{from_prefix.format(layer_idx,'input_layernorm')}.weight"),
        )
        # post_attention_layernorm -> norm2
        update_state_dict(
            mapped_state_dict,
            f"{to_prefix.format(layer_idx,'norm2')}.weight",
            state_dict.pop(
                f"{from_prefix.format(layer_idx,'post_attention_layernorm')}.weight"
            ),
        )

        # attention block
        from_ = from_prefix.format(layer_idx, "self_attn")
        to_ = to_prefix.format(layer_idx, "mixer")
        # self_attn.rotary_emb.inv_freq -> mixer.rotary_emb.inv_freq (only in fp32), use default
        # update_state_dict(
        #     mapped_state_dict, f"{to_}.rotary_emb.inv_freq", state_dict.pop(f"{from_}.rotary_emb.inv_freq").float()
        # )

        # [self_attn.q_proj,self_attn.k_proj,self_attn.v_proj] -> mixer.Wqkv
        q, k, v, o = [
            state_dict.pop(f"{from_}.{k}_proj.weight") for k in ["q", "k", "v", "o"]
        ]
        q = permute(q)
        k = permute(k)
        update_state_dict(
            mapped_state_dict, f"{to_}.Wqkv.weight", torch.cat([q, k, v], dim=0)
        )
        # self_attn.o_proj -> mixer.out_proj
        update_state_dict(mapped_state_dict, f"{to_}.out_proj.weight", o)

        # mlp
        # [mlp.gate_proj,mlp.up_proj] -> mlp.fc1
        from_ = from_prefix.format(layer_idx, "mlp")
        to_ = to_prefix.format(layer_idx, "mlp")
        update_state_dict(
            mapped_state_dict,
            f"{to_}.fc1.weight",
            torch.cat(
                [state_dict.pop(f"{from_}.{k}_proj.weight") for k in ["up", "gate"]],
                dim=0,
            ),
        )
        update_state_dict(
            mapped_state_dict,
            f"{to_}.fc2.weight",
            state_dict.pop(f"{from_}.down_proj.weight"),
        )
    return mapped_state_dict


def remap_state_dict_byteformer_llama(state_dict, n_layers, n_heads, n_embd):
    def permute(w):
        return (
            w.view(n_heads, 2, n_embd // n_heads // 2, n_embd)
            .transpose(1, 2)
            .reshape(n_embd, n_embd)
        )

    mapped_state_dict = OrderedDict()
    # word_embeddings
    # embed_tokens -> embeddings.word_embeddings
    update_state_dict(
        mapped_state_dict,
        "transformer.embeddings.word_embeddings.weight",
        state_dict.pop("transformer.wte.weight"),
    )
    # norm -> ln_f
    update_state_dict(
        mapped_state_dict,
        "transformer.ln_f.weight",
        state_dict.pop("transformer.ln_f.weight"),
    )
    # lm_head -> lm_head
    update_state_dict(
        mapped_state_dict, "lm_head.weight", state_dict.pop("lm_head.weight")
    )

    # attention block
    from_prefix = "transformer.h.{}.{}"
    to_prefix = "transformer.layers.{}.{}"
    for layer_idx in range(n_layers):
        # RMS-Norm
        # input_layernorm -> norm1
        update_state_dict(
            mapped_state_dict,
            f"{to_prefix.format(layer_idx,'norm1')}.weight",
            state_dict.pop(f"{from_prefix.format(layer_idx,'ln_1')}.weight"),
        )
        # post_attention_layernorm -> norm2
        update_state_dict(
            mapped_state_dict,
            f"{to_prefix.format(layer_idx,'norm2')}.weight",
            state_dict.pop(f"{from_prefix.format(layer_idx,'ln_2')}.weight"),
        )

        from_ = from_prefix.format(layer_idx, "attn")
        to_ = to_prefix.format(layer_idx, "mixer")

        # [self_attn.q_proj,self_attn.k_proj,self_attn.v_proj] -> mixer.Wqkv
        q, k, v = [state_dict.pop(f"{from_}.to_{k}.weight") for k in ["q", "k", "v"]]
        # q = permute(q)
        # k = permute(k)
        update_state_dict(
            mapped_state_dict, f"{to_}.Wqkv.weight", torch.cat([q, k, v], dim=0)
        )
        # self_attn.o_proj -> mixer.out_proj
        update_state_dict(
            mapped_state_dict,
            f"{to_}.out_proj.weight",
            state_dict.pop(f"{from_}.WO.weight"),
        )

        # mlp
        # [mlp.gate_proj,mlp.up_proj] -> mlp.fc1
        from_ = from_prefix.format(layer_idx, "mlp")
        to_ = to_prefix.format(layer_idx, "mlp")
        update_state_dict(
            mapped_state_dict,
            f"{to_}.fc1.weight",
            torch.cat(
                [state_dict.pop(f"{from_}.{k}.weight") for k in ["c_fc2", "c_fc1"]],
                dim=0,
            ),
        )
        update_state_dict(
            mapped_state_dict,
            f"{to_}.fc2.weight",
            state_dict.pop(f"{from_}.c_proj.weight"),
        )
    return mapped_state_dict


def remap_state_dict_meta_llama(state_dict, config):
    def key_mapping_layers(key):
        return f"transformer.{key}" if not key.startswith("output.") else key

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        return re.sub(
            r"^transformer.tok_embeddings.",
            "transformer.embeddings.word_embeddings.",
            key,
        )

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple)
        * pad_vocab_size_multiple
    )
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict[
            "transformer.embeddings.word_embeddings.weight"
        ]
    else:
        output_embeddings = state_dict.pop("output.weight")
        # Need to recompute vocab_size since LLaMa shards the word embeddings and output embeddings
        # differently.
        vocab_size = (
            math.ceil(output_embeddings.shape[0] / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict["lm_head.weight"] = F.pad(
            output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0])
        )

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.norm.", r"transformer.ln_f.", key)
        key = re.sub(
            r"^transformer.layers.(\d+).attention_norm.",
            r"transformer.layers.\1.norm1.",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).ffn_norm.", r"transformer.layers.\1.norm2.", key
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for l in range(config.n_layer):
        w1 = state_dict.pop(f"transformer.layers.{l}.feed_forward.w1.weight")
        w3 = state_dict.pop(f"transformer.layers.{l}.feed_forward.w3.weight")
        # Our ordering is different
        state_dict[f"transformer.layers.{l}.mlp.fc1.weight"] = torch.cat(
            [w3, w1], dim=0
        )

    def key_mapping_mlp(key):
        return re.sub(
            r"^transformer.layers.(\d+).feed_forward.w2.",
            r"transformer.layers.\1.mlp.fc2.",
            key,
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f"transformer.layers.{l}.attention.wq.weight")
        Wk = state_dict.pop(f"transformer.layers.{l}.attention.wk.weight")
        Wv = state_dict.pop(f"transformer.layers.{l}.attention.wv.weight")
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat(
            [Wq, Wk, Wv], dim=0
        )
        # We don't store these
        state_dict.pop(
            f"transformer.layers.{l}.attention.inner_attention.rope.freqs", None
        )

    def key_mapping_attn(key):
        return re.sub(
            r"^transformer.layers.(\d+).attention.wo.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def config_from_checkpoint(checkpoint_path: str, model_name: str) -> LlamaConfig:
    """Load a LlamaConfig from a checkpoint path."""
    with open(Path(checkpoint_path) / model_name / "params.json") as f:
        params = json.load(f)
    config = LlamaConfig(
        hidden_size=params["dim"],
        intermediate_size=None,
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
    )
    return config


def state_dicts_from_checkpoint(checkpoint_path: str, model_name: str) -> dict:
    # Need to sort, otherwise we mess up the ordering and the weights are wrong
    return [
        torch.load(path, map_location="cpu")
        for path in sorted(
            (Path(checkpoint_path) / model_name).glob("consolidated.*.pth")
        )
    ]


def llama_config_to_gpt2_config(
    llama_config: LlamaConfig,
    use_flash_attn=True,
    fused_bias_fc=True,
    fused_mlp=False,
    fused_dropout_add_ln=True,
    residual_in_fp32=True,
    rotary_emb_compat="default",
    flashattn_version=1,
    **kwargs,
) -> GPT2Config:
    assert rotary_emb_compat in ["default", "byteformer"]
    return GPT2Config(
        vocab_size=llama_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,  # align with byteformer
        rotary_emb_compat=rotary_emb_compat,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        use_flash_attn=use_flash_attn,
        fused_bias_fc=fused_bias_fc,
        fused_mlp=fused_mlp,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=residual_in_fp32,
        flashattn_version=flashattn_version,
        **kwargs,
    )


def get_n_inner_dim(n_embd: int, multiple_of: int = 256):
    n_inner = 4 * n_embd
    n_inner = int(2 * n_inner / 3)
    N = multiple_of
    return ((n_inner - 1) // N) * N + N


def byteformer_llama_config_to_gpt_config(
    byteformer_llama_config,
    use_flash_attn=True,
    fused_bias_fc=True,
    fused_mlp=False,
    fused_dropout_add_ln=True,
    residual_in_fp32=True,
    flashattn_version=1,
    **kwargs,
):
    return llama_config_to_gpt2_config(
        LlamaConfig(
            vocab_size=byteformer_llama_config.vocab_size,
            hidden_size=byteformer_llama_config.n_embd,
            intermediate_size=get_n_inner_dim(byteformer_llama_config.n_embd),
            num_hidden_layers=byteformer_llama_config.n_layer,
            num_attention_heads=byteformer_llama_config.n_head,
            initializer_range=byteformer_llama_config.initializer_range,
            rms_norm_eps=byteformer_llama_config.rms_norm_epsilon,
        ),
        use_flash_attn=use_flash_attn,
        fused_bias_fc=fused_bias_fc,
        fused_mlp=fused_mlp,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=residual_in_fp32,
        rotary_emb_compat="byteformer",
        flashattn_version=flashattn_version,
        **kwargs,
    )


def gpt2_config_to_llama_config(gpt2_config: GPT2Config):
    return LlamaConfig(
        vocab_size=gpt2_config.vocab_size,
        hidden_size=gpt2_config.n_embd,
        intermediate_size=gpt2_config.n_inner or get_n_inner_dim(gpt2_config.n_embd),
        num_attention_heads=gpt2_config.n_head,
        num_hidden_layers=gpt2_config.n_layer,
        hidden_act="silu",
        rms_norm_eps=gpt2_config.layer_norm_epsilon,
        bos_token_id=gpt2_config.bos_token_id,
        eos_token_id=gpt2_config.eos_token_id,
        pad_token_id=gpt2_config.pad_token_id,
        tie_word_embeddings=False,
    )


def save_as_hf_llama_model(model, save_path):
    gpt2_config = model.config
    assert isinstance(gpt2_config, GPT2Config)

    hf_pretrained_state_dict = remap_state_dict_ctiga_llama_to_hf(
        model.state_dict(), gpt2_config.n_layer, gpt2_config.n_head, gpt2_config.n_embd
    )
    llama_config = gpt2_config_to_llama_config(gpt2_config)
    model_hf = LlamaForCausalLM(llama_config)
    model_hf.load_state_dict(hf_pretrained_state_dict, strict=False)
    model_hf.save_pretrained(save_path)

    return model_hf
