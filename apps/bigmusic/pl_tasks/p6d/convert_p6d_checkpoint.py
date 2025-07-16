import json
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import torch
import transformers
from seed_models.utils import logging
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .configuration_p6d import P6DenseConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


def convert_p6_dense_to_xperf_compatible(state_dict):
    """Convert state dict to xperf compatible p4 layout

    Args:
        state_dict (OrderedDict): P4ForCausalLM pretrained state dict

    Returns:
        dict: Converted state_dict
    """
    assert isinstance(state_dict, dict)
    converted_weights = {}

    # attention_bias = True, attention_out_bias = False -> padding o bias
    pad_o_proj_bias = ("model.layers.0.self_attn.q_proj.bias" in state_dict) and (
        "model.layers.0.self_attn.o_proj.bias" not in state_dict
    )

    for k, v in state_dict.items():
        origin_key = "gpt." + k.replace("model.", "transformer.").replace("layers.", "h.")

        if "q_proj.weight" in k:
            q_key = k
            k_key = k.replace("q_proj.weight", "k_proj.weight")
            v_key = k.replace("q_proj.weight", "v_proj.weight")
            origin_key = origin_key.replace("self_attn.q_proj.weight", "attn.c_attn.weight")

            q_w = state_dict[q_key]
            k_w = state_dict[k_key]
            v_w = state_dict[v_key]
            converted_weights[origin_key] = torch.cat((q_w, k_w, v_w), dim=0).transpose(-2, -1).contiguous()

            logger.debug(
                f"🔁 Merge {q_key, q_w.shape}, {k_key, k_w.shape}, {v_key, v_w.shape} to "
                "{origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "q_proj.bias" in k:
            q_key = k
            k_key = k.replace("q_proj.bias", "k_proj.bias")
            v_key = k.replace("q_proj.bias", "v_proj.bias")
            origin_key = origin_key.replace("self_attn.q_proj.bias", "attn.c_attn.bias")

            q_b = state_dict[q_key]
            k_b = state_dict[k_key]
            v_b = state_dict[v_key]
            converted_weights[origin_key] = torch.cat((q_b, k_b, v_b), dim=0).contiguous()
            logger.debug(
                f"🔁 Merge {q_key, q_b.shape}, {k_key, k_b.shape}, {v_key, v_b.shape} to"
                " {origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "k_proj" in k or "v_proj" in k:
            # pass k_proj and v_proj because they are already merged in c_attn
            pass
        elif "q_norm.weight" in k:
            origin_key = origin_key.replace("self_attn.q_norm.weight", "attn.q_norm.weight")
            converted_weights[origin_key] = state_dict[k]
        elif "k_norm.weight" in k:
            origin_key = origin_key.replace("self_attn.k_norm.weight", "attn.k_norm.weight")
            converted_weights[origin_key] = state_dict[k]
        elif "self_attn.o_proj.weight" in k:
            origin_key = origin_key.replace("self_attn.o_proj.weight", "attn.c_proj.weight")
            converted_weights[origin_key] = state_dict[k].transpose(-2, -1).contiguous()
            logger.debug(
                f"🔁 Convert back {k, state_dict[k].shape} to "
                f"{origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
            if pad_o_proj_bias:
                bias_key = origin_key.replace(".weight", ".bias")
                converted_weights[bias_key] = torch.zeros(state_dict[k].shape[0], dtype=torch.bfloat16)
                logger.debug(
                    f"🔁 Have self_attn_q/k/v_proj.bias but no self_attn.o_proj.bias, pad 0, set to {bias_key} ⚠️"
                )

        elif "self_attn.o_proj.bias" in k:
            origin_key = origin_key.replace("self_attn.o_proj.bias", "attn.c_proj.bias")
            converted_weights[origin_key] = state_dict[k]
            logger.debug(
                f"🔁 Convert back {k, state_dict[k].shape} to "
                f"{origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "mlp.gate_proj.weight" in k:
            origin_key = origin_key.replace("gate_proj", "c_fc")
            converted_weights[origin_key] = state_dict[k].transpose(-2, -1).contiguous()
            logger.debug(
                f"🔁 Convert back {k, state_dict[k].shape} to "
                f"{origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "mlp.up_proj.weight" in k:
            origin_key = origin_key.replace("up_proj", "c_fc.swiglu")
            converted_weights[origin_key] = state_dict[k].transpose(-2, -1).contiguous()
            logger.debug(
                f"🔁 Convert back {k, state_dict[k].shape} to "
                f"{origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "mlp.down_proj.weight" in k:
            origin_key = origin_key.replace("down_proj", "c_proj")
            converted_weights[origin_key] = state_dict[k].transpose(-2, -1).contiguous()
            logger.debug(
                f"🔁 Convert back {k, state_dict[k].shape} to "
                f"{origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "embed_tokens" in k:
            origin_key = origin_key.replace("embed_tokens", "wte")
            converted_weights[origin_key] = state_dict[k]
            logger.debug(
                f"🔁 Convert back {k, v.shape} to {origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "lm_head.weight" in k:
            converted_weights[origin_key] = state_dict[k]
            logger.debug(
                f"🔁 Convert back {k, v.shape} to {origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "input_layernorm" in k:
            origin_key = origin_key.replace("input_layernorm", "ln_1")
            converted_weights[origin_key] = state_dict[k]
            logger.debug(
                f"🔁 Convert back {k, v.shape} to {origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "post_attention_layernorm" in k:
            origin_key = origin_key.replace("post_attention_layernorm", "ln_2")
            converted_weights[origin_key] = state_dict[k]
            logger.debug(
                f"🔁 Convert back {k, v.shape} to {origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "norm" in k:
            origin_key = origin_key.replace("norm", "ln_f")
            converted_weights[origin_key] = state_dict[k]
            logger.debug(
                f"🔁 Convert back {k, v.shape} to {origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        elif "score" in k:
            origin_key = "gpt." + k.replace("score", "score_head")
            converted_weights[origin_key] = state_dict[k]
            logger.debug(
                f"🔁 Convert back {k, v.shape} to {origin_key, converted_weights[origin_key].shape} successfully! ✅"
            )
        else:
            raise ValueError(f"Unknown key name: {k}")

    logger.debug("Convert P6-dense to xperf compatible ")

    return converted_weights


def p6_dense_get_temporary_xperf_config(config):
    """Create temporary config.json for xperf.
    Note that most configs are converted from model, however, still exist some
    manually overwritten values.

    Args:
        config (P6Config): P6Config
    """
    if config.attention_bias is False:
        assert config.attention_out_bias is False, "qkv bias is False, but o bias is True is not supported"

    xperf_config = {
        "model_name": "SeedLLaMAForCausalLM",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "embed_dim": config.hidden_size,
        "hidden_size": config.hidden_size,
        "ffn_internal_dim": config.intermediate_size,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "num_layers": config.num_hidden_layers,
        "head_dim": config.head_dim,
        "gqa_weights_layout": "AABB",
        "quant_mode": "NO_QUANT",
        "is_meta": True,
        "dtype": "bfloat16",
        "has_mlp_bias": config.mlp_bias,
        "has_attn_bias": config.attention_bias,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_base": int(config.rope_theta),
        "rope_mode": config.rope_scaling.get("rope_type", "default") if config.rope_scaling else None,
        "rope_scale": int(config.rope_scaling.get("factor", 1)) if config.rope_scaling else None,
    }
    if config.use_qk_rmsnorm:
        xperf_config["querynorm"] = True
        xperf_config["keynorm"] = True
    tmp_dir = tempfile.gettempdir()
    tmp_config_file = os.path.join(tmp_dir, "p6dense_config.json")
    with open(tmp_config_file, "w") as fd:
        json.dump(xperf_config, fd)

    logger.info(f"Write xperf config {xperf_config} to {tmp_config_file}")
    return tmp_config_file


def p6_dense_convert_pt_to_hf(
    pt_path: Optional[str],
    pt_state_dict: Optional[dict] = None,
    hf_path: str = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    safe_serialization: bool = False,
    return_dict: bool = False,
    language_model_prefix: str = "",
):
    config = P6DenseConfig.from_pretrained(hf_path)
    logger.info(f"Model config: {config}")

    hidden_size = config.hidden_size
    query_head_scale_factor = config.query_head_scale_factor
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads

    q_size = num_heads * head_dim * query_head_scale_factor
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    logger.info(f"q_size: {q_size}")
    logger.info(f"k_size: {k_size}")
    logger.info(f"v_size: {v_size}")

    logger.info("Load Model with mmap = True")
    start_time = time.time()
    if pt_state_dict is None:
        pt_state_dict: Dict[str, Any] = torch.load(pt_path, mmap=True)
    end_time = time.time()
    logger.info(f"Loading time with mmap={end_time - start_time}")

    hf_state_dict = {}
    for pt_name, pt_tensor in pt_state_dict.items():
        if not isinstance(pt_tensor, torch.Tensor):
            logger.info(f"{pt_name} in the model dict is not torch.Tensor, skip it")
            continue

        # rename prefix
        if pt_name.startswith("gpt."):
            hf_name = pt_name.replace("gpt.", "")
            if hf_name.startswith("transformer."):
                hf_name = hf_name.replace("transformer.", "model.")
            if "h." in hf_name:
                hf_name = hf_name.replace("h.", "layers.")
            hf_name = language_model_prefix + hf_name
        elif pt_name.startswith("score_head"):
            # omnistore rm does not start with gpt.
            hf_name = pt_name
        else:
            continue

        if "lm_head.weight" in hf_name:
            hf_state_dict[hf_name] = pt_tensor.clone()
        elif "wte.weight" in hf_name:
            hf_name = hf_name.replace("wte", "embed_tokens")
            hf_state_dict[hf_name] = pt_tensor.clone()
        elif "mlp.c_fc.swiglu" in hf_name:
            hf_name = hf_name.replace("c_fc.swiglu", "up_proj")
            if "weight" in hf_name:
                hf_state_dict[hf_name] = pt_tensor.clone().transpose(0, 1).contiguous()
            elif 'bias" in hf_name:':
                # skip this because it is not used in Llama style mlp
                if config.mlp_bias is True:
                    hf_state_dict[hf_name] = pt_tensor.clone()
        elif "mlp.c_fc" in hf_name:
            hf_name = hf_name.replace("c_fc", "gate_proj")
            if "weight" in hf_name:
                hf_state_dict[hf_name] = pt_tensor.clone().transpose(0, 1).contiguous()
            elif 'bias" in hf_name:':
                # skip this because it is not used in Llama style mlp
                if config.mlp_bias is True:
                    hf_state_dict[hf_name] = pt_tensor.clone()
        elif "mlp.c_proj" in hf_name:
            hf_name = hf_name.replace("c_proj", "down_proj")
            if "weight" in hf_name:
                hf_state_dict[hf_name] = pt_tensor.clone().transpose(0, 1).contiguous()
            elif 'bias" in hf_name:':
                # skip this because it is not used in Llama style mlp
                if config.mlp_bias is True:
                    hf_state_dict[hf_name] = pt_tensor.clone()
        elif "attn.c_proj" in hf_name:
            hf_name = hf_name.replace("attn.c_proj", "self_attn.o_proj")
            if "weight" in hf_name:
                hf_state_dict[hf_name] = pt_tensor.clone().transpose(0, 1).contiguous()
            else:
                hf_state_dict[hf_name] = pt_tensor.clone()
        elif "attn.c_attn" in hf_name:
            if "weight" in hf_name:
                q_tensor, k_tensor, v_tensor = torch.split(pt_tensor, (q_size, k_size, v_size), dim=1)
            else:
                q_tensor, k_tensor, v_tensor = torch.split(pt_tensor, (q_size, k_size, v_size), dim=0)

            q_name = hf_name.replace("attn.c_attn", "self_attn.q_proj")
            k_name = hf_name.replace("attn.c_attn", "self_attn.k_proj")
            v_name = hf_name.replace("attn.c_attn", "self_attn.v_proj")

            if "weight" in hf_name:
                hf_state_dict[q_name] = q_tensor.clone().transpose(0, 1).contiguous()
                hf_state_dict[k_name] = k_tensor.clone().transpose(0, 1).contiguous()
                hf_state_dict[v_name] = v_tensor.clone().transpose(0, 1).contiguous()
            else:
                hf_state_dict[q_name] = q_tensor.clone()
                hf_state_dict[k_name] = k_tensor.clone()
                hf_state_dict[v_name] = v_tensor.clone()

            hf_name = ",".join(hf_name.replace("c_attn", f"{qkv}_proj") for qkv in ["q", "k", "v"])
        elif "ln_1" in hf_name:
            hf_name = hf_name.replace("ln_1", "input_layernorm")
            hf_state_dict[hf_name] = pt_tensor.clone()
        elif "ln_2" in hf_name:
            hf_name = hf_name.replace("ln_2", "post_attention_layernorm")
            hf_state_dict[hf_name] = pt_tensor.clone()
        elif "ln_f" in hf_name:
            hf_name = hf_name.replace("ln_f", "norm")
            hf_state_dict[hf_name] = pt_tensor.clone()
        elif "score_head" in hf_name:
            hf_name = hf_name.replace("score_head", "score")
            hf_state_dict[hf_name] = pt_tensor.clone()
        else:
            raise ValueError(f"Unknown name: {pt_name}")

        logger.info(f"🔁 Convert {pt_name} to {hf_name} successfully! ✅")
        del pt_tensor

    if return_dict:
        return hf_state_dict

    if "ForCausalLM" in config.architectures[0]:
        AUTO_MODEL_CLS = AutoModelForCausalLM
    elif "ForSequenceClassification" in config.architectures[0]:
        AUTO_MODEL_CLS = AutoModelForSequenceClassification
        hf_state_dict.pop("lm_head.weight", None)
    elif "ForTokenClassification" in config.architectures[0]:
        AUTO_MODEL_CLS = AutoModelForTokenClassification
        hf_state_dict.pop("lm_head.weight", None)
    else:
        raise ValueError(f"Unknown model architectures: {config.architectures}")

    logger.info("Init model state_dict on meta device")
    with torch.device("meta"):
        # model: "PreTrainedModel" = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        model: "PreTrainedModel" = AUTO_MODEL_CLS.from_config(config, torch_dtype=torch_dtype)

    logger.info("Load new state_dict to model")
    output = model.load_state_dict(hf_state_dict, strict=True, assign=True)
    logger.info(output)
    logger.info("Load new state_dict to model successfully! ✅")

    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"{n} has not been loaded!"

    logger.info(f"Save model to {hf_path}")
    model.save_pretrained(hf_path, safe_serialization=safe_serialization)

    # check tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
    except Exception:
        tokenizer = None

    if tokenizer is not None:
        chat_template = (
            "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
            "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
            "{% set role = message['role'] %}{{ bos_token + role + '\n' + message['content'] | trim + eos_token }}"
            "{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\n'}}{% endif %}"
        )

        if tokenizer.chat_template is None:
            tokenizer.chat_template = chat_template

        logger.info(f"Save tokenizer to {hf_path}")
        tokenizer.save_pretrained(hf_path)


def convert_p6dense_config_from_cruise(
    cruise_config: dict,
    model_type: str = "P6Dense",
    auto_model: Literal["ForCausalLM", "ForSequenceClassification", "ForTokenClassification"] = "ForCausalLM",
    return_dict: bool = False,
):
    if "policy" in cruise_config["model"] and "network" in cruise_config["model"]["policy"]:
        network_config = cruise_config["model"]["policy"]["network"]
    elif "network" in cruise_config["model"]:
        network_config = cruise_config["model"]["network"]
    else:
        raise ValueError("Network config not found in cruise config")

    hf_config = {}
    hf_config["architectures"] = [f"{model_type}{auto_model}"]

    if cruise_config["data"].get("tokenizer_type", "bbpe") == "bbpe":
        hf_config["bos_token_id"] = 0
        hf_config["eos_token_id"] = 2
    else:
        hf_config["bos_token_id"] = 1
        hf_config["eos_token_id"] = 2

    hf_config["hidden_size"] = network_config["hidden_size"]

    hf_config["num_attention_heads"] = network_config["n_head"]
    hf_config["num_key_value_heads"] = network_config["n_head"] // network_config["n_shared_qhead"]

    hf_config["num_hidden_layers"] = network_config["n_layer"] - len(network_config["noop_transformer_layers"])

    hf_config["hidden_act"] = network_config["activation_function"]
    hf_config["initializer_range"] = network_config["initializer_range"]
    hf_config["max_position_embeddings"] = network_config["max_position_embeddings"]
    hf_config["model_type"] = f"seed_{model_type.lower()}"

    if network_config.get("sparse_attention_window_size") is not None and (
        network_config["sparse_attention_window_size"] != [-1]
    ):
        # remove -1 from sliding window, which is for noop transformer layers
        hf_config["sliding_window"] = list(filter(lambda x: x != -1, network_config["sparse_attention_window_size"]))

    hf_config["attention_dropout"] = network_config["attn_pdrop"]
    hf_config["resid_pdrop"] = network_config["resid_pdrop"]

    if network_config.get("layer_norm_type"):
        if network_config["layer_norm_type"] == "layernorm":
            hf_config["layer_norm_eps"] = network_config["layer_norm_epsilon"]
        elif network_config["layer_norm_type"] == "rmsnorm":
            hf_config["rms_norm_eps"] = network_config["layer_norm_epsilon"]
    else:
        hf_config["layer_norm_eps"] = network_config["layer_norm_epsilon"]

    if network_config.get("dense_ffn_type") == "swiglu":
        hf_config["hidden_act"] = "silu"
        hf_config["mlp_bias"] = False

    # Refer to `get_mlp_inner_dim` at
    # https://code.byted.org/seed/mariana/blob/pretrain_mariana_gqa_240617/mariana/models/text/config.py#L283
    def get_mlp_inner_dim(network_config: Dict[str, Any]) -> int:
        kv_dim = network_config["hidden_size"] // network_config["n_shared_qhead"]
        ffn_hidden_size = 4 * network_config["hidden_size"]
        # supply reduced weight from MQA to MLP
        default_inner_dim = ffn_hidden_size + (network_config["hidden_size"] - kv_dim)

        target_inner_dim = default_inner_dim

        if network_config["n_inner"] is not None and not (
            network_config["n_inner"] == ffn_hidden_size and network_config["n_inner"] != default_inner_dim
        ):
            target_inner_dim = network_config["n_inner"]

        logger.info(f"got target_inner_dim: {target_inner_dim}")
        return target_inner_dim

    hf_config["intermediate_size"] = get_mlp_inner_dim(network_config)

    if network_config.get("query_head_scale_factor"):
        hf_config["query_head_scale_factor"] = network_config["query_head_scale_factor"]

    if network_config.get("use_attention_bias"):
        hf_config["attention_bias"] = network_config["use_attention_bias"]

    if network_config.get("rope_mode"):
        hf_config["rope_scaling"] = {}
        hf_config["rope_scaling"]["rope_type"] = network_config["rope_mode"]
        if network_config.get("rope_scale") is not None:
            hf_config["rope_scaling"]["factor"] = float(network_config["rope_scale"])

    hf_config["rope_theta"] = network_config.get("rope_base", 10000)

    if network_config.get("tie_weight") is not None:
        hf_config["tie_word_embeddings"] = network_config["tie_weight"]

    dtype_mapping = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }

    if cruise_config.get("merge_ckpt_dtype"):
        hf_config["torch_dtype"] = dtype_mapping[cruise_config["merge_ckpt_dtype"]]
    else:
        hf_config["torch_dtype"] = "bfloat16"

    hf_config["transformers_version"] = transformers.__version__
    hf_config["use_cache"] = True
    hf_config["vocab_size"] = network_config["vocab_size"]

    # note for both SequenceClassification (RM) and TokenClassification (Value),
    # the num_labels is 1. We may change this later.
    if auto_model in ["ForSequenceClassification", "ForTokenClassification"]:
        hf_config["num_labels"] = 1

    return hf_config
