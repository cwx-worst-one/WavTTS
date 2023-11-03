# coding=utf-8

import json
import os
import tempfile

import yaml
from cruise import CruiseConfig
from cruise.utilities.hdfs_io import hcopy, hexists, hglob, hisdir, hopen
from rich import print as rprint
from transformers import AutoTokenizer

from mariana.data.gpt.tokenization import CasterTokenizer

DefaultGenerationConfig = {
    "is_encoder_decoder": False,
    "max_length": 1024,
    "max_new_tokens": 896,
    "min_length": 0,
    "do_sample": True,
    "early_stopping": False,
    "num_beams": 1,
    "num_beam_groups": 1,
    "temperature": 1.0,
    "penalty_alpha": None,
    "top_k": None,
    "top_p": 0.7,
    "typical_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "force_words_ids": None,
    "bos_token_id": None,
    "pad_token_id": None,
    "eos_token_id": None,
    "use_cache": True,
    "renormalize_logits": False,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "encoder_no_repeat_ngram_size": 0,
    "num_return_sequences": 1,
    "max_time": None,
    "decoder_start_token_id": None,
    "diversity_penalty": None,
    "prefix_allowed_tokens_fn": None,
    "logits_processor": None,
    "stopping_criteria": None,
    "constraints": None,
    "output_attentions": False,
    "output_hidden_states": False,
    "output_scores": False,
    "return_dict_in_generate": True,
    "remove_invalid_values": None,
    "synced_gpus": False,
    "exponential_decay_length_penalty": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "forced_decoder_ids": None,
}

DefaultModelConfig = {
    "hidden_size": 2048,
    "n_embed": 2048,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 64000,
    "max_position_embeddings": 2048,
    "layer_norm_epsilon": 1.0e-5,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0,
    "attn_pdrop": 0.1,
    "scale_attn_weights": True,  # TODO:
    "scale_attn_by_inverse_layer_idx": False,  # TODO:
    "reorder_and_upcast_attn": False,  # TODO:
    "initializer_range": 0.013975424859373685,
    "gradient_checkpointing": False,
    "tie_weight": True,
    "pad_idx": 1,
    "use_ft_flash_attn": False,
    "use_ft_linear": False,
    "use_ft_layernorm": False,
    "use_cache": True,
    "use_rmpad": False,
}

DefaultGeneratorConfig = {
    "trial_num": 1,
    "steps": 1024,
    "tempreature": 0.7,
    "do_sample": True,
    "omega": 0.3,
    "top_k": None,
    "top_p": 0.9,
    "n_eos": 1,
}

DefaultTokenizer = "hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase"  # noqa

DefaultTokenizerType = "caster"


def disable_xperf(model_config):
    model_config.network.use_ft_flash_attn = False
    model_config.network.use_ft_linear = False
    model_config.network.use_ft_layernorm = False
    return model_config


def is_caster_tokenizer(tokenizer):
    return isinstance(tokenizer, CasterTokenizer)


def create_tokenizer(tokenizer, tokenizer_type):
    if tokenizer.startswith("hdfs"):
        # try download it to local once per node and load it in setup
        tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(tokenizer))
        hcopy(tokenizer, tmp_dir)
        if tokenizer_type == "caster":
            tokenizer = CasterTokenizer.from_pretrained(tmp_dir, max_len=-1)
        elif tokenizer_type == "bbpe":
            tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
        else:
            raise NotImplementedError
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def setup_generation_config(tokenizer):
    generation_config = DefaultGenerationConfig
    generation_config["pad_token_id"] = tokenizer.pad_token_id
    generation_config["eos_token_id"] = tokenizer.eos_token_id
    if is_caster_tokenizer(tokenizer):
        generation_config["bos_token_id"] = 0
    else:
        generation_config["bos_token_id"] = tokenizer.bos_token_id
    generation_config = CruiseConfig(generation_config)
    return generation_config


def parse_model_dir(model_dir: str, step: str = None):
    with hopen(os.path.join(model_dir, "cruise_cli.json"), "r") as f:
        config = CruiseConfig(json.loads(f.read()))
    if step:
        checkpoint_dir = os.path.join(model_dir, "checkpoints", "global_step_" + step)
    elif hexists(os.path.join(model_dir, "checkpoints", "latest")):
        with hopen(os.path.join(model_dir, "checkpoints", "latest"), "r") as f:
            latest = f.read().strip()
            if not isinstance(latest, str):
                latest = latest.decode("utf-8")
        checkpoint_dir = os.path.join(model_dir, "checkpoints", latest)
    else:
        steps = hglob(os.path.join(model_dir, "checkpoints", "global_step_*"))
        latest = max(steps, key=lambda s: int(s.split("_")[-1]))
        checkpoint_dir = os.path.join(model_dir, "checkpoints", latest)

    # try to find state dict
    checkpoint = os.path.join(checkpoint_dir, "pytorch_model.bin")

    # if no state dict found, find merged zero3 state dict
    if not hexists(checkpoint):
        checkpoint = os.path.join(checkpoint_dir, "zero3_merge_states.pt")

    if not hexists(checkpoint):
        checkpoint = os.path.join(checkpoint_dir, "megatron_merge_states.pt")

    if not hexists(checkpoint):
        checkpoint = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")

    if not hexists(checkpoint):
        raise ValueError(f"Checkpoint {checkpoint} does not exist.")

    model_config = CruiseConfig(config["model"])
    data_config = CruiseConfig(config["data"])
    tokenizer = create_tokenizer(
        data_config["tokenizer"], data_config.get("tokenizer_type", "caster")
    )

    return model_config, checkpoint, tokenizer


def parse_args(FLAGS):
    """Prepare model, tokenizer, checkpoint"""

    # Default config
    model_config = CruiseConfig(DefaultModelConfig)
    generator_config = CruiseConfig(DefaultGeneratorConfig)
    if FLAGS.tokenizer and FLAGS.tokenizer_type:
        tokenizer = create_tokenizer(FLAGS.tokenizer, FLAGS.tokenizer_type)
    else:
        tokenizer = create_tokenizer(DefaultTokenizer, DefaultTokenizerType)
    checkpoint = FLAGS.checkpoint

    if hisdir(FLAGS.checkpoint):
        model_config, checkpoint, tokenizer = parse_model_dir(
            FLAGS.checkpoint, FLAGS.step
        )

    generator_config = setup_generation_config(tokenizer)

    if FLAGS.model:
        with hopen(FLAGS.model, "r") as f:
            model_config = CruiseConfig(yaml.safe_load(f))

    if FLAGS.generator:
        with hopen(FLAGS.generator, "r") as f:
            generator_config = CruiseConfig(yaml.safe_load(f))

    model_config = disable_xperf(model_config)
    model_config.network.gradient_checkpointing = False

    rprint("Model: ", json.dumps(model_config.asdict(), indent=2))
    rprint("Tokenizer: ", tokenizer)
    rprint("Generator: ", json.dumps(generator_config.asdict(), indent=2))

    return model_config, generator_config, checkpoint, tokenizer
