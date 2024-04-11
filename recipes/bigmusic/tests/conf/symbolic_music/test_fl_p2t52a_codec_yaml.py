import os

import torch

from recipes.bigmusic.tests.lightning.common import (
    load_yaml,
    test_bm_yaml_fit,
)

yaml_path = os.path.join(
    os.environ["DEPS_DIR"],
    "samantha/recipes/bigmusic/conf/symbolic_music/20240328.fl_p2t52a_codec.yaml"
)


def test_yaml_fit():
    cache_dir = os.path.join(
        os.environ["DUMP_DIR"],
        "ai_music/20240312.symbolic.dump/module_cache/bestrq"
    )
    os.environ["BYTED_RAY_CLUSTER"] = "a"
    test_bm_yaml_fit(yaml_path, extra_args=[
        "--run_opts.cache_dir", cache_dir,
    ])
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_gpt_model_debug():
    from samantha.models.ctiga.gpt import GPTLMHeadModel
    from transformers import GPT2Config
    model_config = GPT2Config(
        vocab_size = 0,
        max_position_embeddings = 0,
        use_cache = False,
        n_positions = 0,
        num_logits = 1000,
        n_embd = 768,
        n_layer = 12,
        n_head = 12,
        n_inner = 2048,
        use_rms_norm = True,
        activation_function = "swiglu",
        resid_pdrop = 0.0,
        embd_pdrop = 0.0,
        attn_pdrop = 0.0,
        layer_norm_epsilon = 1e-6,
        initializer_range = 0.02,
        rescale_prenorm_residual = False,
        rms_norm = True,
        rotary_emb_fraction = 1.0,
        rotary_emb_interleaved = True,
        rotary_emb_compat = "default",
        tie_word_embeddings = False,
        qkv_proj_bias = False,
        out_proj_bias = False,
        mlp_fc1_bias = False,
        mlp_fc2_bias = False,
        use_flash_attn = True,
        fused_bias_fc = True,
        fused_mlp = False,
        fused_dropout_add_ln = True,
        residual_in_fp32 = True,
    )
    model = GPTLMHeadModel(config=model_config)
    inputs_embeds = torch.rand((2, 2852, 512)).to("cuda:0")
    model.to("cuda:0")
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
    outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)


def test_gpt_yaml_debug():
    hparams = load_yaml(yaml_path)
    pl_module = hparams["pl_module"]
    model_config = hparams["model_config"]
    inputs_embeds = torch.rand((2, 2852, 512)).to("cuda:0")
    pl_module.model.to("cuda:0")
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        outputs = pl_module.model(inputs_embeds=inputs_embeds, output_hidden_states=True)