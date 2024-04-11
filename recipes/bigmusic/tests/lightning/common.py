import os


def get_default_model_cls(num_logits):
    from transformers import GPT2Config
    from samantha.models.ctiga.gpt import GPTLMHeadModel
    """gpu needed, but not a100
    """
    model_config = GPT2Config(
        vocab_size = 1,
        max_position_embeddings = 0,
        use_cache = False,
        n_positions = 0,
        num_logits = num_logits,
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
        use_flash_attn = False,
        fused_bias_fc = True,
        fused_mlp = False,
        fused_dropout_add_ln = True,
        residual_in_fp32 = True,
    )
    return lambda : GPTLMHeadModel(config=model_config)


def get_default_optimizer_cls():
    from torch.optim import AdamW

    return lambda : AdamW(
        lr = 1.0e-4,
        weight_decay = 0.01,
        betas = [0.9, 0.96],
        eps = 0.00000001,
    )


def get_default_scheduler_cls():
    from recipes.musiclm.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine
    return lambda : WarmupCosine(
        init_lr = 1.0e-4,
        warmup_steps = 2000,
        cycle_steps = 30000,
        min_lr = 1e-5,
    )


def get_default_criterion_cls():
    from recipes.musiclm.lightning.modules import MaskedCrossEntropy
    return MaskedCrossEntropy


def get_default_bestrq_required_module_dict(cache_dir):
    """To load the umm model, we need to set "BYTED_RAY_CLUSTER" to
    some string other than an empty string
    """
    from recipes.umm.requires.model_initializer import init_stage3
    os.environ["BYTED_RAY_CLUSTER"] = "a"
    return {
        "hpath": "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt",
        "initializer": lambda x, y: init_stage3(x, y, cache_dir=cache_dir),
    }


def get_default_semantic_module_extra_params():
    """Semantic module requires certain extra params to exist to run
    Not sure if hidden_size needs to be the same as n_embd in model_config
    """
    return {
        "hidden_size": 768,
        "semantic_codebook_size": 100,
    }


def load_yaml(yaml_path, extra_args=[]):
    from hyperpyyaml import load_hyperpyyaml
    from samantha.utils.parser import parse_arguments
    args = [
        "fit", "-c", yaml_path,
        "--run_opts.num_workers", "0",
        # "--model_config.use_flash_attn", "False",
        # "--model_config.fused_dropout_add_ln", "False",
        *extra_args,
        # "--run_opts.lyrics_tokenizer", "zh_wordpiece",
        # "--run_opts.language", "['EN']",
    ]
    hparams_file, run_opts, overrides = parse_arguments(args)

    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    return hparams


def test_bm_yaml_fit(yaml_path, extra_args=[]):
    hparams = load_yaml(yaml_path, extra_args)

    pl_module = hparams["pl_module"]
    pl_datamodule = hparams["pl_datamodule"]
    trainer = hparams["trainer"]

    metrics = trainer.fit(model=pl_module, datamodule=pl_datamodule)