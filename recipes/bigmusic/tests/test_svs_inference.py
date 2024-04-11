from typing import Dict
import os
import time

import numpy as np
import torch
from transformers import GPT2Config
from tqdm import tqdm

from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams
from recipes.bigmusic.lightning.embedding_modules import (
    BestRQTokenEmbedder,
    LeadsheetTokenEmbedderV2,
)
from recipes.musiclm.inference.utils import sample, top_p_logits


def test_l2l2s_inference():
    leadsheet_max_seq_len = 800
    duration = 30
    top_p = 0.9
    temperature = 1.0
    sample_mode = "top_p"

    start_time = time.time()

    ## 1. Load semantic ckpt, which contains all parameters for 
    ## semantic model and embedders
    dump_dir = os.path.join(
        os.environ["AI_MUSIC_DIR"],
        "../../deps/samantha/.module_cache/musiclm"
    )
    semantic_ckpt = os.path.join(dump_dir, "step=068000-tr_loss=1.7907-val_accu_0=88.17.ckpt")
    ckpt = torch.load(semantic_ckpt)
    state_dict: Dict[str, torch.Tensor] = ckpt["state_dict"]
    extra_params = ckpt["hyper_parameters"]["extra_params"]
    device = state_dict[list(state_dict.keys())[-1]].device

    current_time = time.time()
    ckpt_loading_duration = current_time - start_time
    print(f"ckpt_loading_duration={ckpt_loading_duration}")
    start_time = current_time

    ## 2. Load leadsheet embedder parameter
    prefix = "input_embedders.leadsheet_tokens."
    leadsheet_embedder_state_dict = {
        k.partition(prefix)[2]: v \
            for k, v in state_dict.items() \
            if prefix in k
    }
    leadsheet_embedder = LeadsheetTokenEmbedderV2(
        vocab_size=extra_params["leadsheet_codebook_size"],
        embedding_dim=extra_params["hidden_size"],
        add_sos=True
    )
    leadsheet_embedder.load_state_dict(leadsheet_embedder_state_dict)
    leadsheet_embedder.half()
    leadsheet_embedder.to(device)
    leadsheet_embedder.eval()

    ## 3. Load semantic model parameter
    prefix = "model."
    model_state_dict = {
        k.partition(prefix)[2]: v \
            for k, v in state_dict.items() \
            if prefix in k
    }
    model_config = GPT2Config(
        vocab_size = 1, # placeholder. Embedding table gets removed on load
        max_position_embeddings = 0,
        use_cache = False,
        n_positions = 0,  # No absolute position embedding
        num_logits = extra_params["semantic_codebook_size"] + 2 + extra_params["leadsheet_codebook_size"],  # +2 for EOS
        n_embd = extra_params["hidden_size"],
        n_layer = extra_params["num_hidden_layers"],
        n_head = extra_params["num_attention_heads"],
        n_inner = extra_params["intermediate_size"],
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
        rotary_emb_compat = "default", #"byteformer"
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
    model = gpt.GPTLMHeadModel(model_config)
    ## Since we are not using transformer.embeddings.word_embeddings.weight
    ## we load the state dict with strict=False
    model.load_state_dict(model_state_dict, strict=False)
    model.half()
    model.to(device)
    model.eval()

    ## 4. Load target embedder
    prefix = "target_embedder."
    target_embedder_state_dict = {
        k.partition(prefix)[2]: v \
            for k, v in state_dict.items() \
            if prefix in k
    }
    target_embedder = BestRQTokenEmbedder(
        vocab_size=extra_params["semantic_codebook_size"],
        embedding_dim=extra_params["hidden_size"],
        add_sos=True,
        add_eos=True,
    )
    ## semantic_codebook_size + sos + eos
    num_target_embeddings = target_embedder.embedder.num_embeddings
    target_embedder.load_state_dict(target_embedder_state_dict)
    target_embedder.half()
    target_embedder.to(device)
    target_embedder.eval()

    current_time = time.time()
    state_dict_loading_duration = current_time - start_time
    print(f"state_dict_loading_duration={state_dict_loading_duration}")
    start_time = current_time

    ## 5. Prepare inputs
    inputs_embeds_path = os.path.join(
        os.environ['DEPS_DIR'], 'samantha/recipes/bigmusic/tests/data/20240227.l2l2s.inferinputs.npy'
    )
    inputs_embeds = torch.from_numpy(np.load(inputs_embeds_path)).to(device)
    batch_size, seq_len, _ = inputs_embeds.size()
    # For L2S, replace the audio sos from the inputs by leadsheet sos.
    sos_embeds = leadsheet_embedder.get_sos_embed(batch_size)
    model_input = {
        "inputs_embeds": torch.cat([inputs_embeds, sos_embeds], dim=1).half()
    }

    ## 6. Do inference
    num_tokens = extra_params["semantic_frame_rate"] * duration + leadsheet_max_seq_len
    gpt_max_seq_len = 4000 if num_tokens < 2500 else 8000
    inference_params = InferenceParams(
        max_sequence_len=gpt_max_seq_len, max_batch_size=batch_size
    )
    pbar = tqdm(range(num_tokens))
    output_tokens = None
    for i in pbar:
        output = model(
            **model_input,
            inference_params=inference_params,
            position_ids=None,
            last_token_only=False,
            output_hidden_states=True,
        )
        ## The output is a namedtuple of some keys and I don't know
        ## how to do type inference on that...
        logits = output.logits
        inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
        logits = logits[:, -1:, :] # only predicting on last logit.
        try:
            predict_token = sample(logits, temperature, top_p, sample_mode)
        except:
            from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
        # for leadsheet, use leadsheet embedder. for audio, use target embedder.
        if i < leadsheet_max_seq_len:
            # TODO (QQ) When time tokens are used, handle the number tokens differently. Read from output_hidden_states.            
            predict_token_emb = leadsheet_embedder.embedder(
                torch.clamp(predict_token - num_target_embeddings, min=0)
            )
        else:
            predict_token_emb = target_embedder.embedder(
                torch.clamp(predict_token, max=num_target_embeddings - 3)
            )

        model_input['inputs_embeds'] = predict_token_emb.half()
        output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token

    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
    output_tokens = output_tokens[:, leadsheet_max_seq_len + 1:]
    return output_tokens