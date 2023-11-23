from recipes.bigmusic.lightning.semantic_varlen_modules import SemanticModuleVarlen
from recipes.bigmusic.lightning.semantic_modules import SemanticModule

import xperf_gpt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import transformers
import math
import logging
from tqdm import tqdm

from samantha.models.ctiga import gpt as ctiga_gpt
from samantha.models.ctiga import llama as ctiga_llama
from samantha.models import flash_llama

xperf_gpt.load_xperf_gpt()
XPERF_SUPPORT_HEADDIM = 128
device = "cuda:0"

def is_two_exp(number):
    return number & (number - 1) == 0

def semantic_torch2hf(torch_model: flash_llama.LlamaForCausalLM):
    torch_model.model.set_input_embeddings(
        nn.Embedding(
            torch_model.config.num_logits, torch_model.config.hidden_size
        ).eval()
    )
    torch_model.eval()
    hf_llama_config = transformers.LlamaConfig(
        vocab_size=torch_model.config.num_logits,
        hidden_size=torch_model.config.hidden_size,
        intermediate_size=torch_model.config.intermediate_size,
        num_attention_heads=torch_model.config.num_attention_heads,
        num_hidden_layers=torch_model.config.num_hidden_layers,
        rms_norm_eps=torch_model.config.rms_norm_eps,
        max_position_embeddings=torch_model.config.max_position_embeddings,
    )
    unpad_state_dict = torch_model.state_dict()
    headdim = hf_llama_config.hidden_size // hf_llama_config.num_attention_heads
    assert headdim <= XPERF_SUPPORT_HEADDIM and is_two_exp(headdim)
    scale = XPERF_SUPPORT_HEADDIM // headdim
    if scale != 1:
        print(f"semantic model need to pad when use_xperf due to headdim={headdim}")
        hf_llama_config.hidden_size *= scale
        padded_hf_llama = transformers.LlamaForCausalLM(hf_llama_config).eval()
        pad_llama_model(padded_hf_llama, unpad_state_dict, scale)
    else:
        padded_hf_llama = transformers.LlamaForCausalLM(hf_llama_config).eval()
        padded_hf_llama.load_state_dict(unpad_state_dict)

    return padded_hf_llama


def semantic_ctiga2hf(ctiga_model: ctiga_gpt.GPTLMHeadModel):
    ctiga_model.eval()
    hf_llama_config = hf_llama_config = transformers.LlamaConfig(
        vocab_size=ctiga_model.config.num_logits,
        hidden_size=ctiga_model.config.n_embd,
        intermediate_size=ctiga_model.config.n_inner,
        num_attention_heads=ctiga_model.config.n_head,
        num_hidden_layers=ctiga_model.config.n_layer,
        rms_norm_eps=ctiga_model.config.layer_norm_epsilon,
    )
    unpad_state_dict = ctiga_llama.remap_state_dict_ctiga_llama_to_hf(
        ctiga_model.state_dict(),
        n_layers=hf_llama_config.num_hidden_layers,
        n_heads=hf_llama_config.num_attention_heads,
        n_embd=hf_llama_config.hidden_size,
    )

    headdim = hf_llama_config.hidden_size // hf_llama_config.num_attention_heads
    assert headdim <= XPERF_SUPPORT_HEADDIM and is_two_exp(headdim)
    scale = XPERF_SUPPORT_HEADDIM // headdim

    if scale != 1:
        print(f"semantic model need to pad when use_xperf due to headdim={headdim}")
        hf_llama_config.hidden_size *= scale
        padded_hf_llama = transformers.LlamaForCausalLM(hf_llama_config).eval()
        pad_llama_model(padded_hf_llama, unpad_state_dict, scale)
    else:
        padded_hf_llama = transformers.LlamaForCausalLM(hf_llama_config).eval()
        padded_hf_llama.load_state_dict(unpad_state_dict, strict=False)
    return padded_hf_llama


def semantic_hf2xperf(
    hf_model: transformers.LlamaForCausalLM, max_batch_size, max_length, dtype, **kwargs
):
    engine = xperf_gpt.init_inference(
        hf_model,
        max_batch_size=max_batch_size,
        max_length=max_length,
        use_xperf_gpt=True,
        dtype=dtype,
        mp_size=1,
        rank0_split=False,
        checkpoint_path=None,
    )
    if hasattr(engine.module, "wte"):
        delattr(engine.module, "wte")
    return engine


def pad_llama_model(hf_model: transformers.LlamaForCausalLM, unpad_state_dict, scale):
    def pad_linear(m, w, factor=1.0, pad_ic=True, pad_oc=True):
        new_w = torch.zeros_like(w)
        new_w = new_w.repeat(scale if pad_oc else 1, scale if pad_ic else 1)
        if pad_oc and pad_ic:
            new_w[::scale, ::scale] = w
        elif pad_oc and not pad_ic:
            new_w[::scale] = w
        elif not pad_oc and pad_ic:
            new_w[:, ::scale] = w
        else:
            new_w = w
        m.weight.data.copy_(new_w * factor)

    def pad_attention(m, w):
        pad_linear(m.q_proj, w[0], math.sqrt(scale))
        pad_linear(m.k_proj, w[1])
        pad_linear(m.v_proj, w[2])
        pad_linear(m.o_proj, w[3])

    def pad_mlp(m, w):
        pad_linear(m.gate_proj, w[0], pad_ic=True, pad_oc=False)
        pad_linear(m.up_proj, w[1], pad_ic=True, pad_oc=False)
        pad_linear(m.down_proj, w[2], pad_ic=False, pad_oc=True)

    def pad_norm(m, w):
        new_w = torch.zeros_like(w)
        new_w = new_w.repeat(scale)
        new_w[::scale] = w
        m.weight.data.copy_(new_w)
        m.variance_epsilon *= scale

    def pad_embedding(m, w):
        new_w = torch.zeros_like(w)
        new_w = new_w.repeat(1, scale)
        new_w[::scale] = w
        m.weight.data.copy_(new_w)

    pad_norm(hf_model.model.norm, unpad_state_dict.pop("model.norm.weight"))
    if "model.embed_tokens.weight" in unpad_state_dict:
        pad_embedding(
            hf_model.model.embed_tokens,
            unpad_state_dict.pop("model.embed_tokens.weight"),
        )
    if "model.lm_head.weight" in unpad_state_dict:
        pad_embedding(hf_model.lm_head, unpad_state_dict.pop("model.lm_head"))

    for layer_idx in range(hf_model.config.num_hidden_layers):
        layer = hf_model.model.layers[layer_idx]
        w_prefix = f"model.layers.{layer_idx}"
        pad_attention(
            layer.self_attn,
            [
                unpad_state_dict.pop(f"{w_prefix}.self_attn.{subkey}_proj.weight")
                for subkey in ["q", "k", "v", "o"]
            ],
        )
        pad_mlp(
            layer.mlp,
            [
                unpad_state_dict.pop(f"{w_prefix}.mlp.{subkey}_proj.weight")
                for subkey in ["gate", "up", "down"]
            ],
        )
        pad_norm(
            layer.input_layernorm,
            unpad_state_dict.pop(f"{w_prefix}.input_layernorm.weight"),
        )
        pad_norm(
            layer.post_attention_layernorm,
            unpad_state_dict.pop(f"{w_prefix}.post_attention_layernorm.weight"),
        )


def xperf_init_inputs_for_generation(
    xperf_engine: xperf_gpt.InferenceEngine,
    input_token_ids,
    input_embeds,
    input_lengths=None,
    num_return=1,
):
    assert (input_token_ids is None) ^ (input_embeds is None)
    dtype = xperf_engine.config.dtype.value

    if input_token_ids is not None:
        input_embeds = xperf_engine.module.wte(input_token_ids)
    input_embeds = input_embeds.to(dtype)

    bs, max_seqlen = input_embeds.shape[:2]

    expanded_return_idx = (
        torch.arange(bs).view(-1, 1).repeat(1, num_return).view(-1).to(device)
    )
    input_embeds = input_embeds.index_select(0, expanded_return_idx)

    attention_mask = torch.ones(bs, max_seqlen, dtype=torch.long, device=device)
    if input_lengths is not None:
        for bidx in range(bs):
            attention_mask[bidx, input_lengths[bidx] :] *= 0

    attention_mask = attention_mask.index_select(0, expanded_return_idx)

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    return input_embeds.to(dtype), attention_mask.to(dtype), position_ids


def xperf_predict(
    semantic_module,
    inputs_embeds,
    num_tokens,
    temperature,
    sample_mode="gumbel",
    expand_size=1,
):
    logging.info("======================== xperf predict ======================")
    xperf_engine = semantic_module.model

    assert isinstance(xperf_engine, xperf_gpt.InferenceEngine)
    batch_size = inputs_embeds.size(0)
    device = inputs_embeds.device
    dtype = xperf_engine.config.dtype.value
    sos_embeds = semantic_module.target_embedder.get_sos_embed(batch_size)
    if semantic_module.use_cross_attn:
        model_input = {
            "inputs_embeds": sos_embeds,
            "encoder_hidden_states": inputs_embeds,
        }
    else:
        model_input = {
            "inputs_embeds": torch.cat([inputs_embeds, sos_embeds], dim=1).to(dtype)
        }

    output_tokens = None

    input_embeds, attn_mask, pos_ids = xperf_init_inputs_for_generation(
        xperf_engine, None, model_input["inputs_embeds"], None, expand_size
    )

    for i in tqdm(range(num_tokens)):
        hidden_states = input_embeds.to(dtype)
        attn_mask = attn_mask.type_as(hidden_states)

        if i == 0:
            xperf_engine.module.check_status(context_only=False)
            xperf_engine.module.reset_decoder_step()

        logits = xperf_engine.module.custom_decoder.forward(hidden_states, attn_mask, i)
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        next_token_ids = semantic_module.sample_logits(
            i, logits, temperature, sample_mode
        )
        if next_token_ids.ndim == 1:
            next_token_ids = next_token_ids.unsqueeze(0)

        input_embeds = semantic_module.target_embedder.embedder(next_token_ids)

        output_tokens = (
            torch.cat([output_tokens, next_token_ids], dim=1)
            if output_tokens is not None
            else next_token_ids
        )

    return output_tokens

class SemanticModuleXperf(SemanticModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        use_xperf=True,
    ):
        super().__init__(model_cls, criterion_cls, optimizer_cls, scheduler_cls, required_modules, checkpointing, extra_params)
        self.use_xperf = use_xperf

    def replace_ctiga_to_xperf(
        self,
        batch_size=4,
        num_return=4,
        lyrics_max_seq_len=500,
        num_tokens=750,
        infer_dtype="fp16"
        ):
        if isinstance(self.model, ctiga_gpt.GPTLMHeadModel):
            if self.use_xperf:
                self.model = semantic_hf2xperf(
                    semantic_ctiga2hf(self.model),
                    batch_size * num_return,
                    max_length=lyrics_max_seq_len + 4 + num_tokens,
                    dtype=torch.float16 if infer_dtype == "fp16" else torch.bfloat16
                )
                print("Load semantic ctiga-xperf model successful")
            else:
                self.model.to(dtype=torch.float16 if infer_dtype == "fp16" else torch.bfloat16)
                print("Load semantic ctiga-ctiga model successful")

        elif isinstance(self.model, flash_llama.LlamaForCausalLM):
            if self.use_xperf:
                self.model = semantic_hf2xperf(
                    semantic_torch2hf(self.model),
                    batch_size * num_return,
                    max_length=2048,
                    dtype=torch.float16 if infer_dtype == "fp16" else torch.bfloat16
                )
                print("Load semantic flashllama-xperf model successful")
            else:
                print("Load semantic flashllama-flashllama model successful")
        logging.info("load semantic module success")
    
    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode

        if self.use_xperf:
            inputs_embeds = self.prepare_inputs_embeddings(batch)

            semantic_samples = xperf_predict(
                    self,
                    inputs_embeds,
                    num_tokens,
                    temperature=1.0,
                    sample_mode='gumbel',
                    expand_size=beam,
                )
            return semantic_samples
        else:
            return super().predict(batch, hp, beam, ref_samples)

class SemanticModuleVarlenXperf(SemanticModuleVarlen):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        use_xperf=True,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

        self.use_xperf = use_xperf
    
    def replace_ctiga_to_xperf(
        self,
        batch_size=4,
        num_return=4,
        lyrics_max_seq_len=1500,
        num_tokens=3000,
        infer_dtype="fp16"
        ):
        if isinstance(self.model, ctiga_gpt.GPTLMHeadModel):
            if self.use_xperf:
                self.model = semantic_hf2xperf(
                    semantic_ctiga2hf(self.model),
                    batch_size * num_return,
                    max_length=lyrics_max_seq_len + 4 + num_tokens,
                    dtype=torch.float16 if infer_dtype == "fp16" else torch.bfloat16
                )
                print("Load semantic ctiga-xperf model successful")
            else:
                self.model.to(dtype=torch.float16 if infer_dtype == "fp16" else torch.bfloat16)
                print("Load semantic ctiga-ctiga model successful")

        elif isinstance(self.model, flash_llama.LlamaForCausalLM):
            if self.use_xperf:
                self.model = semantic_hf2xperf(
                    semantic_torch2hf(self.model),
                    batch_size * num_return,
                    max_length=2048,
                    dtype=torch.float16 if infer_dtype == "fp16" else torch.bfloat16
                )
                print("Load semantic flashllama-xperf model successful")
            else:
                print("Load semantic flashllama-flashllama model successful")
        logging.info("load semantic module success")

    @torch.no_grad()
    def predict(self, batch, hp, beam=None,):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        predict_lyrics = hp.get("predict_lyrics", False)

        if self.use_xperf:
            model_inputs = self.prepare_model_inputs(batch)

            # zip inputs and concat
            token_embeds = zip(*[i['token_embeds'] for i in model_inputs])
            token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
            input_seq_lengths = torch.vstack([i['token_seq_lengths'] for i in model_inputs]).sum(dim=0)

            # pad front
            token_embeds_reversed = [embeds.flip(dims=(0,)) for embeds in token_embeds]
            token_embeds_pad = pad_sequence(token_embeds_reversed, batch_first=True, padding_value=0).flip(dims=(1,))

            semantic_samples = xperf_predict(
                    self,
                    token_embeds_pad,
                    num_tokens,
                    temperature=1.0,
                    sample_mode='gumbel',
                    expand_size=beam,
                )
            return semantic_samples
        else:
            return super().predict(batch, hp, beam)

if __name__ == "__main__":
    from recipes.bigmusic.datasets.inference import inference_dataset_from_prompt
    from recipes.bigmusic.datasets.lyrics import LyricsDataModule
    from recipes.musiclm.inference.utils import dump_wav, save_wav

    import time
    import os
    import sys
    from hyperpyyaml import load_hyperpyyaml

    class DotDict(dict):
        """Dictionary that supports dot notation.

        Arguments
        --------
        py_dict: dict, {}
            A python dict object, will be recursively converted to DotDict.

        Example
        --------
        >>> d = {"key1": "val1", "key2": {"key3": "val3"}}
        >>> dot_d = DotDict(d)
        >>> dot_d.key1
        'val1'
        >>> dot_d["key1"]
        'val1'
        >>> dot_d.key2.key3
        'val3'
        >>> dot_d["key2"]["key3"]
        'val3'
        >>> dot_d.key2.key3 = "new_val"
        >>> dot_d.key2.key3
        'new_val'
        """

        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

        def __init__(self, py_dict: dict = {}):
            for key, value in py_dict.items():
                if isinstance(value, list):
                    for i in range(len(value)):
                        if isinstance(value[i], dict):
                            value[i] = DotDict(value[i])
                if isinstance(value, dict):
                    value = DotDict(value)
                self[key] = value

    def load_cached_models():
        logging.info("***** start loading model *****")
        conf_path = "/mnt/bn/music-ai-unified-repo-lq/lihui/py_demo_server/thirdparty/samantha/recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s_xperf.yaml"
        try:
            with open(conf_path, "r", encoding="utf-8") as f:
                configs = DotDict(load_hyperpyyaml(f))
        except Exception as err:
            logging.error(err)
            sys.exit(0)
        logging.info(f"===== model config: {configs}")

        pl_module = configs.pl_module
        extra_params = configs.extra_params
        trainer = configs.trainer

        pl_module.to(device).eval()

        logging.info("***** load model success *****")
        return pl_module, extra_params, trainer

    lyrics = """
Imagine there's no countries
It isn't hard to do
Nothing to kill or die for
And no religion, too
Imagine all the people
Livin' life in peace
    """
    mood = "Happy"
    genre = "hip hop/rap"
    gender = "Female"

    logging.info(f"lyrics2song request, lyrics:{lyrics} mood:{mood} genre:{genre} gender:{gender}")

    pl_module, extra_params, trainer = load_cached_models()

    start = time.time()
    logging.info(f"load model cost {time.time() - start} s")

    n_samples = 4
    style_prompt_metadata = {}
    style_prompt_metadata['final_genre'] = genre
    style_prompt_metadata['final_mood'] = mood
    style_prompt_metadata['merge_aed'] = gender

    prompts = {'metadata': [style_prompt_metadata] * n_samples,
               'lyrics': [lyrics] * n_samples}
    inference_dataset = inference_dataset_from_prompt(
        prompts, conditions="style_text,lyrics_tokens",
        batch_size=n_samples,
        lyrics_max_seq_len=extra_params.lyrics_max_seq_len)

    pl_datamodule = LyricsDataModule(predict_dataset=inference_dataset, num_workers=0)
    predictions = trainer.predict(pl_module, pl_datamodule)
    output_wavs = predictions[0]['generated_audio']

    sr = extra_params.sample_rate
    urls = []

    for idx, output_wav in enumerate(output_wavs):
        save_wav(output_wav.cpu().float(), f"generated_wav_{idx}.wav")
