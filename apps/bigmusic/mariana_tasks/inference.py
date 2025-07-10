import copy
import importlib
import yaml

from typing import Optional
import pytorch_lightning as pl

from samantha.utils.hparams import DotDict
from recipes.bigmusic.lightning.semantic_modules import process_eos_indexes, truncate_wav_to_eos
import logging
from cruise import CruiseConfig

from apps.bigmusic.mariana_tasks.semantic_seed_train import _m8_network_config, _inference_config
from apps.bigmusic.umm.diffusion.requires.model_initializer import run_diffusion_vocoder_batch
import torch


def process_eos_indexes(semantic_samples, eos_id, semantic_frame_rate=25, sample_rate=24000):
    semantic_samples = semantic_samples.clone()
    eos_index_list = []

    if eos_id is not None:
        """
        @renyi 08/02/2024: if we set eos_padding_id to 0, the "semantic_samples == eos_padding_id" will also include the real acoustic code 0, 
        leading to end of sentence when the real acoustic code 0 appears.
        So we set a EOS padding id (-10000) to a placeholder which is impossibly shown in the acoustic tokens. 
        """
        eos_padding_id = -10000
        eos_mask = torch.cumsum(semantic_samples == eos_id, 1) > 0
        semantic_samples[eos_mask] = eos_padding_id
        token2wav_rate = int(sample_rate / semantic_frame_rate)
        eos_index_list = ((semantic_samples == eos_padding_id).bool().cumsum(axis=1) == 0).bool().sum(
            axis=1) * token2wav_rate
        # @qinxin: temp fix, currently tokenizer_pad_id = eos_id - 1
        semantic_samples[eos_mask] = eos_id - 1
    return semantic_samples, eos_index_list


class SemanticInferenceModule(pl.LightningModule):
    def __init__(
        self,
        semantic_cls_path: str,
        required_modules: dict,
        extra_params: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        logging.info(f"extra params: {self.extra_params}")

        *module_paths, cls_name = semantic_cls_path.split('.')
        module = importlib.import_module('.'.join(module_paths))
        semantic_class = getattr(module, cls_name)

        network_cfg_file = self.extra_params.get("network_cfg")
        network_overrides = self.extra_params.get("network_overrides", None) or {}
        with open(network_cfg_file, 'r') as file:
            update_cfg = yaml.safe_load(file)
        network_cfg = copy.deepcopy(_m8_network_config)
        network_cfg.update(update_cfg["model"]["network"])
        network_cfg.update(network_overrides)
        # TODO: support control this config in the config file and command line
        network_cfg['use_flash_attn_kvcache'] = True # infer with flash attn
        network_cfg['gpt_use_fused_block'] = False  # bigop not support infer
        network_cfg['use_llm_bf16'] = True

        inference_cfg = copy.deepcopy(_inference_config)
        inference_cfg.update(self.extra_params.get("inference", {}))

        self.semantic_module = semantic_class(
            emb_path=self.extra_params.get("emb_path"),
            llm_path=self.extra_params.get("llm_path"),
            partial_pretrain=self.extra_params.get("partial_pretrain"),
            network = CruiseConfig(dict(network_cfg)),
            inference = CruiseConfig(dict(inference_cfg)),
        )
        self.requires = {}
        self.predict_step_seed: Optional[int] = self.extra_params.get("predict_step_seed")

    def setup(self, stage: str) -> None:
        if self.local_rank == 0:
            self.semantic_module.local_rank_zero_prepare()
        self.semantic_module.setup()
        # Modules must be loaded in setup function for correct local rank / multi-gpu training
        required_modules = {}
        # this is the tts token2wav
        self.decoding_fn = run_diffusion_vocoder_batch
        required_modules.update(self.hparams.required_modules['diffusion_modules'])
        self.decoding_params = DotDict({ **self.extra_params })
        self.load_required_modules(required_modules)

    def load_required_modules(self, required_modules):
        for item in required_modules.values():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        if self.predict_step_seed is not None:
            pl.seed_everything(self.predict_step_seed)  # for batch size invariant reproducibility

        raw_semantic_samples = self.semantic_module.predict(
            batch,
            self.extra_params,
            beam=self.extra_params.beam_size,
        )
            
        eos_id = batch.get('eos_id', None)
        if eos_id is None:
            eos_id = self.semantic_module.emb.target_embedder.eos_id
        else:
            eos_id = eos_id - batch.get("text_codebook_size", 0)

        semantic_samples, eos_index_list = process_eos_indexes(
            raw_semantic_samples,
            eos_id,
            self.extra_params.semantic_frame_rate,
            self.extra_params.sample_rate,
        )
        duration = self.extra_params.duration            
        raw_wav_output = self.decoding_fn(self.requires, semantic_samples, prompt_wav_paths=batch.get("vocal_prompt", None))
        raw_wav_output = raw_wav_output[..., :duration * self.extra_params.sample_rate]

        outputs = {}
        raw_wav_output = raw_wav_output.detach().cpu()
        wavs = truncate_wav_to_eos(raw_wav_output, eos_index_list)
        raw_semantic_samples = raw_semantic_samples.detach().cpu()
        outputs.update({
            'generated_audio': wavs,
            'generated_audio_tensor': raw_wav_output,
            'generated_semantic_tokens': raw_semantic_samples,
        })
        if "generated_leadsheet_tokens" in batch:
            outputs["generated_leadsheet_tokens"] = batch["generated_leadsheet_tokens"]
        if "generated_string" in batch:
            outputs["generated_string"] = batch["generated_string"]

        return outputs
