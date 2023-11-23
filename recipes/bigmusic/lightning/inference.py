import os
from pydoc import classname

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio

from samantha.utils.hparams import DotDict
from typing import Any
from recipes.musiclm.inference.utils import slugify, save_wav, generate_hash, format_name, load_wav
from recipes.diffusion.models.diffusion_model.utils import run_diffusion
from recipes.bigmusic.lightning.embedding_modules import get_bestrq_umm_tokens
import torch.functional
import importlib
from recipes.bigmusic.utils.model_initializer import run_2ar
from itertools import zip_longest
from recipes.bigmusic.lightning.base_modules import BaseModule
from recipes.bigmusic.lightning.semantic_modules import process_eos_indexes, truncate_wav_to_eos
import json
from pathlib import Path
import numpy as np
from samantha.models.flash_llama import LlamaPreTrainedModel
from samantha.models.ctiga import gpt


class SemanticInferenceModule(pl.LightningModule):
    def __init__(
        self,
        semantic_cls_path,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)

        *module_paths, cls_name = semantic_cls_path.split('.')
        module = importlib.import_module('.'.join(module_paths))
        semantic_class = getattr(module, cls_name)

        self.semantic_module: BaseModule = semantic_class.load_from_checkpoint(
            self.extra_params.semantic_ckpt,
            # pay attention to the logs to make sure the model is loaded correctly
            strict=True,
        ).eval()
        if cls_name == "SemanticModuleVarlenXperf" or cls_name == "SemanticModuleXperf":
            self.semantic_module.replace_ctiga_to_xperf()
        self.requires = {}
        
        required_modules = {}
        if self.extra_params.token2wav_type == 'diffusion':
            self.decoding_fn = run_diffusion
            required_modules.update(self.hparams.required_modules['diffusion_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['diffusion_params'] })
        elif self.extra_params.token2wav_type == 'ar':
            self.decoding_fn = run_2ar
            required_modules.update(self.hparams.required_modules['ar_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['ar_params'] })
        else:
            raise ValueError(f"Unhandled type: {self.extra_params.token2wav_type}")

        if self.extra_params.use_reranker:
            if self.extra_params.beam_size <= 1:
                print(f"[WARNING] use_reranker=True but beam_size={self.extra_params.beam_size}")
            required_modules.update({"reranker": self.hparams.required_modules["reranker"]})

        self.load_required_modules(required_modules)


    def setup(self, stage: str) -> None:
        if isinstance(self.semantic_module.model, gpt.GPTLMHeadModel) and ('32' in self.trainer.precision):
            raise Exception(f"Invalid precision for cTIGA model {self.trainer.precision}. Please set --run_opts.precision 16")
        if isinstance(self.semantic_module.model, LlamaPreTrainedModel) and ('16' in self.trainer.precision):
            raise Exception(f"Invalid precision for flash llama model {self.trainer.precision}. Please set --run_opts.precision 32")

    def load_required_modules(self, required_modules):
        for name, item in required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
            
        # set semantic mulan ckpt if passed in
        if self.extra_params.get('mulan_ckpt', None) and self.extra_params.mulan_ckpt != 'infer_from_semantic_ckpt':
            self.semantic_module.hparams.required_modules['mulan']['hpath'] = self.extra_params.mulan_ckpt
        self.semantic_module.load_required_modules(ignore=('bestrq',))

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        semantic_samples = self.semantic_module.predict(
            batch,
            self.extra_params,
            beam=self.extra_params.beam_size,
        )
        semantic_samples, eos_index_list = process_eos_indexes(
            semantic_samples,
            self.semantic_module,
            self.extra_params.sample_rate,
        )
        raw_wav_output = self.decoding_fn(self.requires, semantic_samples, self.decoding_params)
        assert len(raw_wav_output.shape) == 2, "Wavs must be 2 sim [b, seq_len]"
        raw_wav_output = raw_wav_output[..., :self.extra_params.duration * self.extra_params.sample_rate]

        outputs = {}
        if self.extra_params.use_reranker:
            raw_wav_output, eos_index_list, rewards_breakdown = self.requires["reranker"].rerank(
                raw_wav_output,
                eos_index_list,
                batch,
                self.extra_params,
            )
            outputs["metadata"] = [{"rewards": x} for x in rewards_breakdown]
        
        raw_wav_output = raw_wav_output.detach().cpu()
        wavs = truncate_wav_to_eos(raw_wav_output, eos_index_list)
        outputs.update({ 
            'generated_audio': wavs,
            'generated_audio_tensor': raw_wav_output,
        })
        return outputs


class GTInferenceModule(pl.LightningModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        
        required_modules = {}
        if self.extra_params.token2wav_type == 'diffusion':
            self.decoding_fn = run_diffusion
            required_modules.update(self.hparams.required_modules['diffusion_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['diffusion_params'] })
        elif self.extra_params.token2wav_type == 'ar':
            self.decoding_fn = run_2ar
            required_modules.update(self.hparams.required_modules['ar_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['ar_params'] })
        else:
            raise ValueError(f"Unhandled type: {self.extra_params.token2wav_type}")

        if self.extra_params.semantic_type == 'bestrq':
            required_modules.update(self.hparams.required_modules['bestrq_modules'])
            self.encoding_fn = get_bestrq_umm_tokens

        self.load_required_modules(required_modules)
        self.wer = []

    def load_required_modules(self, required_modules):
        for name, item in required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        batch['target_audio'] = batch['style_audio'] # prepare_inputs expects target_audio key
        semantic_samples = self.encoding_fn(self.requires, batch['target_audio'])
        wavs = self.decoding_fn(self.requires, semantic_samples, self.decoding_params) # 
        return { 
            'generated_audio': wavs,
            'generated_audio_tensor': wavs
        }
