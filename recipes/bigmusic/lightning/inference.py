import os

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
from recipes.bigmusic.lightning.semantic_modules import process_eos_indexes, truncate_wav_to_eos
import json
from pathlib import Path
import numpy as np

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

        self.semantic_module = semantic_class.load_from_checkpoint(self.extra_params.semantic_ckpt).eval()
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
        self.load_required_modules(required_modules)

    def load_required_modules(self, required_modules):
        for name, item in required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
        self.semantic_module.load_required_modules()

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        semantic_samples = self.semantic_module.predict(batch, self.extra_params)
        semantic_samples, eos_index_list = process_eos_indexes(semantic_samples, self.semantic_module, self.extra_params.sample_rate)
        raw_wav_output = self.decoding_fn(self.requires, semantic_samples, self.decoding_params).detach().cpu()
        assert len(raw_wav_output.shape) == 2, "Wavs must be 2 sim [b, seq_len]"
        
        wavs = truncate_wav_to_eos(raw_wav_output, eos_index_list)
        return { 
            'generated_audio': wavs,
            'generated_audio_tensor': raw_wav_output
        }


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
