from functools import partial
from typing import Optional

import pytorch_lightning as pl

from samantha.utils.hparams import DotDict
from recipes.diffusion.models.diffusion_model.utils import run_diffusion
# from recipes.voicebox.lit_modules.lit_diffusion_reconstruct import run_diffusion_vocoder
from apps.bigmusic.umm.diffusion.requires.model_initializer import run_diffusion_vocoder_batch
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.soundstorm.lightning.utils import run_soundstorm
from recipes.musiclm.utils.dist import local_zero_first
from recipes.bigmusic.lightning.embedding_modules import get_bestrq_umm_tokens
import importlib
from recipes.bigmusic.utils.model_initializer import run_2ar
from recipes.bigmusic.lightning.base_modules import BaseModule
from recipes.bigmusic.lightning.semantic_modules import process_eos_indexes, truncate_wav_to_eos
from samantha.models.flash_llama import LlamaPreTrainedModel
from samantha.models.ctiga import gpt
import logging
from pathlib import Path

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
        logging.info(f"extra params: {self.extra_params}")

        *module_paths, cls_name = semantic_cls_path.split('.')
        module = importlib.import_module('.'.join(module_paths))
        semantic_class = getattr(module, cls_name)

        with local_zero_first():
            semantic_dir = Path(self.extra_params.semantic_ckpt).parent.parent.name
            semantic_ckpt_path = download_checkpoint(self.extra_params.semantic_ckpt, f'.module_cache/{semantic_dir}')

        self.semantic_module: BaseModule = semantic_class.load_from_checkpoint(
            semantic_ckpt_path,
            # need to be False to load RL ckpts
            strict=False,
        ).eval()
        if cls_name == "SemanticModuleVarlenXperf" or cls_name == "SemanticModuleXperf":
            self.semantic_module.replace_ctiga_to_xperf()
        self.requires = {}
        self.predict_step_seed: Optional[int] = self.extra_params.get("predict_step_seed")

    def setup(self, stage: str) -> None:
        if isinstance(self.semantic_module.model, gpt.GPTLMHeadModel) and ('32' in self.trainer.precision):
            raise Exception(
                f"Invalid precision for cTIGA model {self.trainer.precision}. Please set --run_opts.precision 16")
        if isinstance(self.semantic_module.model, LlamaPreTrainedModel) and ('16' in self.trainer.precision):
            raise Exception(
                f"Invalid precision for flash llama model {self.trainer.precision}. Please set --run_opts.precision 32")
        
        # Modules must be loaded in setup function for correct local rank / multi-gpu training
        required_modules = {}
        if self.extra_params.token2wav_type == 'diffusion':
            # this is the old music token2wav
            self.decoding_fn = run_diffusion
            required_modules.update(self.hparams.required_modules['diffusion_modules'])
            self.decoding_params = DotDict({**self.extra_params, **self.extra_params['diffusion_params']})
        elif self.extra_params.token2wav_type == 'ar-diffusion-vocoder':
            # this is the tts token2wav
            self.decoding_fn = run_diffusion_vocoder_batch
            required_modules.update(self.hparams.required_modules['diffusion_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **self.extra_params['diffusion_params'] })
        elif self.extra_params.token2wav_type == 'ar':
            # this is the soundstorm token2wav
            self.decoding_fn = run_2ar
            required_modules.update(self.hparams.required_modules['ar_modules'])
            self.decoding_params = DotDict({**self.extra_params, **self.extra_params['ar_params']})
        elif 'dualumm' in self.extra_params.token2wav_type:
            from recipes.umm.modules.lit_module_mkii_dual import run_dualMSS_decode
            self.decoding_fn = partial(
                run_dualMSS_decode, token_type=self.extra_params.token2wav_type.replace("dualumm_", ""))
            required_modules.update(self.hparams.required_modules['dualumm_modules'])
            self.decoding_params = DotDict({**self.extra_params})
        elif self.extra_params.token2wav_type == 'soundstorm':
            self.decoding_fn = run_soundstorm
            required_modules.update(self.hparams.required_modules['soundstorm_modules'])
            self.decoding_params = DotDict({**self.extra_params, **self.extra_params['soundstorm_params']})
        else:
            raise ValueError(f"Unhandled type: {self.extra_params.token2wav_type}")

        logging.info(f"use reranker: {self.extra_params.use_reranker}")
        if self.extra_params.use_reranker:
            if self.extra_params.beam_size <= 1:
                print(f"[WARNING] use_reranker=True but beam_size={self.extra_params.beam_size}")
            required_modules.update({"reranker": self.hparams.required_modules["reranker"]})

        if self.extra_params.get("mixv2", False):
            required_modules.update(self.hparams.required_modules['bestrq_modules'])
        
        if self.extra_params.get("chordprob_callback", False):
            required_modules.update({"chord": self.hparams.required_modules['chord']})

        self.load_required_modules(required_modules)
        if 'dualumm' in self.extra_params.token2wav_type:
            self.requires['umm'] = self.semantic_module.requires['Stage3']

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
                
        # override cache_dir with extra_params
        cache_dir = Path(self.extra_params.get('cache_dir', '.module_cache'))
        cache_dir.mkdir(exist_ok=True, parents=True)
        for k, v in self.semantic_module.hparams.required_modules.items():
            fn_partial = v['initializer']
            _, _, (f, fn_args, fn_kwargs, n) = fn_partial.__reduce__()
            fn_kwargs.update({'cache_dir': cache_dir})
            fn_partial.__setstate__((f, fn_args, fn_kwargs, n))
        
        if self.extra_params.get('app_type', None):
            self.semantic_module.load_required_modules(
                ignore=('sampler', 'diffusion', 'vocoder', 'chord', 'chord_lms', 'structure', 'asr')
            )
        else:
            self.semantic_module.load_required_modules(
                ignore=('bestrq', 'sampler', 'diffusion', 'vocoder', 'chord', 'chord_lms', 'structure', 'asr')
            )

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        if self.predict_step_seed is not None:
            pl.seed_everything(self.predict_step_seed + batch_idx)
        if "semantic_tokens" in batch:
            raw_semantic_samples = batch["semantic_tokens"]
        else:
            raw_semantic_samples = self.semantic_module.predict(
                batch,
                self.extra_params,
                beam=self.extra_params.beam_size,
            )
        semantic_samples, eos_index_list = process_eos_indexes(
            raw_semantic_samples,
            self.semantic_module,
            self.extra_params.sample_rate,
        )
        # TODO (QQ) use semantic_samples embedding as context input for decoding fn
        if self.extra_params.get("mixv2", False):
            semantic_samples = self.requires["Stage3"].model.vq.embedding(semantic_samples)
        if "duration" in batch:
            duration = batch["duration"]
        else:
            duration = self.extra_params.duration            
        if self.extra_params.token2wav_type == 'ar-diffusion-vocoder':
            # TODO, check if the key is the same as SVS,
            # SVS: style_audio, SVC, vocal_prompt
            # breakpoint()
            raw_wav_output = self.decoding_fn(self.requires, semantic_samples, prompt_wav_paths=batch.get("vocal_prompt", None))
        else:
            raw_wav_output = self.decoding_fn(self.requires, semantic_samples, self.decoding_params)
        raw_wav_output = raw_wav_output[..., :duration * self.extra_params.sample_rate]

        outputs = {}
        if self.extra_params.use_reranker:
            batch["sampled_semantic_tokens"] = raw_semantic_samples
            raw_wav_output, eos_index_list, rewards_breakdown = self.requires["reranker"].rerank(
                raw_wav_output,
                eos_index_list,
                batch,
                self.extra_params,
            )
            outputs["metadata"] = [{"rewards": x} for x in rewards_breakdown]
            # After re-ranking, sampled_semantic_tokens in batch will be sorted by reward
            raw_semantic_samples = batch["sampled_semantic_tokens"]
        
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
        return outputs


class SemanticInferenceModuleDualUMMFull(SemanticInferenceModule):
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

        # override cache_dir with extra_params
        cache_dir = Path(self.extra_params.get('cache_dir', '.module_cache'))
        cache_dir.mkdir(exist_ok=True, parents=True)
        for k, v in self.semantic_module.hparams.required_modules.items():
            if 'initializer' in v:
                fn_partial = v['initializer']
                _, _, (f, fn_args, fn_kwargs, n) = fn_partial.__reduce__()
                fn_kwargs.update({'cache_dir': cache_dir})
                fn_partial.__setstate__((f, fn_args, fn_kwargs, n))

        self.semantic_module.load_required_modules(
            ignore=('sampler', 'diffusion', 'vocoder', 'chord', 'chord_lms', 'structure', 'asr')
        )


    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        if "semantic_tokens" in batch:
            raw_semantic_samples = batch["semantic_tokens"]
        else:
            raw_semantic_samples = self.semantic_module.predict(
                batch,
                self.extra_params,
                beam=self.extra_params.beam_size,
            )
        semantic_samples, eos_index_list = process_eos_indexes(
            raw_semantic_samples,
            self.semantic_module,
            self.extra_params.sample_rate,
        )
        # TODO (QQ) use semantic_samples embedding as context input for decoding fn
        if self.extra_params.get("mixv2", False):
            semantic_samples = self.requires["Stage3"].model.vq.embedding(semantic_samples)
        if self.extra_params.token2wav_type == 'ar-diffusion-vocoder':
            # TODO, check if the key is the same as SVS,
            # SVS: style_audio, SVC, vocal_prompt
            raw_wav_output = self.decoding_fn(
                self.requires, semantic_samples,
                prompt_wav=batch.get('vocal_prompt', batch.get('style_audio'))[0])
        else:
            self.decoding_params.update({'batch': batch})
            raw_wav_output = self.decoding_fn(self.requires, semantic_samples, self.decoding_params)

        duration = self.extra_params.duration
        raw_wav_output = raw_wav_output[..., :duration * self.extra_params.sample_rate]

        outputs = {}
        if self.extra_params.use_reranker:
            batch["sampled_semantic_tokens"] = raw_semantic_samples
            raw_wav_output, eos_index_list, rewards_breakdown = self.requires["reranker"].rerank(
                raw_wav_output,
                eos_index_list,
                batch,
                self.extra_params,
            )
            outputs["metadata"] = [{"rewards": x} for x in rewards_breakdown]
            # After re-ranking, sampled_semantic_tokens in batch will be sorted by reward
            raw_semantic_samples = batch["sampled_semantic_tokens"]

        # DualUMM full track has double token length, so the actual generated audio length should be halved.
        eos_index_list = eos_index_list // 2
        raw_wav_output = raw_wav_output.unsqueeze(0)

        raw_wav_output = raw_wav_output.detach().cpu()
        wavs = truncate_wav_to_eos(raw_wav_output, eos_index_list)
        raw_semantic_samples = raw_semantic_samples.detach().cpu()
        outputs.update({
            'generated_audio': wavs,
            'generated_audio_tensor': raw_wav_output,
            'generated_semantic_tokens': raw_semantic_samples,
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
            self.decoding_params = DotDict({**self.extra_params, **extra_params['diffusion_params']})
        elif self.extra_params.token2wav_type == 'ar':
            self.decoding_fn = run_2ar
            required_modules.update(self.hparams.required_modules['ar_modules'])
            self.decoding_params = DotDict({**self.extra_params, **extra_params['ar_params']})
        else:
            raise ValueError(f"Unhandled type: {self.extra_params.token2wav_type}")

        if self.extra_params.semantic_type == 'bestrq':
            required_modules.update(self.hparams.required_modules['bestrq_modules'])
            self.encoding_fn = get_bestrq_umm_tokens

        self.load_required_modules(required_modules)

    def load_required_modules(self, required_modules):
        for name, item in required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        batch['target_audio'] = batch['style_audio']  # prepare_inputs expects target_audio key
        semantic_samples = self.encoding_fn(self.requires, batch['target_audio'])
        wavs = self.decoding_fn(self.requires, semantic_samples, self.decoding_params)  #
        return {
            'generated_audio': wavs,
            'generated_audio_tensor': wavs
        }
