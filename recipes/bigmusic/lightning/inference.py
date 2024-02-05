import pytorch_lightning as pl

from samantha.utils.hparams import DotDict
from recipes.diffusion.models.diffusion_model.utils import run_diffusion
from recipes.soundstorm.lightning.utils import run_soundstorm
from recipes.bigmusic.lightning.embedding_modules import get_bestrq_umm_tokens
import importlib
from recipes.bigmusic.utils.model_initializer import run_2ar
from recipes.bigmusic.lightning.base_modules import BaseModule
from recipes.bigmusic.lightning.semantic_modules import process_eos_indexes, truncate_wav_to_eos
from samantha.models.flash_llama import LlamaPreTrainedModel
from samantha.models.ctiga import gpt
import logging


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

        self.semantic_module: BaseModule = semantic_class.load_from_checkpoint(
            self.extra_params.semantic_ckpt,
            # pay attention to the logs to make sure the model is loaded correctly
            strict=True,
        ).eval()
        if cls_name == "SemanticModuleVarlenXperf" or cls_name == "SemanticModuleXperf":
            self.semantic_module.replace_ctiga_to_xperf()
        self.requires = {}

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
            self.decoding_fn = run_diffusion
            required_modules.update(self.hparams.required_modules['diffusion_modules'])
            self.decoding_params = DotDict({**self.extra_params, **self.extra_params['diffusion_params']})
        elif self.extra_params.token2wav_type == 'ar':
            self.decoding_fn = run_2ar
            required_modules.update(self.hparams.required_modules['ar_modules'])
            self.decoding_params = DotDict({**self.extra_params, **self.extra_params['ar_params']})
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

        self.load_required_modules(required_modules)

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
        
        if self.extra_params.get('app_type', None):
            self.semantic_module.load_required_modules(
                ignore=('sampler', 'diffusion', 'vocoder', 'chord', 'chord_lms', 'structure', 'asr')
            )
        else:
            self.semantic_module.load_required_modules(
                ignore=('bestrq', 'sampler', 'diffusion', 'vocoder', 'chord', 'chord_lms', 'structure', 'asr')
            )

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
        # TODO (QQ) use semantic_samples embedding as context input for decoding fn
        if self.extra_params.get("mixv2", False):
            semantic_samples = self.requires["Stage3"].model.vq.embedding(semantic_samples)
        raw_wav_output = self.decoding_fn(self.requires, semantic_samples, self.decoding_params)
        assert len(raw_wav_output.shape) == 2, "Wavs must be 2 sim [b, seq_len]"
        if "duration" in batch:
            duration = batch["duration"]
        else:
            duration = self.extra_params.duration
        raw_wav_output = raw_wav_output[..., :duration * self.extra_params.sample_rate]

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
