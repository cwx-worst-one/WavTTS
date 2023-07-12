import os

import pytorch_lightning as pl
import torch

from recipes.musiclm.lightning.modules import SemanticModule
from recipes.l2v.lightning.lyrics_modules import MulanPhonemeCoarseModule, ConditionalMulanPhonemeCoarseModule, LyricsSemanticEmbedModule, EmbedMulanPhonemeCoarseModule
# from recipes.l2v.lightning.bestrq_modules import BestRQCoarseCrossAttnModule
from recipes.musiclm.lightning.modules import CoarseModule, CoarseCrossAttnModule
from recipes.musiclm.lightning.modules import FineModule
from recipes.musiclm.lightning.inference import BaseModule
from samantha.utils.hparams import DotDict
from recipes.musiclm.inference.utils import slugify, save_wav
from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization


# TODO: remove this once coarse modules have been re-trained with fixed semantic_type
class CoarseCrossAttnModuleFix(CoarseCrossAttnModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        if extra_params['semantic_type'] == 'best_rq_minz':
            extra_params['semantic_type'] = 'best_rq'
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
class GTInferenceModule(BaseModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.gt_infer_mode = self.extra_params.gt_infer_mode
        self.semantic_type = self.extra_params.get("semantic_type", "wav2vec")
        self.use_continuous_embedding = self.extra_params.get("use_continuous_embedding", False)
        if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "semantic->coarse->fine", "coarse->fine", "all"]:
            self.fine_module = FineModule.load_from_checkpoint(self.extra_params.fine_ckpt).eval()
            if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "semantic->coarse->fine", "all"]:
                if self.use_continuous_embedding:
                    self.coarse_module = CoarseCrossAttnModuleFix.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
                else:
                    self.coarse_module = CoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
                if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "all"]:
                    self.semantic_module = LyricsSemanticEmbedModule.load_from_checkpoint(self.extra_params.semantic_ckpt).eval()
        else:
            raise ValueError(f"Invalid ground truth inference mode: {self.gt_infer_mode}")

        if self.use_continuous_embedding:
            if self.semantic_type == "wav2vec":
                self.semantic_fn = self.get_wav2vec_embeds
            elif self.semantic_type == "best_rq":
                self.semantic_fn = self.get_best_rq_embeds
            else:
                raise KeyError(f"Invalid semantic_type, got {self.semantic_type}")
        else:
            if self.semantic_type == "wav2vec":
                self.semantic_fn = self.get_wav2vec_tokens
            elif self.semantic_type == "best_rq":
                self.semantic_fn = self.get_best_rq_tokens
            else:
                raise KeyError(f"Invalid semantic_type, got {self.semantic_type}")
        self.requires = {}
        self.load_required_modules()
    
    def _load_required_module(self, name):
        print('Loading required module:', name)
        item = self.hparams.required_modules[name]
        if isinstance(item, (list, tuple)):
            hpath, initializer = item
        elif isinstance(item, dict):
            hpath, initializer = item['hpath'], item['initializer']
        self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def load_required_modules(self):
        if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "semantic->coarse->fine", "coarse->fine", "all"]:
            self._load_required_module("soundstream_dec")
            if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "all"]:
                self._load_required_module("mulan")
                self._load_required_module("mulan_centers")
            if self.gt_infer_mode in ["semantic->coarse->fine", "all"]:
                if self.extra_params.semantic_type == "best_rq":
                    self._load_required_module("best_rq")
                if self.extra_params.semantic_type == "wav2vec":
                    self._load_required_module("wav2vec")
                    self._load_required_module("semantic_centers")
            if self.gt_infer_mode in ["coarse->fine", "all"]:
                self._load_required_module("soundstream")

    def _predict_step(self, batch, batch_idx, round):
        if torch.is_tensor(batch): # handle musiclm format
            if batch.dim() == 3:
                batch = batch.squeeze(1)
            batch = { 'mulan_audio': batch }
        lyrics = batch.get('lyrics')
        lyrics_tokens = batch.get('lyrics_tokens')
        if lyrics_tokens is not None:
            lyrics_tokens = lyrics_tokens.to(self.device)
        mulan_audio = batch.get('mulan_audio', None).to(self.device)
        
        device = mulan_audio.device
        bs = mulan_audio.size(0)

        if self.extra_params.gt_infer_mode == "mulan->semantic->coarse->fine":
            mulan_ids = self.get_mulan_embeds(mulan_audio) # TODO: make this flexible to switch between tokens
            semantic_tokens = self.semantic_module.predict(mulan_ids, self.extra_params, lyrics_tokens)
            coarse_samples = self.coarse_module.predict(semantic_tokens, self.extra_params)
        elif self.extra_params.gt_infer_mode == "semantic->coarse->fine":
            semantic_tokens = self.semantic_fn(mulan_audio)
            coarse_samples = self.coarse_module.predict(semantic_tokens, self.extra_params)
        elif self.extra_params.gt_infer_mode == "coarse->fine":
            soundstream_ids = self.get_soundstream_tokens(mulan_audio)
            coarse_samples = (
                soundstream_ids[:, :, 0 : self.extra_params.num_coarse]
                + torch.arange(self.extra_params.num_coarse, device=device) * self.extra_params.soundstream_codebook_size
            ).reshape((bs, -1))

        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)

        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_coarse + self.extra_params.num_fine, device=coarse_samples.device)
            * self.extra_params.soundstream_codebook_size
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        output_wavs = self.requires["ss_dec"](vqgan_inputs).squeeze(1)
        subdir_name = self.extra_params.gt_infer_mode.replace('->', '_')
        for i, output_wav in enumerate(output_wavs):
            wav_dir = os.path.join(self.extra_params.output_dir, subdir_name, 'nophoneme')
            os.makedirs(wav_dir, exist_ok=True)
            fp = os.path.join(wav_dir, f"{batch_idx * bs + i}")
            print(f"[Saving] {fp}")
            save_wav(output_wav.cpu(), f"{fp}.{round}.wav", sr=24000)
            save_wav(mulan_audio[i].cpu(), f"{fp}.wav", sr=24000)
            with open(f"{fp}.txt", 'w') as f:
                f.write(f'Lyrics: {lyrics}')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.extra_params.num_rounds):
            self._predict_step(batch, batch_idx, i)

class ConditionalMulanPhonemeInferenceModule(pl.LightningModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.coarse_module = EmbedMulanPhonemeCoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
        self.fine_module = FineModule.load_from_checkpoint(self.extra_params.fine_ckpt).eval()
        self.requires = {}
        self.load_required_modules()

    def load_required_modules(self):
        for name, item in self.hparams.required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
            self.coarse_module.requires = self.requires
            # if name == "mulan_centers":
            #     assert self.extra_params.mulan_num_rvq == self.requires["mulan_centers"].shape[0]
    
    def _predict_step(self, batch, round):
        lyrics = batch['lyrics']
        lyrics_tokens = batch['lyrics_tokens'].to(self.device)
        prompts = batch['mulan_text']
        mulan_audio = batch.get('mulan_audio', None)
        vocal_audio = batch.get('vocal_audio', None)
        vocal_chroma = batch.get('vocal_chroma', None)
        bs = lyrics_tokens.size(0) if lyrics_tokens is not None else mulan_audio.size(0)
        conditions = self.extra_params.conditions.split(',')
        coarse_samples = self.coarse_module.predict(batch, self.extra_params, conditions)
        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)
        
        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_coarse + self.extra_params.num_fine, device=coarse_samples.device)
            * self.extra_params.soundstream_codebook_size
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = self.requires["ss_dec"](vqgan_inputs).squeeze(1)
        for i, wav in enumerate(wavs):
            wav_dir = os.path.join(self.extra_params.output_dir)
            os.makedirs(wav_dir, exist_ok=True)
            file_name = ""
            if 'lyrics' in conditions:
                file_name += slugify(lyrics[i])[:128]
            if 'text_prompt' in conditions:
                file_name += '--' + slugify(prompts[i])[:128]
            if file_name: 
                file_name += f'.{round}-{i}'
            else:
                file_name += f'{round}-{i}'
            wav_fp = os.path.join(wav_dir, f"{file_name}.wav")
            print(f"[Saving] {wav_fp}")
            save_wav(wav.cpu(), wav_fp, sr=24000)

            txt_fp = os.path.join(wav_dir, f"{file_name}.txt")
            with open(txt_fp, 'w') as f:
                f.write(f'Conditions: {conditions}\n')
                if 'lyrics' in conditions:
                    f.write(f'Lyrics: {lyrics[i]}\n')
                if 'text_prompt' in conditions:
                    f.write(f'Prompt: {prompts[i]}\n')
                if 'vocal_chroma' in conditions:
                    f.write(f'Chroma: {vocal_chroma[i]}\n')

            if mulan_audio is not None and 'audio_prompt' in conditions:
                input_wav_fp = os.path.join(wav_dir, f"{file_name}.audio_prompt.wav")
                save_wav(mulan_audio[i].cpu(), input_wav_fp, sr=24000)

            if vocal_audio is not None and 'mulan_vocals' in conditions:
                input_vocals_fp = os.path.join(wav_dir, f"{file_name}.vocal_prompt.wav")
                save_wav(vocal_audio[i].cpu(), input_vocals_fp, sr=24000)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.extra_params.num_rounds):
            self._predict_step(batch, i)
