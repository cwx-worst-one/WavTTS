import os

import pytorch_lightning as pl
import torch

from recipes.bigmusic.lightning.semantic_modules import SemanticModule
from recipes.bigmusic.lightning.phoneme_coarse_modules import LyricsCoarseModule
from recipes.bigmusic.lightning.acoustic_modules import CoarseModule
# from recipes.musiclm.lightning.modules import CoarseModule
from recipes.musiclm.lightning.modules import FineModule
from samantha.utils.hparams import DotDict
from recipes.musiclm.inference.utils import slugify, save_wav

class SemanticInferenceModule(pl.LightningModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)

        self.semantic_module = SemanticModule.load_from_checkpoint(self.extra_params.semantic_ckpt).eval()
        self.coarse_module = CoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
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
        self.semantic_module.load_required_modules()
    
    def _predict_step(self, batch, round):
        semantic_samples = self.semantic_module.predict(batch, self.extra_params)
        coarse_samples = self.coarse_module.predict(semantic_samples, self.extra_params)
        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)
        bs = coarse_samples.size(0)
        
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
        save_outputs(wavs, batch, round, self.extra_params.output_dir)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.extra_params.num_rounds):
            self._predict_step(batch, i)

class ConditionalMulanPhonemeInferenceModule(pl.LightningModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.coarse_module = LyricsCoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
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
        self.coarse_module.load_required_modules()
    
    def _predict_step(self, batch, round):
        coarse_samples = self.coarse_module.predict(batch, self.extra_params)
        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)
        bs = coarse_samples.size(0)
        
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
        save_outputs(wavs, batch, round, self.extra_params.output_dir)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.extra_params.num_rounds):
            self._predict_step(batch, i)

def save_outputs(wavs, conditions, batch, round, output_dir):
    conditions = batch['conditions']
    lyrics = batch['lyrics']
    prompts = batch['mulan_text']
    mulan_audio = batch.get('mulan_audio', None)
    vocal_audio = batch.get('vocal_audio', None)
    for i, wav in enumerate(wavs):
        wav_dir = os.path.join(output_dir)
        os.makedirs(wav_dir, exist_ok=True)
        file_name = ""
        if 'lyrics_tokens' in conditions:
            file_name += slugify(lyrics[i])[:128]
        if 'mulan_text' in conditions:
            file_name += '--' + slugify(prompts[i])[:128]
        if file_name: 
            file_name += f'.{round}-{i}'
        else:
            file_name += f'{round}-{i}'
        wav_fp = os.path.join(wav_dir, f"{file_name}.wav")
        print(f"[Saving] {wav_fp}")
        save_wav(wav.cpu().float(), wav_fp, sr=24000)

        txt_fp = os.path.join(wav_dir, f"{file_name}.txt")
        with open(txt_fp, 'w') as f:
            f.write(f'Conditions: {conditions}\n')
            if 'lyrics_tokens' in conditions:
                f.write(f'Lyrics: {lyrics[i]}\n')
            if 'mulan_text' in conditions:
                f.write(f'Prompt: {prompts[i]}\n')

        if mulan_audio is not None and 'audio_prompt' in conditions:
            input_wav_fp = os.path.join(wav_dir, f"{file_name}.audio_prompt.wav")
            save_wav(mulan_audio[i].cpu().float(), input_wav_fp, sr=24000)

        if vocal_audio is not None and 'vocal_audio_prompt' in conditions:
            input_vocals_fp = os.path.join(wav_dir, f"{file_name}.vocal_prompt.wav")
            save_wav(vocal_audio[i].cpu().float(), input_vocals_fp, sr=24000)
