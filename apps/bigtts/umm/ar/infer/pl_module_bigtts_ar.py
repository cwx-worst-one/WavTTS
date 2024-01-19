import os
import random
from typing import Any

import librosa
import numpy as np
import torch
from apps.bigtts.umm.ar.data.lyrics import (AddConditionsTransform,
                                            LyricsTokenTransform)
from apps.bigtts.umm.ar.infer.semantic_modules import SemanticModule_Valle
from pytorch_lightning import LightningModule

from samantha.utils.hparams import DotDict

#torch.backends.cuda.matmul.allow_tf32 = True

def set_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)

def prepare_semantic_model(semantic_model_path, device):
    semantic_model = SemanticModule_Valle.load_from_checkpoint(semantic_model_path).to(device).eval()
    return semantic_model

def prepare_umm(umm_ckpt_path, device):
    rank = int(device[-1])
    from recipes.umm.requires.model_initializer import init_stage3
    token_model = init_stage3(umm_ckpt_path, rank, "./")["Stage3"].eval()
    return token_model

class SemanticInference(LightningModule):
    def __init__(
        self,
        seed,
        semantic_precision,
        umm_ckpt_path,
        semantic_model_path,
        output_path,
        src_lang,
        tgt_lang,
        temperature=0.9,
        thresh=0.9,
        mode='naive',
        max_blank_length=5,
        step_out_blank: bool=False
    ):
        super().__init__()
        if semantic_precision == "bf16":
            self.semantic_precision = torch.bfloat16
        elif semantic_precision == "fp16":
            self.semantic_precision = torch.float16
        elif semantic_precision == "fp32":
            self.semantic_precision = torch.float32

        self.umm_ckpt_path = umm_ckpt_path
        self.output_path = output_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lang2id = {
            "en": 0,
            "zh": 1,
            "zh_en": 2,
        }

        self.seed = seed
        seed = set_seed(seed)
        self.semantic_model = prepare_semantic_model(semantic_model_path, self.device)

        self.semantic_ar_model_hp = DotDict(
            {
                "duration": 60,
                "semantic_temperature": temperature,
                "semantic_thresh": thresh,
                "sample_mode": mode,
                "max_blank_length": max_blank_length,
                "step_out_blank": step_out_blank,
            }
        )
        lyrics_max_seq_len = self.semantic_model.extra_params.get("lyrics_max_seq_len", 600)
        self.lyrics_token_transform = LyricsTokenTransform.init_sami_tokenizer(
            lyrics_max_seq_len=lyrics_max_seq_len, truncate_long_lyrics=True,
            test_wer=True)
        self.add_conditions_transform = AddConditionsTransform("lyrics_tokens")

    def preprocess_prompt_wav(self, prompt_wav_path, device):
        wav, sr = librosa.load(prompt_wav_path, sr=24000, mono=True)
        wav = torch.FloatTensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        if sr != 24_000:
            wav = librosa.core.resample(wav.numpy(), sr, 24_000)
            wav = torch.from_numpy(wav)
        wav = wav.to(device)
        return wav

    def text2semantic(self, prompt_wav, prompt_lab, infer_lab):
        semantic_batch = {}
        prompt_umm_token = self.umm.wav2token(prompt_wav)
        # hard-code from @kainan.
        prompt_umm_token = prompt_umm_token[:,:-1]
        semantic_batch['audio_prompt'] = prompt_umm_token
        # prepare text-ids.
        semantic_batch['lyrics'] = prompt_lab + "|" + infer_lab
        semantic_batch = self.lyrics_token_transform(semantic_batch)
        semantic_batch = self.add_conditions_transform(semantic_batch)
        # batching.
        semantic_batch['lyrics_tokens'] = semantic_batch['lyrics_tokens'].unsqueeze(0)
        semantic_batch['prompt_text_lens'] = semantic_batch['prompt_text_lens'].unsqueeze(0)
        semantic_batch['phones'] = semantic_batch['phones'].unsqueeze(0)
        semantic_batch['tones'] = semantic_batch['tones'].unsqueeze(0)
        semantic_batch['wordsegs'] = semantic_batch['wordsegs'].unsqueeze(0)
        # language.
        semantic_batch['src_lang'] = torch.LongTensor([self.lang2id.get(self.src_lang)])
        semantic_batch['tgt_lang'] = torch.LongTensor([self.lang2id.get(self.tgt_lang)])
        torch.backends.cuda.matmul.allow_tf32 = True # hard-code
        with torch.autocast(device_type="cuda", dtype=self.semantic_precision, enabled=True):
            infer_umm_token = self.semantic_model.predict(semantic_batch, self.semantic_ar_model_hp)
        torch.backends.cuda.matmul.allow_tf32 = False # hard-code
        return infer_umm_token

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        set_seed(self.seed)
        uttid, prompt_lab, prompt_wav_path, infer_lab = batch
        prompt_wav = self.preprocess_prompt_wav(prompt_wav_path, self.device)

        infer_umm_token = self.text2semantic(prompt_wav, prompt_lab, infer_lab)
        if infer_umm_token is not None:
            infer_umm_token = infer_umm_token.detach().cpu().numpy()
            np.save(
                os.path.join(self.output_path, uttid+".npy"),
                infer_umm_token,
                allow_pickle=False,
            )

    def setup(self, stage):
        device = f"cuda:{self.trainer.local_rank}"
        self.umm = prepare_umm(self.umm_ckpt_path, device)
