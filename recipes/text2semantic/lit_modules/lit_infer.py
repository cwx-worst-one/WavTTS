import logging
import os
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from scipy.io.wavfile import write

from ..datasets import PhoneTokenizerWithAudioTokens
from ..datasets.text_converter import TextToTacolabID
from ..scripts.infer_utils import (
    load_torch_script,
    setup_seed,
    spectrogram_torch,
    trim_silence,
    trim_prompt_silence,
    save_wav
)
from .llama.lit_vae_t2s_ctiga import VAET2SModule

logger = logging.getLogger(__name__)


def model_loader(name, ckpt_path):
    if name == "VAET2SModule":
        return VAET2SModule.load_from_checkpoint(checkpoint_path=ckpt_path).eval()
    else:
        raise ValueError(f"{name} is not supported.")


class BigTTSWVAEInfer(LightningModule):
    def __init__(
        self,
        ar_model_name,
        ckpt_path,
        wvae_encoder,
        wvae_decoder,
        output_dir,
        phone_tokens_num=7370,
        speaker_tokens_num=8192,
        text2id_path="recipes/valle/datasets/dict/metaid_to_textid.json",
        module_cache=".module_cache",
        seed=1996,
        save_prompt=False,
        trim_generated_wav=False,
        scale_generated_wav=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ar_model = model_loader(ar_model_name, ckpt_path).eval()
        self.text2id = TextToTacolabID(text2id_path)
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            phone_token_num=phone_tokens_num, audio_token_num=speaker_tokens_num
        )
        self.save_prompt = save_prompt
        self.trim_generated_wav = trim_generated_wav
        self.scale_generated_wav = scale_generated_wav

    def setup(self, stage):
        if stage == "predict":
            self.wvae_encoder = load_torch_script(
                model_path=self.hparams.wvae_encoder,
                rank=self.trainer.local_rank,
                cache_dir=self.hparams.module_cache,
            )
            self.wvae_decoder = load_torch_script(
                model_path=self.hparams.wvae_decoder,
                rank=self.trainer.local_rank,
                cache_dir=self.hparams.module_cache,
            )

    def _decode(self, z_outputs):
        z_outputs = z_outputs.transpose(2, 1)
        generated_wav = self.wvae_decoder(z_outputs).squeeze()
        return generated_wav

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        setup_seed(self.hparams.seed)
        sample, prompt_wav, prompt_wav_max = self.encode(batch)
        utt_ids = sample[-1]
        z_outputs, _ = self.ar_model.predict(sample, None)
        generated_wav = self._decode(z_outputs)
        output_dir = f"{self.hparams.output_dir}"
        os.makedirs(output_dir, exist_ok=True)

        generated_wav = generated_wav.cpu().numpy()
        if self.trim_generated_wav:
            generated_wav = trim_silence(generated_wav)
        if self.scale_generated_wav:
            generated_wav *= min(0.99, prompt_wav_max) / max(0.01, np.max(np.abs(generated_wav)))
        save_wav(generated_wav, f"{output_dir}/{utt_ids[0]}.wav", 24000)
        if self.save_prompt:
            save_wav(np.concatenate((prompt_wav, generated_wav), axis=0), f"{output_dir}/prompt-{utt_ids[0]}.wav", 24000)

    def encode(self, sample):
        device = f"cuda:{self.trainer.local_rank}"
        uttid, prompt_text, prompt_wav_path, text = sample
        wav, sr = librosa.load(prompt_wav_path, sr=None)
        prompt_wav_max = np.max(np.abs(wav))
        if sr != 24_000:
            wav = librosa.core.resample(wav, sr, 24_000)
        wav = wav * 0.99 / max(0.01, np.max(np.abs(wav)))
        wav = trim_prompt_silence(wav)
        prompt_wav = wav.copy()
        wav = torch.from_numpy(wav).float()
        wav = F.pad(wav, (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1)))
        wav = torch.stack([wav]).unsqueeze(1)
        spec = spectrogram_torch(wav.squeeze(1), 2048, 24000, 300, 1200)
        _, m, logs = self.wvae_encoder(wav.to(device), spec.to(device))
        m = m.transpose(2, 1)
        logs = logs.transpose(2, 1)
        bn = torch.cat([m, logs], -1)  # (1, t, 64)
        symbol_sets = ['.', ',', '?', '!', '，', '。', '？', '！']
        if prompt_text[-1] in symbol_sets:
            text = prompt_text + ' ' + text.strip()
        else:
            text = prompt_text + ', ' + text
        text_id = self.text2id(text)

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        if text_id is None:
            logger.warning(f"{uttid} TextToTacolabID failed ...")
            return None

        text_id = self.tokenizer.tokenize(text_id, "inputs")

        bn_T, bn_C = bn.shape[0], bn.shape[1]

        seq = (
            [self.tokenizer.bos]
            + list(text_id)
            + [self.tokenizer.sep]
            + [0] * bn_T  # place holder
        )

        text_id = torch.tensor(text_id).long()
        seq = torch.tensor(seq).long().to(device)
        return (
            text_id.unsqueeze(0),
            torch.tensor([text_id.shape[0]]).long().unsqueeze(0),
            bn.unsqueeze(0).to(device),
            torch.tensor([bn.shape[0]]).long().unsqueeze(0),
            seq.unsqueeze(0),
            torch.tensor([seq.shape[0]]).long().unsqueeze(0),
            [uttid]
        ), prompt_wav.squeeze(), prompt_wav_max
