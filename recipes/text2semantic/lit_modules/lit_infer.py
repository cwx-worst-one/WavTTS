import logging
import os
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

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
from .llama.lit_vae_t2s_ctiga_lang_spk import VAET2SLangSpkModule
from ..utils.remote_io import load_json
from transformers import LlamaTokenizer, T5Tokenizer, AutoTokenizer
from zhon.hanzi import punctuation
import string
punctuation_all = punctuation + string.punctuation

logger = logging.getLogger(__name__)


def model_loader(name, ckpt_path):
    if name == "VAET2SModule":
        return VAET2SModule.load_from_checkpoint(checkpoint_path=ckpt_path).eval()
    elif name == "VAET2SLangSpkModule":
        return VAET2SLangSpkModule.load_from_checkpoint(checkpoint_path=ckpt_path).eval()
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
        scale_generated_wav=False,
        tacolab_version='oldv1', # oldv1, newv3, newv3_punc
        text2id_version='v1',
        prompt_tacolab_dir="",
        infer_tacolab_dir="",
        use_sy=False,
        use_lang_id=False,
        lang_tokens_num=0,
        lang2id=None,
        use_spk_id=False,
        spk2id='',
        spkname='',
        use_bpe=False,
        bpe_tokens_num=0,
        bpe_dir='',
        tokenizer_type='',
        max_length=4096
    ):
        super().__init__()
        assert (text2id_version == 'v1' and tacolab_version == 'oldv1') \
            or (text2id_version == 'v1' and tacolab_version == 'newv3') \
            or (text2id_version == 'v2' and tacolab_version == 'newv3') \
            or (text2id_version == 'v2' and tacolab_version == 'newv3_punc') \
            or (text2id_version == 'v3' and tacolab_version == 'newv3') \
            or (text2id_version == 'v3' and tacolab_version == 'newv3_punc'), \
                (text2id_version, tacolab_version)

        self.save_hyperparameters()
        self.ar_model = model_loader(ar_model_name, ckpt_path).eval()
        self.text2id = TextToTacolabID(text2id_path, text2id_version=text2id_version, use_sy=use_sy, tacolab_version=tacolab_version)
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            phone_token_num=phone_tokens_num, speaker_token_num=speaker_tokens_num, bpe_tokens_num=bpe_tokens_num, lang_tokens_num=lang_tokens_num
        )
        self.tacolab_version = tacolab_version
        self.prompt_tacolab_dir = prompt_tacolab_dir
        self.infer_tacolab_dir = infer_tacolab_dir
        self.use_sy = use_sy

        # lang
        self.use_lang_id = use_lang_id
        if self.use_lang_id:
            self.lang2id = load_json(lang2id)
            logger.info(f"Loaded lang2id from {lang2id}")
        else:
            self.lang2id = None

        self.use_bpe = use_bpe
        self.use_spk_id = use_spk_id

        if self.use_bpe:
            logger.info(f'##### Using BPE #####')
            if tokenizer_type == "flan-T5-large":
                self.bpe_tokenizer = T5Tokenizer.from_pretrained(bpe_dir)
            elif tokenizer_type == "byte-T5-base":
                self.bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_dir)
            elif tokenizer_type == "llama":
                self.bpe_tokenizer = LlamaTokenizer.from_pretrained(bpe_dir)
            else:
                raise NotImplementedError(tokenizer_type)
        else:
            self.bpe_tokenizer = None

        if self.use_spk_id:
            self.spk2id = load_json(spk2id)
            logger.info(f"Loaded spk2id from {spk2id}")
            self.spk_id = self.spk2id.get(spkname, self.spk2id["default"])
            print("self.spk_id: ", self.spk_id)
            self.spk_id = self.tokenizer.tokenize(self.spk_id, "spk")
        else:
            self.spk2id = None
        self.max_length = max_length

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

        prompt_wav_max = None
        prompt_wav = None
        if not self.use_spk_id:
            uttid, prompt_text, prompt_wav_path, text = sample
            wav, sr = librosa.load(prompt_wav_path, sr=None)
            if len(wav.shape) == 2 and wav.shape[-1] == 2:
                wav = wav[:, 0]
            prompt_wav_max = np.max(np.abs(wav))
            if sr != 24_000:
                wav = librosa.core.resample(wav, sr, 24_000)
            wav = wav * 0.99 / max(0.01, np.max(np.abs(wav)))
            wav = trim_prompt_silence(wav)
            prompt_wav = wav.copy()
            wav = torch.from_numpy(wav).float()
            wav = wav.to(device)
            wav = torch.stack([wav]).unsqueeze(1).float()
            wav = F.pad(wav, (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1), 0, 0, 0, 0), value=0.)
            # wav = torch.stack([wav]).unsqueeze(1)
            spec = spectrogram_torch(wav.squeeze(1), 2048, 24000, 300, 1200)
            _, m, logs = self.wvae_encoder(wav, spec)
            m = m.transpose(2, 1)
            logs = logs.transpose(2, 1)
            bn = torch.cat([m, logs], -1)  # (1, t, 64)
        else:
            uttid, text = sample
            bn = np.zeros([0, 32])
            bn = torch.from_numpy(bn)

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        # get text_id
        if self.tacolab_version == 'oldv1':
            text_id = self.text2id(text=text)
            if prompt_text is not None:
                prompt_text_id = self.text2id(text=prompt_text)
                text_id = np.hstack([prompt_text_id[:-1], 2, text_id[1:]])
        elif self.tacolab_version in ['newv3', 'newv3_punc']:
            if not self.use_spk_id:
                assert self.prompt_tacolab_dir != ""
                assert self.infer_tacolab_dir != ""
                prompt_utt = prompt_wav_path.split('/')[-1][:-4]
                infer_utt = uttid
                prompt_tacolab_path = os.path.join(self.prompt_tacolab_dir, prompt_utt + '.lab')
                infer_tacolab_path = os.path.join(self.infer_tacolab_dir, infer_utt + '.lab')
                if not os.path.exists(prompt_tacolab_path) or not os.path.exists(infer_tacolab_path):
                    print("prompt_tacolab_path or infer_tacolab_path not exists, skip", prompt_tacolab_path, infer_tacolab_path)
                    return None
                with open(prompt_tacolab_path, 'r', encoding="utf-8") as f:
                    prompt_tacolab = [x.strip('\n ') for x in f.readlines()]
                with open(infer_tacolab_path, 'r', encoding="utf-8") as f:
                    infer_tacolab = [x.strip('\n ') for x in f.readlines()]
                tacolab = '\n'.join(prompt_tacolab + infer_tacolab[1:])
                tacolab_list = list(filter(lambda x: x != "", tacolab.split('\n')))
                text_id = self.text2id(tacolab_list=tacolab_list)
            else:
                assert self.infer_tacolab_dir != ""
                infer_utt = uttid
                infer_tacolab_path = os.path.join(self.infer_tacolab_dir, infer_utt + '.lab')
                if not os.path.exists(infer_tacolab_path):
                    print("infer_tacolab_path not exists, skip", infer_tacolab_path)
                    return None
                with open(infer_tacolab_path, 'r', encoding="utf-8") as f:
                    infer_tacolab = [x.strip('\n ') for x in f.readlines()]
                tacolab = '\n'.join(infer_tacolab[1:])
                tacolab_list = list(filter(lambda x: x != "", tacolab.split('\n')))
                text_id = self.text2id(tacolab_list=tacolab_list)

        if text_id is None:
            logger.warning(f"{uttid} TextToTacolabID failed ...")
            return None
        text_id = self.tokenizer.tokenize(text_id, "inputs")

        # add bpe_id
        if not self.use_spk_id:
            if prompt_text[-1] in punctuation_all:
                text = prompt_text + ' ' + text.strip()
            else:
                text = prompt_text + ', ' + text
        else:
            text = text

        if self.use_bpe:
            bpe_id = np.asarray(self.bpe_tokenizer(
                text, truncation=True, max_length=self.max_length,
            ).input_ids)
            bpe_id = self.tokenizer.tokenize(bpe_id, "bpe")

            # text_id = np.concatenate([text_id, [self.tokenizer.sep], bpe_id])
            text_id = np.concatenate([
                bpe_id, 
                [self.tokenizer.sep], 
                text_id])

        if self.use_lang_id:
            lang_key = self.get_lang_by_text(text)
            if lang_key == None:
                logger.info(f"{text}: Wrong lang_key")
                return None
            lang_id = self.lang2id[lang_key]
            lang_id = self.tokenizer.tokenize(lang_id, "lang")

        bn_T, bn_C = bn.shape[0], bn.shape[1]

        # [bos] bpe phone [sep] spk_id langid wav [eos]
        if self.use_spk_id:
            wav_id = np.asarray([self.spk_id])
        else:
            wav_id = np.zeros([bn_T], dtype=np.int64)

        if self.use_lang_id:
            wav_id = np.concatenate([
                [lang_id],
                wav_id])

        seq = (
            [self.tokenizer.bos]
            + list(text_id)
            + [self.tokenizer.sep]
            + list(wav_id)  # place holder
        )
        # output_dir = f"{self.hparams.output_dir}" + '/../inputs'
        # os.makedirs(output_dir, exist_ok=True)
        # np.save(os.path.join(output_dir, uttid + '.npy'), np.asarray(seq))

        # logger.info("seq: ", seq)
        # exit()
        text_id = torch.tensor(text_id).long()
        seq = torch.tensor(seq).long().to(device)
        return (
            text_id.unsqueeze(0),
            torch.tensor([text_id.shape[0]]).long().unsqueeze(0),
            bn.unsqueeze(0).to(device),
            torch.tensor([bn.shape[0]]).long().unsqueeze(0),
            seq.unsqueeze(0),
            torch.tensor([seq.shape[0]]).long().unsqueeze(0),
            [uttid],
        ), prompt_wav.squeeze(), prompt_wav_max

    def is_english_char(self, char):
        if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a'):
            return True
        else:
            return False

    def is_english_spanish_char(self, char):
        special_Spanish_chars_list = ['á', 'é', 'í', 'ó', 'ú', 'Á', 'É', 'Í', 'Ó', 'Ú', 'ñ', 'Ñ', '¡', '¿', 'ü', 'Ü']
        if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a') or char in special_Spanish_chars_list:
            return True
        else:
            return False

    def get_lang_by_text(self, text):
        text = text.replace('\'', '')
        # en, zh
        len_en_word = 0
        len_zh_char = 0
        i = 0
        while i < len(text):
            x = text[i]
            if x in punctuation_all: # punc
                i += 1
                continue
            elif u'\u4e00' <= x <= u'\u9fff': # zh
                len_zh_char += 1
                i += 1
            elif self.is_english_spanish_char(x): # en with little spanish
                i += 1
                if i >= len(text):
                    len_en_word += 1
                    break
                while self.is_english_spanish_char(text[i]):
                    i += 1
                    if i >= len(text):
                        break
                len_en_word += 1
                continue
            else: # blank or digit
                if not (text[i] == " " or text[i].isdigit()):
                    return None
                i += 1

        lang = 'en'
        if len_zh_char > len_en_word:
            lang = 'zh'

        return lang



# if __name__ == "__main__":
    # lit_infer = BigTTSWVAEInfer('VAET2SModule',
    #                             ckpt_path,
    #                             wvae_encoder,
    #                             wvae_decoder,
    #                             output_dir,
    #                             phone_tokens_num=7370,
    #                             speaker_tokens_num=8192,
    #                             text2id_path="recipes/valle/datasets/dict/metaid_to_textid.json",
    #                             module_cache=".module_cache",
    #                             seed=1996,
    #                             tacolab_version='oldv1', # oldv1, newv3, newv3_punc
    #                             text2id_version='v1',
    #                             prompt_tacolab_dir="",
    #                             infer_tacolab_dir="",
    #                             use_sy=False,
    #                             use_lang_id=False,
    #                             lang_tokens_num=0,
    #                             lang2id=None,
    #                             use_spk_id=False,
    #                             spk2id='',
    #                             spkname='',
    #                             use_bpe=False,
    #                             bpe_tokens_num=0,
    #                             bpe_dir='',
    #                             max_length=4096)

    # collector = ContinuousCollator(tokenizer_pad=0)
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=None, collate_fn=collector)
    # import tqdm

    # # for item in tqdm.tqdm(dataset):
    # for item in tqdm.tqdm(dataloader):
    #     # logger.info(item)
    #     # lengths = item[-1]
    #     # batch_size = len(lengths)
    #     # logger.info(batch_size, max(lengths), batch_size * max(lengths))
    #     exit()
