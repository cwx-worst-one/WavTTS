import logging
from multiprocessing.sharedctypes import Value
import os
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from recipes.text2semantic.scripts.infer_utils import (
    spectrogram_torch,
    trim_silence,
)
from scipy.io.wavfile import read
from zhon.hanzi import punctuation
import string

punctuation_all = punctuation + string.punctuation
from recipes.text2semantic.lit_modules.lit_infer import BigTTSWVAEInfer
from recipes.text2semantic.datasets.frontend import phone_to_int, tone_to_int, phonetone_to_int
from recipes.text2semantic.scripts.infer_utils import (
    load_torch_script,
    setup_seed,
    spectrogram_torch,
    trim_silence,
    trim_prompt_silence,
)
from scipy.io.wavfile import write
from recipes.text2semantic.utils.remote_io import load_json
from recipes.text2semantic.datasets.sami_tacolabel import generate_tacolabels_from_textstr_punc

logger = logging.getLogger(__name__)


class BigTTSWVAEInferLangSpk(BigTTSWVAEInfer):
    def __init__(
            self,
            ar_model_name,
            ckpt_path,
            wvae_encoder,
            wvae_decoder,
            output_dir,
            phone_tokens_num=7370,
            text2id_path="recipes/valle/datasets/dict/metaid_to_textid.json",
            module_cache=".module_cache",
            seed=1996,
            tacolab_version='oldv1',  # oldv1, newv3, newv3_punc
            text2id_version='v1',
            prompt_tacolab_dir="",
            infer_tacolab_dir="",
            use_sy=False,
            use_lang_id=False,
            lang_tokens_num=0,
            lang2id=None,
            use_bpe=False,
            bpe_tokens_num=0,
            bpe_dir='',
            tokenizer_type='',
            max_length=4096,
            use_spk_id=False,
            use_prompt=False,
            spk_tokens_num=8192,
            spk2id='',
            infer_spk_name='',
            get_lang_by_tacolab=False,
            save_tacolab=False,
            tag_id=0,
            use_offline_tacolab=False,
            input_type='2dim',
            **kwargs
    ):
        super().__init__(
            ar_model_name=ar_model_name,
            ckpt_path=ckpt_path,
            wvae_encoder=wvae_encoder,
            wvae_decoder=wvae_decoder,
            output_dir=output_dir,
            phone_tokens_num=phone_tokens_num,
            speaker_tokens_num=spk_tokens_num,
            text2id_path=text2id_path,
            module_cache=module_cache,
            seed=seed,
            tacolab_version=tacolab_version,  # oldv1, newv3, newv3_punc
            text2id_version=text2id_version,
            prompt_tacolab_dir=prompt_tacolab_dir,
            infer_tacolab_dir=infer_tacolab_dir,
            use_sy=use_sy,
            use_lang_id=use_lang_id,
            lang_tokens_num=lang_tokens_num,
            lang2id=lang2id,
            use_spk_id=use_spk_id,
            spk2id=spk2id,
            spkname=infer_spk_name,
            use_bpe=use_bpe,
            bpe_tokens_num=bpe_tokens_num,
            bpe_dir=bpe_dir,
            tokenizer_type=tokenizer_type,
            max_length=max_length,
            **kwargs)

        self.phone_to_int = phone_to_int
        self.tone_to_int = tone_to_int
        self.infer_spk_name = infer_spk_name
        self.get_lang_by_tacolab = get_lang_by_tacolab
        self.save_tacolab = save_tacolab
        self.tag_id = tag_id
        self.use_prompt = use_prompt
        self.use_offline_tacolab = use_offline_tacolab
        if self.use_offline_tacolab:
            assert self.infer_tacolab_dir != "", (self.infer_tacolab_dir)
            if not self.use_spk_id:
                assert self.prompt_tacolab_dir != "", (self.prompt_tacolab_dir)
        self.phonetone_to_int = phonetone_to_int
        self.input_type = input_type
        logging.info("========== init success ==========")

    def get_lang(self, tacolab):
        if len(tacolab[0].split('\t')) != 5:
            if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[
                0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                tacolab = tacolab[1:]
        prefix_phn_list = [x.split('\t')[0][:2] for x in tacolab]
        if 'C0' in prefix_phn_list:
            if 'E0' in prefix_phn_list:
                lang = 'zh_en'
            else:
                lang = 'zh'
        else:
            lang = 'en'
        return lang

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        setup_seed(self.hparams.seed)
        sample = self.encode(batch)
        if sample is None:
            print("encode failed", batch)
            return

        z_outputs, _ = self.ar_model.predict(sample, None)
        generated_wav = self._decode(z_outputs)

        if self.infer_mode == 'offline':
            utt_ids = sample['uttid']
            output_dir = f"{self.hparams.output_dir}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{utt_ids[0]}.wav"
            #        if os.path.exists(output_path):
            #            return
            generated_wav *= (32767) / max(0.01, max(torch.abs(generated_wav)))
            write(
                output_path,
                24000,
                generated_wav.cpu().numpy().astype(np.int16),
            )
        else:
            generated_wav *= 0.95 / max(0.01, max(torch.abs(generated_wav)))
            return generated_wav.cpu().numpy()

    def encode(self, sample):
        device = self.get_device()
        data_dict = dict()

        if len(sample) == 5:
            uttid, prompt_text, prompt_wav_path, infer_text, spk = sample
        elif len(sample) == 4:
            uttid, prompt_text, prompt_wav_path, infer_text = sample
            spk = None
        elif len(sample) == 2:
            uttid, infer_text = sample
            prompt_text, prompt_wav_path, spk = None, None, None
        else:
            raise NotImplementedError(sample)

        if not self.use_prompt:
            bn = np.zeros([0, 32])
            bn = torch.from_numpy(bn)
        else:
            assert prompt_text is not None
            assert prompt_wav_path is not None
            sr, wav = read(prompt_wav_path)
            if len(wav.shape) == 2 and wav.shape[-1] == 2:
                wav = wav[:, 0]
            wav = wav / 32767.0
            wav *= 1.0 / max(0.01, np.max(np.abs(wav)))
            if sr != 24_000:
                wav = librosa.core.resample(wav, sr, 24_000)
            # wav = wav * 0.99 / max(0.01, np.max(np.abs(wav)))
            # wav = trim_prompt_silence(wav)
            wav = trim_silence(wav)
            wav = torch.from_numpy(wav).float()
            wav = wav.to(device)
            wav = torch.stack([wav]).unsqueeze(1).float()
            wav = F.pad(wav, (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1), 0, 0, 0, 0), value=0.)
            # wav = torch.stack([wav]).unsqueeze(1)
            spec = spectrogram_torch(wav.squeeze(1), 2048, 24000, 300, 1200)
            # print("wav: ", wav.shape, wav)
            # print("spec: ", spec.shape, spec)
            # exit()
            _, m, logs = self.wvae_encoder(wav, spec)
            m = m.transpose(2, 1)
            logs = logs.transpose(2, 1)
            bn = torch.cat([m, logs], -1)  # (1, t, 64)

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        # get text_id
        if not self.use_prompt:
            if self.use_offline_tacolab:
                infer_utt = uttid
                infer_tacolab_path = os.path.join(self.infer_tacolab_dir, infer_utt + '.lab')
                if not os.path.exists(infer_tacolab_path):
                    print("infer_tacolab_path not exists, skip", infer_tacolab_path)
                    return None
                with open(infer_tacolab_path, 'r', encoding="utf-8") as f:
                    infer_tacolab = f.read()
            else:
                infer_tacolab = self.generate_tacolabels_engine(infer_text)
                if infer_tacolab is None:
                    return None
                infer_tacolab = infer_tacolab.decode()

            if self.save_tacolab:
                save_infer_tacolab_dir = os.path.join(self.hparams.output_dir, '../infer_tacolab')
                os.makedirs(save_infer_tacolab_dir, exist_ok=True)
                infer_tacolab_path = os.path.join(save_infer_tacolab_dir, uttid + '.lab')
                with open(infer_tacolab_path, 'w', encoding='utf-8') as f_w:
                    f_w.write(infer_tacolab)

            tacolab = infer_tacolab
            tacolab_list = list(filter(lambda x: x != "", tacolab.split('\n')))
            text_id_phones_tones = self.convert_tacolab_to_text_id(tacolab_list)

            if text_id_phones_tones is None:
                logger.warning(f"{uttid} convert_tacolab_to_text_id failed ...")
                return None
            else:
                if self.input_type == '2dim':
                    text_id, phones, tones = text_id_phones_tones
                elif self.input_type == '1dim':
                    text_id, phones, tones, phonetone_ids, phonetones = text_id_phones_tones
                    # print("phonetone_ids: ", phonetone_ids)
                else:
                    raise NotImplementedError
        else:
            if self.use_offline_tacolab:
                prompt_utt = prompt_wav_path.split('/')[-1][:-4]
                infer_utt = uttid
                prompt_tacolab_path = os.path.join(self.prompt_tacolab_dir, prompt_utt + '.lab')
                infer_tacolab_path = os.path.join(self.infer_tacolab_dir, infer_utt + '.lab')
                if not os.path.exists(prompt_tacolab_path) or not os.path.exists(infer_tacolab_path):
                    print("prompt_tacolab_path or infer_tacolab_path not exists, skip", prompt_tacolab_path,
                          infer_tacolab_path)
                    return None
                with open(prompt_tacolab_path, 'r', encoding="utf-8") as f:
                    prompt_tacolab = f.read()
                with open(infer_tacolab_path, 'r', encoding="utf-8") as f:
                    infer_tacolab = f.read()
            else:
                prompt_tacolab = self.generate_tacolabels_engine(prompt_text)
                infer_tacolab = self.generate_tacolabels_engine(infer_text)

                if prompt_tacolab is None or infer_tacolab is None:
                    return None
                prompt_tacolab = prompt_tacolab.decode()
                infer_tacolab = infer_tacolab.decode()

            if self.save_tacolab:
                save_prompt_tacolab_dir = os.path.join(self.hparams.output_dir, '../prompt_tacolab')
                os.makedirs(save_prompt_tacolab_dir, exist_ok=True)
                prompt_tacolab_path = os.path.join(save_prompt_tacolab_dir, uttid + '.lab')
                with open(prompt_tacolab_path, 'w', encoding='utf-8') as f_w:
                    f_w.write(prompt_tacolab)

                save_infer_tacolab_dir = os.path.join(self.hparams.output_dir, '../infer_tacolab')
                os.makedirs(save_infer_tacolab_dir, exist_ok=True)
                infer_tacolab_path = os.path.join(save_infer_tacolab_dir, uttid + '.lab')
                with open(infer_tacolab_path, 'w', encoding='utf-8') as f_w:
                    f_w.write(infer_tacolab)

            # tacolab = prompt_tacolab.strip('\n') + '\n' + infer_tacolab
            prompt_tacolab_list = list(filter(lambda x: x != "", prompt_tacolab.split('\n')))
            infer_tacolab_list = list(filter(lambda x: x != "", infer_tacolab.split('\n')))
            prompt_text_id_phones_tones = self.convert_tacolab_to_text_id(prompt_tacolab_list)
            infer_text_id_phones_tones = self.convert_tacolab_to_text_id(infer_tacolab_list)

            if prompt_text_id_phones_tones is None or infer_text_id_phones_tones is None:
                logger.warning(f"{uttid} convert_tacolab_to_text_id failed ...")
                return None
            else:
                if self.input_type == '2dim':
                    prompt_text_id, prompt_phones, prompt_tones = prompt_text_id_phones_tones
                    infer_text_id, infer_phones, infer_tones = infer_text_id_phones_tones
                    text_id = np.concatenate([prompt_text_id, infer_text_id], axis=-1)
                    phones = prompt_phones + infer_phones
                    tones = prompt_tones + infer_tones
                elif self.input_type == '1dim':
                    prompt_text_id, prompt_phones, prompt_tones, prompt_phonetone_ids, prompt_phonetones = prompt_text_id_phones_tones
                    infer_text_id, infer_phones, infer_tones, infer_phonetone_ids, infer_phonetones = infer_text_id_phones_tones
                    text_id = np.concatenate([prompt_text_id, infer_text_id], axis=-1)
                    phonetone_ids = np.concatenate([prompt_phonetone_ids, infer_phonetone_ids], axis=-1)
                    phones = prompt_phones + infer_phones
                    tones = prompt_tones + infer_tones
                    phonetones = prompt_phonetones + infer_phonetones
                    # print("phonetone_ids: ", phonetone_ids)
                else:
                    raise NotImplementedError

        # pad eos
        text_id = np.concatenate([text_id, np.ones([text_id.shape[0], 1])], axis=-1)

        data_dict["phonetone"] = None
        if self.input_type == '1dim':
            data_dict["phonetone"] = np.concatenate([phonetone_ids, np.ones([1])], axis=0)
            data_dict["phonetone"] = torch.from_numpy(data_dict["phonetone"]).long().unsqueeze(0).to(device)

        if self.use_lang_id:
            if not self.use_prompt:
                prompt_lang_id = 1  # dummy
                lang_seq = np.asarray([prompt_lang_id] * bn.shape[0])

                infer_lang_key = self.get_lang_by_text(infer_text)
                if infer_lang_key == None:
                    logger.info(f"{infer_text}: Wrong lang_key")
                    return None
                infer_lang_id = self.lang2id[infer_lang_key]
                infer_lang_id += 1
            else:
                if self.get_lang_by_tacolab:
                    prompt_lang_key = self.get_lang(prompt_tacolab.split('\n'))
                else:
                    prompt_lang_key = self.get_lang_by_text(prompt_text)
                if prompt_lang_key == None:
                    logger.info(f"{prompt_text}: Wrong lang_key")
                    return None
                prompt_lang_id = self.lang2id[prompt_lang_key]

                if self.get_lang_by_tacolab:
                    infer_lang_key = self.get_lang(infer_tacolab.split('\n'))
                else:
                    infer_lang_key = self.get_lang_by_text(infer_text)

                if infer_lang_key == None:
                    logger.info(f"{infer_text}: Wrong lang_key")
                    return None
                infer_lang_id = self.lang2id[infer_lang_key]

                prompt_lang_id += 1
                infer_lang_id += 1
                lang_seq = np.asarray([prompt_lang_id] * bn.shape[0])

        if self.use_spk_id:
            prompt_spk_id = 1  # dummy
            spk_seq = np.asarray([prompt_spk_id] * bn.shape[0])

            infer_spk_key = self.infer_spk_name if self.infer_spk_name is not None else spk
            if self.spk2id.get(infer_spk_key) is None:
                logger.info(f"{infer_spk_key}: Wrong spk_key")
                return None
            infer_spk_id = self.spk2id[infer_spk_key]
            infer_spk_id += 1

        data_dict['bn'] = bn.unsqueeze(0).to(device).to(device)
        data_dict['text_lens'] = torch.tensor([text_id.shape[1]]).long().to(device)
        data_dict['bn_lens'] = torch.tensor([bn.shape[0]]).long().to(device)
        data_dict["phone"] = torch.from_numpy(text_id[0, :]).long().unsqueeze(0).to(device)
        data_dict["tone"] = torch.from_numpy(text_id[1, :]).long().unsqueeze(0).to(device)

        data_dict['lang_seq'] = None
        data_dict['infer_lang_id'] = None
        if self.use_lang_id:
            data_dict['lang_seq'] = torch.tensor(lang_seq).long().to(device).unsqueeze(0)
            data_dict['infer_lang_id'] = infer_lang_id
        data_dict['uttid'] = [uttid]

        data_dict['spk_seq'] = None
        data_dict['infer_spk_id'] = None
        if self.use_spk_id:
            data_dict['spk_seq'] = torch.tensor(spk_seq).long().to(device).unsqueeze(0)
            data_dict['infer_spk_id'] = infer_spk_id

        # bpe_id
        if self.use_bpe:
            data_dict['bpe_seq'] = torch.from_numpy(
                np.asarray(
                    self.bpe_tokenizer(
                        infer_text,
                        truncation=True,
                        max_length=self.max_length,
                    ).input_ids)).unsqueeze(0).to(device)
            data_dict['bpe_lens'] = torch.tensor([data_dict['bpe_seq'].shape[1]]).long().to(device)
        else:
            data_dict['bpe_seq'] = None

        data_dict['tag_id'] = torch.from_numpy(
            np.asarray([int(self.tag_id)])).long().to(device)

        return data_dict

    def generate_tacolabels_engine(self, text_str):
        lang_key = self.get_lang_by_text(text_str)
        if lang_key == None:
            logger.info(f"{text_str}: Wrong lang_key")
            return None
        tacolab = None
        if lang_key in ['zh', 'zh_en']:
            tacolab = generate_tacolabels_from_textstr_punc(text_str, "Chinese_v3_punc")
        elif lang_key in ['en']:
            tacolab = generate_tacolabels_from_textstr_punc(text_str, "English_v3_punc")

        return tacolab

    def convert_v3_to_v1(self, tacolab):
        tacolab_v1 = []
        # en, zh
        if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[
            0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
            tacolab = tacolab[1:]
        for x in tacolab:
            x_split = x.split('\t')
            if len(x_split) == 7:
                phone, tone, ws, pw, stype, word, _ = x_split
            elif len(x_split) == 6:
                phone, tone, ws, pw, stype, word = x_split
            else:
                print("Wrong tacolab", x_split)
                return None
            tacolab_v1.append('\t'.join([phone, tone, '0.0 0.0 0.0 1.0', ws, pw]))
        return tacolab_v1

    def convert_tacolab_to_text_id(self, tacolab):
        try:
            lang = self.get_lang(tacolab)
            phone_ids = []
            tone_ids = []
            phonetone_ids = []
            phones = []
            tones = []
            phonetones = []
            if lang == 'zh':
                assert len(tacolab[0].split('\t')) == 7, (len(tacolab[0].split('\t')), tacolab[0])
                if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                    tacolab = tacolab[1:]
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, ws, pw, stype, word, unit = x_split
                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    assert phone + '_' + tone in self.phonetone_to_int, f"{phone + '_' + tone} not in phonetone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phonetone_ids.append(self.phonetone_to_int[phone + '_' + tone])
                    phones.append(phone)
                    tones.append(tone)
                    phonetones.append(phone + '_' + tone)
                    if phone[:2] == "C0":
                        if unit in ['S', 'E']:
                            phone_ids.append(self.phone_to_int["syl_sep"])
                            tone_ids.append(self.tone_to_int["syl_sep"])
                            phonetone_ids.append(self.phonetone_to_int["syl_sep_syl_sep"])
                            phones.append("syl_sep")
                            tones.append("syl_sep")
                            phonetones.append("syl_sep_syl_sep")
                            if ws in ["S", "E"]:
                                phone_ids.append(self.phone_to_int["zh_word_sep"])
                                tone_ids.append(self.tone_to_int["zh_word_sep"])
                                phonetone_ids.append(self.phonetone_to_int["zh_word_sep_zh_word_sep"])
                                phones.append("zh_word_sep")
                                tones.append("zh_word_sep")
                                phonetones.append("zh_word_sep_zh_word_sep")
                    elif phone[:2] == "E0":
                        if pw != "0":
                            phone_ids.append(self.phone_to_int["en_word_sep"])
                            tone_ids.append(self.tone_to_int["en_word_sep"])
                            phonetone_ids.append(self.phonetone_to_int["en_word_sep_en_word_sep"])
                            phones.append("en_word_sep")
                            tones.append("en_word_sep")
                            phonetones.append("en_word_sep_en_word_sep")
            elif lang == 'zh_en':
                assert len(tacolab[0].split('\t')) == 7 or len(tacolab[0].split('\t')) == 6, (
                len(tacolab[0].split('\t')), tacolab[0])
                if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit' or tacolab[
                    0] == 'phn\ttone\tws\tpwpp\tsentype\tword':
                    tacolab = tacolab[1:]
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    if len(x_split) == 7:
                        phone, tone, ws, pw, stype, word, unit = x_split
                    elif len(x_split) == 6:
                        phone, tone, ws, pw, stype, word = x_split
                    else:
                        print("Wrong tacolab", x_split)
                        return None

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    assert phone + '_' + tone in self.phonetone_to_int, f"{phone + '_' + tone} not in phonetone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phonetone_ids.append(self.phonetone_to_int[phone + '_' + tone])
                    phones.append(phone)
                    tones.append(tone)
                    phonetones.append(phone + '_' + tone)
                    if phone[:2] == "C0":
                        if unit in ['S', 'E']:
                            phone_ids.append(self.phone_to_int["syl_sep"])
                            tone_ids.append(self.tone_to_int["syl_sep"])
                            phonetone_ids.append(self.phonetone_to_int["syl_sep_syl_sep"])
                            phones.append("syl_sep")
                            tones.append("syl_sep")
                            phonetones.append("syl_sep_syl_sep")
                            if ws in ["S", "E"]:
                                phone_ids.append(self.phone_to_int["zh_word_sep"])
                                tone_ids.append(self.tone_to_int["zh_word_sep"])
                                phonetone_ids.append(self.phonetone_to_int["zh_word_sep_zh_word_sep"])
                                phones.append("zh_word_sep")
                                tones.append("zh_word_sep")
                                phonetones.append("zh_word_sep_zh_word_sep")
                    elif phone[:2] == "E0":
                        if pw != "0":
                            phone_ids.append(self.phone_to_int["en_word_sep"])
                            tone_ids.append(self.tone_to_int["en_word_sep"])
                            phonetone_ids.append(self.phonetone_to_int["en_word_sep_en_word_sep"])
                            phones.append("en_word_sep")
                            tones.append("en_word_sep")
                            phonetones.append("en_word_sep_en_word_sep")
            elif lang == 'en':
                if len(tacolab[0].split('\t')) != 5:
                    tacolab = self.convert_v3_to_v1(tacolab)
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, _, ws, pw = x.split('\t')
                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    assert phone + '_' + tone in self.phonetone_to_int, f"{phone + '_' + tone} not in phonetone set"
                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phonetone_ids.append(self.phonetone_to_int[phone + '_' + tone])
                    phones.append(phone)
                    tones.append(tone)
                    phonetones.append(phone + '_' + tone)
                    if pw != "0":
                        phone_ids.append(self.phone_to_int["en_word_sep"])
                        tone_ids.append(self.tone_to_int["en_word_sep"])
                        phonetone_ids.append(self.phonetone_to_int["en_word_sep_en_word_sep"])
                        phones.append("en_word_sep")
                        tones.append("en_word_sep")
                        phonetones.append("en_word_sep_en_word_sep")
            phone_ids, tone_ids = np.array(phone_ids), np.array(tone_ids)
            phonetone_ids = np.asarray(phonetone_ids)
            if self.input_type == '2dim':
                return np.stack([phone_ids, tone_ids]), phones, tones
            elif self.input_type == '1dim':
                return np.stack([phone_ids, tone_ids]), phones, tones, phonetone_ids, phonetones
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            return None
