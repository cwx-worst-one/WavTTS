import json
import logging
import pickle
import sys
import os 

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import T5Tokenizer, AutoTokenizer

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.parquet import ParquetDataset

from recipes.text2semantic.utils.remote_io import load_json
from transformers import LlamaTokenizer
from zhon.hanzi import punctuation
import string
punctuation_all = punctuation + string.punctuation
import librosa
from scipy.io import wavfile

from recipes.text2semantic.datasets.frontend2 import phone_to_int, tone_to_int, phonetone_to_int, sil_punc_symbols
from collections import defaultdict
import random
import re

logger = logging.getLogger(__name__)

punctuation_for_split = ["sp", "pau", "，", "：", "；", "～", "､", "、", "〜", "…", "﹔", "！", "？", "｡", "。", "!", ",", ".", ":", ";", "?", "~", "......", "...", "……", "--", "——"]

def trim_silence(wav):
    """
    Trim leading and trailing silence
    """
    # These params are separate and tunable per dataset.
    
    origin_wav_len = len(wav)
    wav = np.pad(wav, (5400, 5400))

    unused_trimed, index = librosa.effects.trim(
        wav, top_db=30, frame_length=512, hop_length=128
    )
    # num_sil_samples = int(8 * 300)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - 3200, 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + 5400, len(wav))

    trimmed = wav[start_idx:stop_idx]

    head_trim_nums = 5400 - start_idx
    tail_trim_nums = stop_idx - 5400 - origin_wav_len

    return trimmed, head_trim_nums, tail_trim_nums

def wav_save_orgamp(path, sr, wav):
    wav = wav * 32767
    wavfile.write(path, sr, wav.astype(np.int16))

def modify_alignment(alignment_ori, head_trim_dur, tail_trim_dur):
    phones = [x.split('\t')[0] for x in alignment_ori.strip().split('\n')]
    time_stamp = [float(x.split('\t')[1]) for x in alignment_ori.strip().split('\n')]
    if head_trim_dur > 0:
        time_stamp += head_trim_dur
    else:
        if time_stamp[0] < abs(head_trim_dur):
            logging.raiseExceptions(f"The dur of first phone is less than head_trim_dur, {time_stamp[0]}, {head_trim_dur}")
        else:
            time_stamp += head_trim_dur

    if tail_trim_dur > 0:
        time_stamp[-1] += tail_trim_dur
    else:
        if time_stamp[-1] < abs(tail_trim_dur):
            logging.raiseException(f"The dur of last phone is less than tail_trim_dur, {time_stamp[-1]}, {tail_trim_dur}")
        else:
            time_stamp[-1] += head_trim_dur
    
    alignment = ""
    for i in range(len(phones)):
        alignment += phones[i] + '\t' + str(time_stamp[i]) + '\n'
    return alignment


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ContinuousTTSLangSpkSerAliDataset(IterableDataset):
    def __init__(self,
        data_id,
        drop_last=False,
        batcher_config=None,
        use_lang_id=False,
        use_spk_id=False,
        spk2id=None,
        lang2id=None,
        use_code_switch_data=True,
        get_lang_by_tacolab=False,
        en_foreigner_list=None,
        bpe_dir=None, 
        max_length=4096,
        use_bpe=False,
        use_extra_tag=False,
        spk2tag=None,
        tokenizer_type="llama",
        use_foreigner_data=True,
        use_lang_cfg=False,
        input_type='2dim',
        spk_cfg_rate=0.0,
        lang_cfg_rate=0.0,
        use_sp=True,
        refenc_cfg_rate=0.0,
        split_cfg_rate=0.0,
        ):

        self.dataset = (
            ParquetDataset(
                data_id=int(data_id),
                resampled=True,
            )
            .shuffle(2048)
            .map(self.process_meta)
            .map(self.get_text_wavid)
        )

        self.max_length = max_length
        self.use_bpe = use_bpe
        self.use_extra_tag = use_extra_tag

        logger.info(f"dataset/use_extra_tag: {use_extra_tag}")

        self.drop_last = drop_last
        self.batcher = BucketBatcher(**batcher_config)

        # lang
        self.use_lang_id = use_lang_id
        if self.use_lang_id:
            self.lang2id = load_json(lang2id)
            logger.info(f"Loaded lang2id from {lang2id}")
        else:
            self.lang2id = None
        print("self.lang2id: ", self.lang2id)

        # spk
        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            self.spk2id = load_json(spk2id)
            logger.info(f"Loaded spk2id from {spk2id}")
        else:
            self.spk2id = None
        # print("self.spk2id: ", self.spk2id)

        # phone/tone
        self.phone_to_int = phone_to_int
        self.tone_to_int = tone_to_int
        self.phonetone_to_int = phonetone_to_int
            
        logger.info(f"{self.phone_to_int=}, {self.tone_to_int=}")
        logger.info(f"{self.phonetone_to_int=}")

        # for cross-lingual
        self.use_code_switch_data = use_code_switch_data
        self.get_lang_by_tacolab = get_lang_by_tacolab

        self.en_foreigner_list = None
        if en_foreigner_list:
            self.en_foreigner_list = [x.strip() for x in open(en_foreigner_list).readlines()]
        self.use_foreigner_data = use_foreigner_data
        if not self.use_foreigner_data:
            assert self.en_foreigner_list != None
        self.use_lang_cfg = use_lang_cfg

        if self.use_extra_tag:
            self.tag_dict = load_json(spk2tag)

        if self.use_bpe:
            logger.info(f'##### Using BPE #####')
            if tokenizer_type == "flan-T5-large":
                self.bpe_tokenizer = T5Tokenizer.from_pretrained(bpe_dir)
            elif tokenizer_type == "byte-T5-base":
                self.bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_dir)
            elif tokenizer_type == "llama":
                self.bpe_tokenizer = LlamaTokenizer.from_pretrained(bpe_dir)
            else:
                raise NotImplementedError
        else:
            self.bpe_tokenizer = None
        
        self.input_type = input_type

        # cfg
        self.spk_cfg_rate = spk_cfg_rate
        assert 0 <= self.spk_cfg_rate <= 1
        self.lang_cfg_rate = lang_cfg_rate
        assert 0 <= self.lang_cfg_rate <= 1

        # use_sp
        self.use_sp = use_sp

        self.refenc_cfg_rate = refenc_cfg_rate
        assert 0 <= self.refenc_cfg_rate <= 1

        # split for streaming
        self.split_cfg_rate = split_cfg_rate
        assert 0 <= self.split_cfg_rate <= 1

    def process_meta(self, sample):
        meta_obj = json.loads(sample["meta"])
        while not isinstance(meta_obj, dict):
            meta_obj = json.loads(meta_obj)
        item = {}
        item["labels"] = str(meta_obj.get("labels", ""))
        item["speaker_name"] = str(meta_obj.get("speaker_id", ""))
        item["snr"] = str(meta_obj.get("snr", "10.0"))
        item["mos"] = str(meta_obj.get("mos", "5.0"))
        item["rms_stats_rms_max"] = "-1"
        item["speaker_similarity_min"] = "1.0"
        item["alignment"] = str(meta_obj.get("alignment", ""))
        sample.update(item)
        return sample

    def get_text_wavid(self, sample):

        bn = pickle.loads(sample["bns"])
        text = sample["text"]
        labels = sample["labels"]
        utt_id = sample["__key__"]
        url = sample["__data_url__"]
        wav = np.frombuffer(sample["wav"][44:], dtype=np.int16) / 32768.0
        wav = wav.astype(np.float32)

        data_dict = dict()

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split('\n')))

        try:
            labels = self.remove_head(labels)
        except Exception:
            logger.warning(f"wrong labels, skip: \n{labels}")
            return None

        match_the_need_of_split = True
        alignment_ori = sample["alignment"]
        if alignment_ori == "":
            logger.warning(f"{utt_id}, missing alignment, can not be splitted.")
            match_the_need_of_split = False
        else:
            wav_trim = torch.FloatTensor(wav)
            scale = max(0.01, torch.max(torch.abs(wav_trim)))
            wav_trim = wav_trim / scale * 1.0
            trimmed, head_trim_nums, tail_trim_nums = trim_silence(wav_trim)
            try:
                alignment = modify_alignment(alignment_ori, head_trim_nums / 24000, tail_trim_nums / 24000)
            except Exception:
                logger.warning(f"{utt_id}, {url}, head or tail dur not match the alignment, can not be splitted.")
                # wav_save_orgamp(f"{utt_id}.wav", 24000, wav)
                # wav_save_orgamp(f"{utt_id}_trimmed.wav", 24000, trimmed)
                # print(head_trim_nums / 24000, tail_trim_nums / 24000)
                # print(f"alignment_ori: {alignment_ori}")
                # print("alignment_ori: ", len(alignment_ori.strip().split('\n')), alignment_ori)
                # # print("alignment: ", len(alignment.strip().split('\n')), alignment)
                # print("labels: ", len(labels), labels)
                match_the_need_of_split = False

            if match_the_need_of_split:
                alignment = alignment.strip().split('\n')
                frame_num_alignment = int(float(alignment[-1].split('\t')[1]) / 0.025) + 1
                if abs(frame_num_alignment - bn.shape[0]) > 1:
                    logger.warning(f"{utt_id}, {url}, frame num mismatch between alignment and bn, can not be splitted. {frame_num_alignment}, {bn.shape[0]}")
                    match_the_need_of_split = False

                punc_split_indexs = []
                bn_split_indexs = []
                for i in range(len(alignment)):
                    phone, time_stamp = alignment[i].split('\t')
                    time_stamp = float(time_stamp)
                    if phone in punctuation_for_split and i != len(alignment) - 1:
                        if not (self.alignment_all_punc(alignment[:i+1]) or self.alignment_all_punc(alignment[i+1:])):
                            punc_split_indexs.append(i)
                            bn_split_indexs.append(round(time_stamp / 0.025))

        text_ = None
        if self.split_cfg_rate > 0:
            if np.random.rand() < self.split_cfg_rate and match_the_need_of_split and len(bn_split_indexs) != 0:
                split_pos = random.randint(0, len(punc_split_indexs) - 1)
                punc_split_index = punc_split_indexs[split_pos]
                bn_split_index = bn_split_indexs[split_pos]

                # split bn/tacolab/alignment/text
                prompt_bn = bn[:bn_split_index + 1, :]
                bn = bn[bn_split_index + 1:, :]

                prompt_labels = labels[:punc_split_index + 1]
                labels = labels[punc_split_index + 1:]

                prompt_alignment = alignment[:punc_split_index + 1]
                alignment = alignment[punc_split_index + 1:]

                prompt_text, text_ = self.split_text_by_labels(text, prompt_labels, labels)
                if prompt_text is None:
                    logger.warning(f"{utt_id}, {url}, split_text_by_labels fail, can not be splitted.")
                    prompt_bn = np.zeros([0, 64])
            else:
                prompt_bn = np.zeros([0, 64])

        if text_ is not None:
            text = text_

        if not self.use_sp:
            labels = self.remove_replace_sp(labels, utt_id)
            if labels is None:
                return None

        # text_id, [text_len]
        text_id_phones_tones = self.convert_tacolab_to_text_id(labels, url)
        if text_id_phones_tones is None:
            logger.warning(f"{utt_id} convert_tacolab_to_text_id failed, {labels}")
            return None
        else:
            if self.input_type == '2dim':
                text_id, phones, tones = text_id_phones_tones
            elif self.input_type == '1dim':
                text_id, phones, tones, phonetone_ids, phonetones = text_id_phones_tones
                # print("phonetone_ids: ", phonetone_ids)
            else:
                raise NotImplementedError
    
        # # prompt_text_id, [prompt_text_len]
        # if prompt_labels != []:
        #     prompt_text_id_phones_tones = self.convert_tacolab_to_text_id(prompt_labels, url)
        #     if prompt_text_id_phones_tones is None:
        #         logger.warning(f"(prompt_labels) {utt_id} convert_tacolab_to_text_id failed, {prompt_labels}")
        #         return None
        #     else:
        #         if self.input_type == '2dim':
        #             prompt_text_id, prompt_phones, prompt_tones = prompt_text_id_phones_tones
        #         elif self.input_type == '1dim':
        #             prompt_text_id, prompt_phones, prompt_tones, prompt_phonetone_ids, prompt_phonetones = prompt_text_id_phones_tones
        #         else:
        #             raise NotImplementedError
        # else:
        #     prompt_text_id = None

        if self.use_extra_tag:
            dataset_name = sample.get("dataset_name")
            speaker_name = sample.get("speaker_name")
            if not dataset_name or not speaker_name:
                logger.warning(f"{utt_id}: No speaker name or No dataset name, use 'default' as spk_key")
                spk_key = 'default'
            else:
                spk_key = '/'.join([dataset_name, speaker_name])
                if spk_key not in self.tag_dict:
                    spk_key = dataset_name
                    if spk_key not in self.tag_dict:
                        logger.warning(f"{utt_id} {dataset_name} {speaker_name}: dataset_name/speaker_name and dataset_name not in spk2tag, use 'default' as spk_key")
                        spk_key = 'default'
            tag_id = self.tag_dict[spk_key]
        else:
            tag_id = None

        # if prompt_labels != []:
        #     prompt_text_id = np.concatenate([prompt_text_id, np.ones([text_id.shape[0], 1]) * 3], axis=-1)
        #     data_dict["prompt_phone"] = prompt_text_id[0, :]
        #     data_dict["prompt_tone"] = prompt_text_id[1, :]

        text_id = np.concatenate([np.ones([text_id.shape[0], 1]) * 2, text_id, np.ones([text_id.shape[0], 1])], axis=-1)
        # else:
        #     text_id = np.concatenate([text_id, np.ones([text_id.shape[0], 1])], axis=-1)
        #     data_dict["prompt_phone"] = None
        #     data_dict["prompt_tone"] = None

        data_dict["phone"] = text_id[0, :]
        data_dict["tone"] = text_id[1, :]

        data_dict["phonetone"] = None
        if self.input_type == '1dim':
            data_dict["phonetone"] = np.concatenate([phonetone_ids, np.ones([1])], axis=0)

        # lang seq
        lang_seq = None        
        if self.use_lang_id:
            if self.get_lang_by_tacolab:
                lang_key = self.get_lang(labels)
            else:
                lang_key = self.get_lang_by_text(text.encode("utf-8"))
            if self.en_foreigner_list:
                speaker_name = sample.get("speaker_name")
                if speaker_name:
                    if speaker_name.decode() in self.en_foreigner_list and lang_key == 'en':
                        lang_key = 'en_foreigner'
                        if not self.use_foreigner_data:
                            print(f"{utt_id}, {text}: Wrong data, Foreigner data.")
                            return None
            if lang_key == None or lang_key not in self.lang2id.keys():
                print(f"{utt_id}, {text}, {lang_key}: Wrong lang_key")
                return None
            lang_id = self.lang2id[lang_key]

            if self.lang_cfg_rate > 0:
                if np.random.rand() < self.lang_cfg_rate:
                    lang_id = self.lang2id['default']

            lang_id += 1
            lang_seq = np.asarray([lang_id] * bn.shape[0])

        # spk_seq
        spk_seq = None        
        if self.use_spk_id:
            dataset_name = sample.get("dataset_name")
            speaker_name = sample.get("speaker_name")
            if not dataset_name or not speaker_name:
                logger.warning(f"{utt_id}: No speaker name or No dataset name, use 'default' as spk_key")
                spk_key = 'default'
            else:
                spk_key = '/'.join([dataset_name, speaker_name])
                if spk_key not in self.spk2id:
                    spk_key = dataset_name
                    if spk_key not in self.spk2id:
                        logger.warning(f"{utt_id} {dataset_name} {speaker_name}: dataset_name/speaker_name and dataset_name not in spk2id, use 'default' as spk_key")
                        spk_key = 'default'
                if spk_key != 'default':
                    if self.spk_cfg_rate > 0:
                         if np.random.rand() < self.spk_cfg_rate:
                            spk_key = 'default'
            spk_id = self.spk2id[spk_key]
            spk_id += 1
            spk_seq = np.asarray([spk_id] * bn.shape[0])

        spk_embd_mask = 1
        if self.refenc_cfg_rate > 0:
            if spk_key == 'default':
                spk_embd_mask = 0
            else:
                if np.random.rand() < self.refenc_cfg_rate:
                    spk_embd_mask = 0

        # stop_token
        stop_token = np.zeros([prompt_bn.shape[0] + text_id.shape[1] + bn.shape[0]])
        stop_token[-1] = 1
    
        # bpe_id
        if self.use_bpe:
            bpe_seq = np.asarray(self.bpe_tokenizer(
                text, truncation=True, max_length=self.max_length,
            ).input_ids)
        else:
            bpe_seq = None

        if bn.shape[0] <= 1:
            logger.warning(f"{utt_id} {url}: wrong bn")
            return None

        data_dict.update({
            "stop_token": stop_token,
            "bn": bn,
            "utt_id": utt_id,
            "lang_seq": lang_seq,
            "spk_seq": spk_seq,
            "bpe_seq": bpe_seq,
            "tag_id": tag_id,
            "wav":  wav,
            "spk_embd_mask": spk_embd_mask,
            "prompt_bn": prompt_bn,
            })
        return data_dict

    def __iter__(self):
        for item in self.dataset:
            batch = self.batcher.collate_batch(item)
            if batch:
                yield batch
        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield batch

    def convert_v3_to_v1(self, tacolab, url):
        tacolab_v1 = []
        # en, zh
        if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
            tacolab = tacolab[1:]
        for x in tacolab:
            x_split = x.split('\t')
            if len(x_split) == 7:
                phone, tone, ws, pw, stype, word, _ = x_split
            elif len(x_split) == 6:
                phone, tone, ws, pw, stype, word = x_split
            else:
                logger.warning(f"Wrong tacolab {x_split} {url=}")
                return None
            tacolab_v1.append('\t'.join([phone, tone, '0.0 0.0 0.0 1.0', ws, pw]))
        return tacolab_v1

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
            elif self.is_english_spanish_char(x): # en
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
            else: # blank
                if not (text[i] == " " or text[i].isdigit()):
                    print("text[i]: ", text[i])
                    return None
                i += 1

        lang = 'en'
        if len_zh_char > len_en_word:
            lang = 'zh'

        if not self.use_code_switch_data:
            if not (len_zh_char == 0 or len_en_word == 0):
                if self.use_lang_cfg:
                    return 'default'
                else:
                    return None
        return lang

    def get_lang_by_text_infer(self, text):
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
            else:
                i += 1

        if len_zh_char > 0:
            if len_en_word > 0:
                lang = 'zh_en'
            else:
                lang = 'zh'
        else:
            if len_en_word > 0:
                lang = 'en'
            else:
                raise NotImplementedError

        return lang

    def get_lang(self, tacolab):
        if len(tacolab[0].split('\t')) != 5:
            if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
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

    def remove_replace_sp(self, labels, utt_id):
        new_labels = []
        for label in labels:
            if len(label.split('\t')) == 7:
                phone, tone, ws, pw, stype, word, unit = label.split('\t')
                if phone == 'sp':
                    if word == '':
                        pass
                    else:
                        assert word in punctuation_all, (word)
                        phone = word
                new_label = '\t'.join([phone, tone, ws, pw, stype, word, unit])
            elif len(label.split('\t')) == 6:
                phone, tone, ws, pw, stype, word = label.split('\t')
                if phone == 'sp':
                    if word == '':
                        pass
                    else:
                        assert word in punctuation_all, (word)
                        phone = word
                new_label = '\t'.join([phone, tone, ws, pw, stype, word])
            elif len(label.split('\t')) == 5:
                # if label.split('\t')[0] == 'sp':
                print(f"labels: {utt_id} \n{labels}")
                new_label = label
            else:
                logger.warning(f"{utt_id} labels wrong format, {labels}, skip")
                return None
            new_labels.append(new_label)

        return new_labels

    def convert_tacolab_to_text_id(self, tacolab, url):
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
                    # if not self.use_sp:
                    #     assert phone != 'sp', (phone, tacolab)

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
                assert len(tacolab[0].split('\t')) == 7 or len(tacolab[0].split('\t')) == 6, (len(tacolab[0].split('\t')), tacolab[0])
                if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword':
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
                    # if not self.use_sp:
                    #     assert phone != 'sp', (phone, tacolab)

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
                    tacolab = self.convert_v3_to_v1(tacolab, url)
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, _, ws, pw = x.split('\t')
                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    assert phone + '_' + tone in self.phonetone_to_int, f"{phone + '_' + tone} not in phonetone set"
                    # if not self.use_sp:
                    #     assert phone != 'sp', (phone, tacolab)

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

    def remove_head(self, labels):
        first_phn = labels[0].split('\t')[0]
        if first_phn == 'phn':
            return labels[1:]
        else:
            return labels
    
    def alignment_all_punc(self, alignment):
        for x in alignment:
            if x.split('\t')[0] not in sil_punc_symbols:
                return False
        return True
    
    def split_text_by_labels(self, text, prompt_labels, labels):
        if not (len(prompt_labels[0].split('\t')) == 6 or len(prompt_labels[0].split('\t')) == 7):
            logger.warning(f"current prompt_labels not match v3, can not be splitted.\n{text}\n{prompt_labels}")
            return None, None

        # get prompt word_char list
        prompt_word_list = [x.split('\t')[5] for x in prompt_labels if x.split('\t')[5] != '']

        # last 2 of prompt word_char
        if prompt_labels[-1].split('\t')[5] == '':
            prompt_last_2_word = prompt_word_list[-1]
        else:
            prompt_last_2_word = ''
            if prompt_word_list[-2] in '*.?+$^[](){}|\/':
                prompt_last_2_word = '\\' + prompt_word_list[-2]
            else:
                prompt_last_2_word = prompt_word_list[-2]

            if prompt_word_list[-1] in '*.?+$^[](){}|\/':
                prompt_last_2_word = prompt_last_2_word + '\\' + prompt_word_list[-1]
            else:
                prompt_last_2_word = prompt_last_2_word + prompt_word_list[-1]

        try:
            # find prompt_last_2_word
            prmopt_start_ends = []
            for m in re.finditer(prompt_last_2_word, text):
                prmopt_start_ends.append([m.start(), m.end()])
        except Exception:
            logger.warning(f"re failed.\n{prompt_last_2_word}\n{text}")
            return None, None

        # if not, reverse TN
        if len(prmopt_start_ends) == 0:
            case_num = 2
            case_index = 0
            while len(prmopt_start_ends) == 0 and case_index < case_num:
                if case_index == 0: # -
                    prompt_last_2_word_ = prompt_last_2_word.replace('-', '—')
                elif case_index == 1: # TN会把句首大写字母变成小写
                    prompt_last_2_word_ = prompt_last_2_word.capitalize()
                try:
                    for m in re.finditer(prompt_last_2_word_, text):
                        prmopt_start_ends.append([m.start(), m.end()])
                except Exception:
                    logger.warning(f"re failed.\n{prompt_last_2_word_}\n{text}")
                    return None, None

                case_index += 1

        # if not, get start_ends
        start_ends = []
        if len(prmopt_start_ends) == 0 or len(prmopt_start_ends) > 10:
            prmopt_start_ends = []
            word_list = [x.split('\t')[5] for x in labels if x.split('\t')[5] != '']
            first_1_word = word_list[0]
            try:
                for m in re.finditer(first_1_word, text):
                    start_ends.append([m.start(), m.end()])
            except Exception:
                logger.warning(f"re failed.\n{first_1_word}\n{text}")
                return None, None

            if len(start_ends) == 0:
                case_num = 2
                case_index = 0
                while len(start_ends) == 0 and case_index < case_num:
                    if case_index == 0: # -
                        first_1_word_ = first_1_word.replace('-', '—')
                    elif case_index == 1: # TN会把句首大写字母变成小写
                        first_1_word_ = first_1_word.capitalize()
                    try:
                        for m in re.finditer(first_1_word_, text):
                            start_ends.append([m.start(), m.end()])
                    except Exception:
                        logger.warning(f"re failed.\n{first_1_word_}\n{text}")
                        return None, None

                    case_index += 1

        # skip
        if not (len(prmopt_start_ends) > 0 or len(start_ends) > 0):
            logger.warning(f"current text is hard to split, skip.\n{text}\n{prompt_labels}")
            return None, None

        # find 
        if len(prmopt_start_ends) > 0:
            if len(prmopt_start_ends) == 1:
                pos = prmopt_start_ends[0][-1]
            elif len(prmopt_start_ends) > 1:
                ratio = len(prompt_labels) / (len(prompt_labels) + len(labels))
                temp_len = int(ratio * len(text))

                min_dis = 10000
                for x in prmopt_start_ends:
                    start, end = x
                    if start <= temp_len and temp_len <= end:
                        pos = end
                        break
                    dis = min(abs(temp_len - start), abs(temp_len - end))
                    if dis < min_dis:
                        min_dis = dis
                        pos = end
        elif len(start_ends) > 0:
            if len(start_ends) == 1:
                pos = start_ends[0][0]
            elif len(start_ends) > 1:
                ratio = len(prompt_labels) / (len(prompt_labels) + len(labels))
                temp_len = int(ratio * len(text))

                min_dis = 10000
                for x in start_ends:
                    start, end = x
                    if start <= temp_len and temp_len <= end:
                        pos = start
                        break
                    dis = min(abs(temp_len - start), abs(temp_len - end))
                    if dis < min_dis:
                        min_dis = dis
                        pos = start

        prompt_text = text[:pos]
        text = text[pos:]
        return prompt_text, text


class ContinuousCollator(object):
    def __init__(self, tokenizer_pad, block_sparse=False, use_bpe=False, use_extra_tag=False, min_crop_ratio=0.15, max_crop_ratio=0.6):
        self.pad = tokenizer_pad
        self.block_sparse = block_sparse
        self.use_bpe = use_bpe
        self.use_extra_tag = use_extra_tag
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio

    def __call__(self, batches):
        results = []
        for item in batches:
            if item is not None:
                results.append(item)

        if len(results) == 0:
            return None
        ret_dict = defaultdict(list)

        text_lens = []
        bn_lens = []
        bpe_lens = []
        wav_lens = []
        spk_embd_masks = []
        # prompt_text_lens = []
        prompt_bn_lens = []
        # utt_ids = []
        # tag_ids = []
        for x in results:
            text_lens.append(x["phone"].shape[0])
            if self.use_bpe:
                bpe_lens.append(x["bpe_seq"].shape[0])
            else:
                bpe_lens = None
            bn_lens.append(x["bn"].shape[0])
            wav_lens.append(x["wav"].shape[0])
            spk_embd_masks.append(x["spk_embd_mask"])
            prompt_bn_lens.append(x["prompt_bn"].shape[0])

            # bpe_lens.append(x["bpe_seq"].shape[0])
            # utt_ids.append(x["utt_id"])
            # if x["tag_id"] is not None:
            #     tag_ids.append(x["tag_id"])

        max_text_len = max(text_lens)
        max_bn_len = max(bn_lens)
        max_wav_len = max(wav_lens)
        if self.use_bpe:
            max_bpe_len = max(bpe_lens)
        # if prompt_bn_lens is not None:
            # max_prompt_text_len = max(prompt_text_lens)
        max_prompt_bn_len = max(prompt_bn_lens)

        text_lens = torch.from_numpy(np.asarray(text_lens))
        bn_lens = torch.from_numpy(np.asarray(bn_lens))
        wav_lens = torch.from_numpy(np.asarray(wav_lens))
        if self.use_bpe:
            bpe_lens = torch.from_numpy(np.asarray(bpe_lens))
        # if prompt_bn_lens is not None:
            # prompt_text_lens = torch.from_numpy(np.asarray(prompt_text_lens))
        prompt_bn_lens = torch.from_numpy(np.asarray(prompt_bn_lens))

        # if prompt_bn_lens is not None:
        max_seq_len = max(prompt_bn_lens + text_lens + bn_lens)
        # else:
        #     max_seq_len = max(text_lens + bn_lens)

        ret_dict["text_lens"] = text_lens
        ret_dict["bn_lens"] = bn_lens
        ret_dict["bpe_lens"] = bpe_lens
        ret_dict["wav_lens"] = wav_lens / max_wav_len
        ret_dict["spk_embd_masks"] = spk_embd_masks
        # ret_dict["prompt_text_lens"] = prompt_text_lens
        ret_dict["prompt_bn_lens"] = prompt_bn_lens
        # ret_dict["utt_id"] = utt_ids
        # if len(tag_ids) > 0:
        #     ret_dict["tag_id"] = np.asarray(tag_ids)

        min_crop_len = int(self.min_crop_ratio * min(bn_lens))
        max_crop_len = int(self.max_crop_ratio * min(bn_lens))
        # assert max_crop_len > 0, (max_crop_len, min(bn_lens), bn_lens)
        crop_len = np.random.randint(min_crop_len, max_crop_len)
        crop_bns = []

        # length padding
        for x in results:
            for k, v in x.items():
                if v is None:
                    ret_dict[k] = None
                    continue
                if k == "__key__":
                    continue
                if k == "bn":
                    crop_begin = np.random.randint(0, v.shape[0] - crop_len)
                    crop_bns.append(v[crop_begin:crop_begin + crop_len])
                    v = np.pad(
                        v,
                        ((0, max_bn_len - v.shape[0]), (0, 0)),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "stop_token":
                    v = np.pad(
                        v,
                        (0, max_seq_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "lang_seq":
                    v = np.pad(
                        v,
                        (0, max_bn_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "spk_seq":
                    v = np.pad(
                        v,
                        (0, max_bn_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "bpe_seq":
                    if self.use_bpe:
                        v = np.pad(
                            v,
                            (0, max_bpe_len - v.shape[0]),
                            mode="constant",
                            constant_values=self.pad,
                        )
                elif k == "wav":
                    v = np.pad(
                        v,
                        (0, max_wav_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "prompt_bn":
                    v = np.pad(
                        v,
                        ((0, max_prompt_bn_len - v.shape[0]), (0, 0)),
                        mode="constant",
                        constant_values=self.pad,
                    )
                # elif k == "prompt_phone":
                #     if v is not None:
                #         v = np.pad(
                #             v,
                #             ((0, max_prompt_text_len - v.shape[0]), (0, 0)),
                #             mode="constant",
                #             constant_values=self.pad,
                #         )
                # elif k == "prompt_tone":
                #     if v is not None:
                #         v = np.pad(
                #             v,
                #             ((0, max_prompt_text_len - v.shape[0]), (0, 0)),
                #             mode="constant",
                #             constant_values=self.pad,
                #         )
                elif k not in ["utt_id", "tag_id", "spk_embd_mask"]:
                    v = np.pad(
                        v,
                        (0, max_text_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                ret_dict[k].append(v)

        for k in ret_dict.keys():
            if k == "bn":
                ret_dict[k] = np.stack(ret_dict[k], axis=0)
            elif k == "wav":
                ret_dict[k] = np.stack(ret_dict[k], axis=0)
            elif k == "prompt_bn":
                ret_dict[k] = np.stack(ret_dict[k], axis=0)
            elif k != "utt_id":
                if ret_dict[k] is not None:
                    ret_dict[k] = np.asarray(ret_dict[k])
            else:
                continue
            if ret_dict[k] is not None:
                try:
                    ret_dict[k] = torch.from_numpy(ret_dict[k])
                except Exception:
                    print("ret_dict[k]: ", ret_dict[k])

        ret_dict["crop_bn"] = torch.from_numpy(np.stack(crop_bns, axis=0))

        to_long_list = ["phone", "tone", "stop_token", "lang_seq", "spk_seq", "bpe_seq", "tag_id", "phonetone"]
        for x in to_long_list:
            if x in ret_dict:
                if ret_dict[x] is not None:
                    ret_dict[x] = ret_dict[x].long()
        return ret_dict


if __name__ == "__main__":

    batcher_config = {
        "buckets": list(range(0, 6000, 100)),  # [0, 100, 200 ... 4000] 4000以上的可以先丢掉
        "dynamic_batch": True,
        "maximum_bucket_size": 200,
        "length_fn": "lambda x: x[\"stop_token\"].shape[0]" # seq.shape
    }

    dataset = ContinuousTTSLangSpkSerAliDataset(data_id=534, # 534, 635
                                batcher_config=batcher_config,
                                drop_last=False,
                                use_lang_id=False,
                                lang2id="recipes/text2semantic/datasets/dict/lang2id.json",
                                use_spk_id=True,
                                spk2id="recipes/text2semantic/datasets/dict/spk2id.parquet.json",
                                input_type='2dim',
                                use_code_switch_data=True,
                                use_extra_tag=True,
                                spk2tag="recipes/text2semantic/datasets/dict/spk2tag.parquet.json",
                                use_sp=False,
                                refenc_cfg_rate=0.0,
                                split_cfg_rate=1.0,
                                )

    collector = ContinuousCollator(tokenizer_pad=0)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=None, collate_fn=collector)
    import tqdm

    # for item in tqdm.tqdm(dataset):
    for item in tqdm.tqdm(dataloader):
        # print(item)
        # lengths = item[-1]
        # batch_size = len(lengths)
        # print(batch_size, max(lengths), batch_size * max(lengths))
        exit()
