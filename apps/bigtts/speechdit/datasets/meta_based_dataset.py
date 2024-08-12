import imp
import os
from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
import librosa
import re
from zhon.hanzi import punctuation
import string

from recipes.text2semantic.datasets.sami_tacolabel import (
    generate_tacolabels_from_textstr_punc,
)
from samantha.dataio.lite.utils.phone_to_id import PhoneToId
from samantha.dataio.lite.utils.speech_alignment import GetAlignment

def has_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))


def replace_pattern(text, replacement="AA"):
    # 定义一个正则表达式模式来匹配英语单词
    # text = re.sub(r'\bop\b', replacement, text)
    pattern = r'[a-zA-Z-\']+'
    replaced_text = re.sub(pattern, replacement, text)
    pattern = r'[…]'
    replaced_text = re.sub(pattern, replacement, replaced_text)
    return replaced_text


class TTSTestDataset(Dataset):
    def __init__(self, meta_lst, lang="en", syn_lang="en", 
            alignment_path=None, infer_mode="dit"):
        self.phone2id = PhoneToId()
        self.infer_mode = infer_mode
        self.meta = self._parse_meta(meta_lst)
        self.alignment_path = alignment_path
        # infer mode 两种类型，dit和cla
        if infer_mode == "dit":
            assert self.alignment_path is not None
        self.lang2fronted = {
            "zh": "Chinese_v3_punc",
            "en": "English_v3_punc",
            "jp": "Japanese_v3_punc",
            "mx": "Spanish_v3_punc",
            "br": "Portuguese_v3_punc",
            "id": "Indonesia_v3_punc",
            "fr": "French_v3_punc",
            "de": "Gemandy_v3_punc"
        }
        self.lang = lang
        self.syn_lang = syn_lang

        self.frontend_version = self.lang2fronted[lang]
        self.syn_frontend_version = self.lang2fronted[syn_lang]
        if syn_lang == "mx":
            self.syn_lang = "es-mx"
        if syn_lang == "br":
            self.syn_lang = "pt-br"
        self.lang2fronted = {
            "zh": "Chinese_v3_punc",
            "en": "English_v3_punc",
            "jp": "Japanese_v3_punc",
            "mx": "Spanish_v3_punc",
            "br": "Portuguese_v3_punc",
            "id": "Indonesia_v3_punc",
            "fr": "French_v3_punc",
            "de": "Gemandy_v3_punc"
        }
        self.lang = lang
        self.syn_lang = syn_lang

        self.frontend_version = self.lang2fronted[lang]
        self.syn_frontend_version = self.lang2fronted[syn_lang]
        if syn_lang == "mx":
            self.syn_lang = "es-mx"
        if syn_lang == "br":
            self.syn_lang = "pt-br"

    def _parse_meta(self, meta_lst):
        meta = []
        if self.infer_mode == "cla-concat":
            with open(meta_lst, "r", encoding="utf8") as f:
                for line in f:
                    uttid, prompt_text, prompt_wav_path, syn_text, raw_prompt_wav_path = line.strip().split("|")
                    if not os.path.isabs(prompt_wav_path):
                        prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                    if not os.path.isabs(raw_prompt_wav_path):
                        raw_prompt_wav_path = os.path.join(os.path.dirname(meta_lst), raw_prompt_wav_path)                    
                    meta.append([uttid, prompt_wav_path, prompt_text, syn_text, raw_prompt_wav_path])
        else:
            with open(meta_lst, "r", encoding="utf8") as f:
                for line in f:
                    uttid, prompt_text, prompt_wav_path, syn_text = line.strip().split("|")
                    if not os.path.isabs(prompt_wav_path):
                        prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                    meta.append([uttid, prompt_wav_path, prompt_text, syn_text])
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        if self.infer_mode == "cla-concat":
            uttid, prompt_wav_path, prompt_text, syn_text, raw_prompt_wav_path = self.meta[index]
        else:
            uttid, prompt_wav_path, prompt_text, syn_text = self.meta[index]
        if self.infer_mode == "dit":
            alignment_file = os.path.join(self.alignment_path, f"{uttid}.pkl")
            if os.path.exists(alignment_file):
                with open(alignment_file, "rb") as db:
                    words_align, phones_align = pickle.load(db)
            else:
                words_align, phones_align = GetAlignment(prompt_wav_path, prompt_text)
                with open(alignment_file, "wb") as db:
                    pickle.dump((words_align, phones_align), db)

            if has_chinese(prompt_text):
                frontend_version = "Chinese_v3_punc"
            else:
                frontend_version = "English_v3_punc"

            prompt_id = os.path.basename(prompt_wav_path)[:-4]
            prompt_taco_file = os.path.join(self.alignment_path, f"{prompt_id}_frontend.pkl")
            if os.path.exists(prompt_taco_file):
                with open(prompt_taco_file, "rb") as db:
                    tacolab = pickle.load(db)
            else:
                tacolab = generate_tacolabels_from_textstr_punc(prompt_text, frontend_version)
                while tacolab is None:
                    tacolab = generate_tacolabels_from_textstr_punc(prompt_text, self.frontend_version)
                tacolab = tacolab.decode()
                tacolab = tacolab.strip().split("\n")
                with open(prompt_taco_file, "wb") as db:
                    pickle.dump(tacolab, db)
            prompt_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)

            if has_chinese(syn_text):
                frontend_version = "Chinese_v3_punc"
            else:
                frontend_version = "English_v3_punc"
            syn_taco_file = os.path.join(self.alignment_path, f"{uttid}_frontend.pkl")

            if os.path.exists(syn_taco_file):
                with open(syn_taco_file, "rb") as db:
                    tacolab = pickle.load(db)
            else:
                tacolab = generate_tacolabels_from_textstr_punc(syn_text, frontend_version)
                while tacolab is None:
                    tacolab = generate_tacolabels_from_textstr_punc(syn_text, frontend_version)
                tacolab = tacolab.decode()
                tacolab = tacolab.strip().split("\n")
                with open(syn_taco_file, "wb") as db:
                    pickle.dump(tacolab, db)
            syn_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
        else:
            # 基本假定是第一条数据是没问题的
            # cla mode不需要alignment
            try:
                if os.path.isfile(prompt_text):
                    tacolab = open(prompt_text).read().split('\n')[1:]
                    prompt_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
                    words_align, phones_align = [], []
                else:
                    words_align, phones_align = GetAlignment(prompt_wav_path, prompt_text)
                    tacolab = generate_tacolabels_from_textstr_punc(prompt_text, self.frontend_version, self.lang).decode()
                    tacolab = tacolab.strip().split("\n")
                    prompt_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
            except:
                print("####### still continue  ########")
                print("[prompt_path]", prompt_wav_path)

            try:
                if os.path.isfile(syn_text):
                    tacolab = open(syn_text).read().split('\n')[1:]
                    syn_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
                    words_align, phones_align = [], []

                else:
                    tacolab = generate_tacolabels_from_textstr_punc(syn_text, self.syn_frontend_version,
                                                                    self.syn_lang).decode()
                    tacolab = tacolab.strip().split("\n")
                    syn_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
            except:
                print('#########[using first record]#######')
                uttid, prompt_wav_path, prompt_text, syn_text = self.meta[0]
                print("[prompt_path]", prompt_wav_path)
                if os.path.isfile(prompt_text):
                    tacolab = open(prompt_text).read().split('\n')[1:]
                    prompt_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
                    words_align, phones_align = [], []
                else:
                    words_align, phones_align = GetAlignment(prompt_wav_path, prompt_text)
                    tacolab = generate_tacolabels_from_textstr_punc(prompt_text, self.frontend_version, self.lang).decode()
                    tacolab = tacolab.strip().split("\n")
                    prompt_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
                if os.path.isfile(syn_text):
                    tacolab = open(syn_text).read().split('\n')[1:]
                    syn_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
                    words_align, phones_align = [], []
                else:
                    tacolab = generate_tacolabels_from_textstr_punc(syn_text, self.syn_frontend_version,
                                                                    self.syn_lang).decode()
                    tacolab = tacolab.strip().split("\n")
                    syn_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
        if self.infer_mode == "cla-concat":
            return uttid, prompt_wav_path, prompt_text_id, syn_text_id, words_align, phones_align, raw_prompt_wav_path
        return uttid, prompt_wav_path, prompt_text_id, syn_text_id, words_align, phones_align


class EditTestDataset(Dataset):
    def __init__(self, meta_lst, lang="en"):
        self.phone2id = PhoneToId()
        self.meta = self._parse_meta(meta_lst)
        self.frontend_version = "English_v3_punc" if lang == "en" else "Chinese_v3_punc"

    def _parse_meta(self, meta_lst):
        meta = []
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                uttid, _, _, syn_text, syn_wav_path = line.strip().split("|")
                if not os.path.isabs(syn_wav_path):
                    syn_wav_path = os.path.join(os.path.dirname(meta_lst), syn_wav_path)
                meta.append([uttid, syn_wav_path, syn_text])
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        uttid, syn_wav_path, syn_text = self.meta[index]
        tacolab = generate_tacolabels_from_textstr_punc(syn_text, self.frontend_version).decode()
        tacolab = tacolab.strip().split("\n")
        syn_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
        return uttid, syn_wav_path, syn_text_id

class EditFormalDataset(Dataset):
    def __init__(self, meta_lst, lang="en"):
        self.phone2id = PhoneToId()
        self.meta = self._parse_meta(meta_lst)
        self.frontend_version = "English_v3_punc" if lang == "en" else "Chinese_v3_punc"
        self.lang = lang

    def _parse_meta(self, meta_lst):
        meta = []
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                #uttid, text, origin_wav_path, acc_wav_path = line.strip().split("|")
                uttid, text, origin_wav_path, acc_wav_path, origin_text, target_text = line.strip().split("|")
                if not os.path.isabs(origin_wav_path):
                    origin_wav_path = os.path.join(os.path.dirname(meta_lst), origin_wav_path)
                #meta.append([uttid, text, origin_wav_path, acc_wav_path])
                meta.append([uttid, text, origin_wav_path, acc_wav_path, origin_text, target_text])
        return meta

    def _preprocess(self, sentence):
        sentence = sentence.lower()
        punctuation_all = punctuation + string.punctuation
        for x in punctuation_all:
            if x in ["{", "}", "[", "]"]:
                continue
            sentence = sentence.replace(x, '')
        return sentence


    def _get_origin_text(self, text):
        result = re.sub(r'\[(.*?)\]', '', text)
        result = re.sub(r'\}', '', re.sub(r'\{', '', result))
        return result

    def _get_target_text(self, text):
        result = re.sub(r'\{(.*?)\}', '', text)
        result = re.sub(r'\]', '', re.sub(r'\[', '', result))
        return result
    
    def _get_matches(self, text):
        preprocess_text = re.sub(r'[^\w\{\}\[\]]', '', text)
        result = re.sub(r'\[(.*?)\]', '', preprocess_text)
        matches = re.finditer(r'\{(.*?)\}', result)
        positions = []
        for i, match in enumerate(matches):
            start = match.start()
            length = len(match.group(1))
            start -= int(i * 2)
            positions.append((start, length))   

        result = re.sub(r'\{(.*?)\}', '', preprocess_text)
        matches = re.finditer(r'\[(.*?)\]', result)
        target_positions = []
        for i, match in enumerate(matches):
            start = match.start()
            length = len(match.group(1))
            start -= int(i * 2)
            target_positions.append((start, length))   
        return positions, target_positions

    def __len__(self):
        return len(self.meta)

    
    def __getitem__(self, index):
        #uttid, text, origin_wav_path, acc_wav_path = self.meta[index]
        REPLACE_WORD = "A"
        uttid, text, origin_wav_path, acc_wav_path, origin_text, target_text = self.meta[index]
        origin_text = self._get_origin_text(text)
        target_text = self._get_target_text(text)

        #---for vaild 
        origin_text_re = replace_pattern(origin_text,  replacement=REPLACE_WORD)
        origin_text_preprocess =  re.sub(r'[^\w\{\}\[\]]', '', origin_text_re)

        target_text_re  = replace_pattern(target_text,  replacement=REPLACE_WORD)
        target_text_preprocess = re.sub(r'[^\w\{\}\[\]]', '', target_text_re)
    

        text_re = replace_pattern(text, replacement=REPLACE_WORD) # replace english word to A 
        positions, target_positions = self._get_matches(text_re)
        words_align, phones_align = GetAlignment(origin_wav_path, origin_text)

        confidence = np.mean([x[3] for x in words_align])  

        assert len(positions) == len(target_positions) 

        assert len(origin_text_preprocess) == len(words_align)
        mask_info = []

        for n, (i, j) in enumerate(positions):
            
            start_time = max(words_align[i][1] - 0.15, 0)
            end_time = words_align[i+j-1][2] + 0.15

            # 最老的方案
            #avg_word_time = np.mean([words_align[i+k][2]-words_align[i+k][1] for k in range(j)])
            #add_time = (target_positions[n][1] - j) * avg_word_time

            # 方案1: 用词边界
            # if 0:
            #     c = 0
            #     avg_word_time = 0
            #     for k in range(j):
            #         if words_align[i+k][3] > 0.5:
            #             avg_word_time += words_align[i+k][2]-words_align[i+k][1]
            #             c += 1

            #     if c == 0:
            #         for k in range(len(words_align)):
            #             if words_align[k][3] > 0.5:
            #                 avg_word_time += words_align[i+k][2]-words_align[i+k][1]
            #                 c += 1
            #     avg_word_time /= c
            #     add_time = (target_positions[n][1] - j) * avg_word_time * 1.2

            # 方案2: 用句子边界
            target_sub_text = target_text_preprocess[target_positions[n][0]:target_positions[n][0]+ target_positions[n][1]]
            sub_replace_count = target_sub_text.count(REPLACE_WORD)

            before_length = words_align[i+j-1][2] - words_align[i][1]
            after_length = (words_align[-1][2] - words_align[0][1]) * (target_positions[n][1]+sub_replace_count) / (len(words_align))
            add_time = after_length - before_length
                
            mask_info.append((start_time, end_time, add_time))

        tacolab = generate_tacolabels_from_textstr_punc(target_text, self.frontend_version).decode()
        tacolab = tacolab.strip().split("\n")
        syn_text_id, _, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)

        return uttid, origin_wav_path, acc_wav_path, syn_text_id, mask_info, confidence

class VCTestDataset(Dataset):
    def __init__(self, meta_lst, lang="en", syn_lang="en"):
        self.phone2id = PhoneToId()
        self.meta = self._parse_meta(meta_lst)
        self.frontend_version = "English_v3_punc" if lang == "en" else "Chinese_v3_punc"
        self.syn_frontend_version = "English_v3_punc" if syn_lang == "en" else "Chinese_v3_punc"

    def _parse_meta(self, meta_lst):
        meta = []


class VCTestDataset(Dataset):
    def __init__(self, meta_lst, lang="en", syn_lang="en"):
        self.phone2id = PhoneToId()
        self.meta = self._parse_meta(meta_lst)
        self.frontend_version = "English_v3_punc" if lang == "en" else "Chinese_v3_punc"
        self.syn_frontend_version = "English_v3_punc" if syn_lang == "en" else "Chinese_v3_punc"

    def _parse_meta(self, meta_lst):
        meta = []
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                uttid, prompt_text, prompt_wav_path, syn_text, syn_wav_path = line.strip().split("|")
                if not os.path.isabs(prompt_wav_path):
                    prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                if not os.path.isabs(syn_wav_path):
                    syn_wav_path = os.path.join(os.path.dirname(meta_lst), syn_wav_path)
                meta.append([uttid, prompt_wav_path, prompt_text, syn_text, syn_wav_path])
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        uttid, prompt_wav_path, prompt_text, syn_text, syn_wav_path = self.meta[index]
        tacolab = generate_tacolabels_from_textstr_punc(prompt_text, self.frontend_version).decode()
        tacolab = tacolab.strip().split("\n")
        prompt_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)

        tacolab = generate_tacolabels_from_textstr_punc(syn_text, self.syn_frontend_version).decode()
        tacolab = tacolab.strip().split("\n")
        syn_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)

        return uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id
