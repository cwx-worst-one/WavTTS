import os
import string

from torch.utils.data import Dataset
from zhon.hanzi import punctuation

from samantha.utils.sami_tacolabel import generate_tacolabels_from_textstr_punc

punctuation_all = punctuation + string.punctuation

def is_english_char(char):
    if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a'):
        return True
    else:
        return False

def is_english_spanish_char(char):
    special_Spanish_chars_list = ['á', 'é', 'í', 'ó', 'ú', 'Á', 'É', 'Í', 'Ó', 'Ú', 'ñ', 'Ñ', '¡', '¿', 'ü', 'Ü']
    if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a') or char in special_Spanish_chars_list:
        return True
    else:
        return False


def get_lang_by_text(text):
    text = text.replace('\'', '')
    # en, zh
    len_en_word = 0
    len_zh_char = 0
    i = 0
    while i < len(text):
        x = text[i]
        if x in punctuation_all:  # punc
            i += 1
            continue
        elif u'\u4e00' <= x <= u'\u9fff':  # zh
            len_zh_char += 1
            i += 1
        elif is_english_spanish_char(x): # en with little spanish
            i += 1
            if i >= len(text):
                len_en_word += 1
                break
            while is_english_spanish_char(text[i]):
                i += 1
                if i >= len(text):
                    break
            len_en_word += 1
            continue
        else:
            i += 1

    lang = 'en'
    if len_zh_char > len_en_word:
        lang = 'zh'
    # TODO: japan
    return lang


def generate_tacolabels_engine(text_str):
    lang_key = get_lang_by_text(text_str)
    if lang_key == None:
        logger.info(f"{text_str}: Wrong lang_key")
        return None
    tacolab = None
    if lang_key in ['zh', 'zh_en']:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "Chinese_v3_punc")
    elif lang_key in ['en']:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "English_v3_punc")
    elif lang_key in ['jp']:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "Japanese_v3_punc")  # TODO: add japan engine
    return tacolab


class ICLMetaDataset(Dataset):
    def __init__(self, meta_lst):
        self.meta = self._parse_meta(meta_lst)

    def _parse_meta(self, meta_lst):
        meta = []
        if meta_lst is None or meta_lst == '':
            return meta
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                assert len(line.strip().split("|")) == 4, line
                uttid, prompt_text, prompt_wav_path, text = line.strip().split("|")
                if not os.path.isabs(prompt_wav_path):
                    prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                meta.append([uttid, prompt_text, prompt_wav_path, text])
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        uttid, prompt_text, prompt_wav_path, infer_text = self.meta[index]
        prompt_lab = generate_tacolabels_engine(prompt_text).decode()
        infer_lab = generate_tacolabels_engine(infer_text).decode()

        return uttid, prompt_lab, prompt_wav_path, infer_lab
