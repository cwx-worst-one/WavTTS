import logging
import os
import time

from torch.utils.data import Dataset

from samantha.dataio.lite.utils.lang import get_lang_by_text
from samantha.utils.sami_tacolabel import (
    generate_tacolabels_from_textstr_punc,
    split_text_engine,
)

logger = logging.getLogger(__name__)


def generate_tacolabels_engine(text_str):
    lang_key = get_lang_by_text(text_str)
    if lang_key == None:
        logger.info(f"{text_str}: Wrong lang_key")
        return None
    tacolab = None
    if lang_key in ["zh", "zh_en"]:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "Chinese_v3_punc")
    elif lang_key in ["en"]:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "English_v3_punc")
    elif lang_key in ["jp"]:
        tacolab = generate_tacolabels_from_textstr_punc(
            text_str, "Japanese_v3_punc"
        )  # TODO: add japan engine
    return tacolab


def split_text(text_str, max_paragraph_phoneme_size=240):
    lang_key = get_lang_by_text(text_str)
    if lang_key == None:
        logger.info(f"{text_str}: Wrong lang_key")
        return None
    split_texts = None
    if lang_key in ["zh", "zh_en"]:
        split_texts = split_text_engine(
            text_str,
            "Chinese_v3_punc",
            max_paragraph_phoneme_size=max_paragraph_phoneme_size,
        )
    elif lang_key in ["en"]:
        split_texts = split_text_engine(
            text_str,
            "English_v3_punc",
            max_paragraph_phoneme_size=max_paragraph_phoneme_size,
        )
    return split_texts


class ICLMetaDataset(Dataset):
    def __init__(self, meta_lst):
        self.meta = self._parse_meta(meta_lst)

    def _parse_meta(self, meta_lst):
        meta = []
        if meta_lst is None or meta_lst == "":
            return meta
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                assert len(line.strip().split("|")) == 4, line
                uttid, prompt_text, prompt_wav_path, text = line.strip().split("|")
                if not os.path.isabs(prompt_wav_path):
                    prompt_wav_path = os.path.join(
                        os.path.dirname(meta_lst), prompt_wav_path
                    )
                meta.append([uttid, prompt_text, prompt_wav_path, text])
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        uttid, prompt_text, prompt_wav_path, infer_text = self.meta[index]

        prompt_lab = generate_tacolabels_engine(prompt_text).decode()
        infer_lab = generate_tacolabels_engine(infer_text).decode()

        return uttid, prompt_lab, prompt_text, prompt_wav_path, infer_lab, infer_text


class ICLMetaDatasetShortform(ICLMetaDataset):

    def __init__(
        self,
        meta_lst,
        max_paragraph_phoneme_size_zh=240,
        max_paragraph_phoneme_size_en=240,
        use_offline_splittext=False,
        offline_splittext_path="",
    ):
        super().__init__(meta_lst)
        self.max_paragraph_phoneme_size_zh = max_paragraph_phoneme_size_zh
        self.max_paragraph_phoneme_size_en = max_paragraph_phoneme_size_en
        self.use_offline_splittext = use_offline_splittext

        self.utt2texts = {}
        if self.use_offline_splittext:
            assert os.path.exists(offline_splittext_path), offline_splittext_path
            with open(offline_splittext_path) as f:
                lines = f.readlines()
                for line in lines:
                    utt, text = line.strip().split("\t")
                    texts = text.split("|")
                    self.utt2texts[utt] = texts

    def __getitem__(self, index):
        uttid, prompt_text, prompt_wav_path, infer_text = self.meta[index]

        prompt_text = prompt_text.replace("~", "。").replace("～", "。")
        infer_text = infer_text.replace("~", "。").replace("～", "。")

        prompt_lab = generate_tacolabels_engine(prompt_text).decode()

        infer_labs = []
        if self.use_offline_splittext:
            infer_texts = self.utt2texts[uttid]
        else:
            infer_texts = self.split_text(infer_text)

        for infer_text in infer_texts:
            infer_lab = generate_tacolabels_engine(infer_text).decode()
            infer_labs.append(infer_lab)

        return uttid, prompt_lab, prompt_text, prompt_wav_path, infer_labs, infer_texts

    def split_text(self, text_str):
        lang_key = get_lang_by_text(text_str)
        if lang_key == None:
            logger.info(f"{text_str}: Wrong lang_key")
            return None
        split_texts = None
        if lang_key in ["zh", "zh_en"]:
            split_texts = split_text_engine(
                text_str,
                "Chinese_v3_punc",
                max_paragraph_phoneme_size=self.max_paragraph_phoneme_size_zh,
            )
        elif lang_key in ["en"]:
            split_texts = split_text_engine(
                text_str,
                "English_v3_punc",
                max_paragraph_phoneme_size=self.max_paragraph_phoneme_size_en,
            )
        return split_texts
