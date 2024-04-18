import logging
import os

from torch.utils.data import Dataset

from samantha.dataio.lite.utils.lang import get_lang_by_text
from samantha.utils.sami_tacolabel import generate_tacolabels_from_textstr_punc

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

        return uttid, prompt_lab, prompt_wav_path, infer_lab
