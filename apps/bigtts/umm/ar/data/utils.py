import math
import string
import numpy as np
from typing import Dict, List, Union

from transformers import BertTokenizer, PreTrainedTokenizer, Wav2Vec2PhonemeCTCTokenizer

_TRANSFORMER_TOKENIZERS_CLS: Dict[str, PreTrainedTokenizer] = {
    "bert": BertTokenizer,
    "wordpiece": BertTokenizer,
    "phoneme": Wav2Vec2PhonemeCTCTokenizer,
}


def gen_duration_buckets(
    min_duration: int = 2, max_duration: int = 30, sample_rate: int = 1
) -> List[int]:
    assert 0 < min_duration < max_duration

    buckets = []
    sec = min_duration
    while sec <= max_duration:
        buckets.append(sec)
        sec += math.ceil(sec * 0.1)

    if buckets[-1] < max_duration:
        buckets.append(max_duration)

    return [b * sample_rate for b in buckets]


def init_tokenizer(name: str = None, **kwargs) -> Union[str, PreTrainedTokenizer, None]:
    if name in _TRANSFORMER_TOKENIZERS_CLS:
        model_name = kwargs.get("model_name")
        if model_name is None:
            raise ValueError(f"model_name must be specified for {name} tokenizer")
        return _TRANSFORMER_TOKENIZERS_CLS[name].from_pretrained(model_name)

    if name == "sami":
        return name

    return None


def normalize_text(text: str):
    nlp_punctuation = string.punctuation.replace("'", "")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    return text.translate(str.maketrans("", "", nlp_punctuation)).strip()


def get_text_lang_ids(tacolab, textlang2id):
    if len(tacolab[0].split("\t")) != 5:
        if (
            tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
            or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
            or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\talignment"
            or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit\talignment"
        ):
            tacolab = tacolab[1:]
    prefix_phn_list = [x.split("\t")[0][:2] for x in tacolab]
    # new_prefix_phn_list = []
    text_langs = []
    text_lang_ids = []
    i = 0
    while i < len(prefix_phn_list):
        # import pdb; pdb.set_trace()
        cur_prefix_phn = prefix_phn_list[i]

        if i == len(prefix_phn_list) - 1:
            next_prefix_phn = ""
        else:
            next_prefix_phn = prefix_phn_list[i + 1]

        if cur_prefix_phn == "C0":
            text_lang = "zh"
        elif cur_prefix_phn == "E0":
            text_lang = "en"
        else:
            if i == 0:
                if next_prefix_phn == "C0":
                    text_lang = "zh"
                elif next_prefix_phn == "E0":
                    text_lang = "en"
                else:
                    text_lang = None
                    j = i + 2
                    while j < len(prefix_phn_list):
                        next_prefix_phn = prefix_phn_list[j]
                        if next_prefix_phn in ["C0", "E0"]:
                            if next_prefix_phn == "C0":
                                text_lang = "zh"
                            elif next_prefix_phn == "E0":
                                text_lang = "en"
                            break
                        else:
                            j += 1
                    if text_lang is None:
                        return None
            else:
                if cur_prefix_phn == "si":
                    i += 1
                    continue
                else:
                    text_lang = text_langs[-1]
        # print(f"{len(prefix_phn_list)} {i} {text_langs}")
        # new_prefix_phn_list.append(cur_prefix_phn)
        text_langs.append(text_lang)
        text_lang_ids.append(textlang2id[text_lang])
        i += 1

    # return (prefix_phn_list, text_langs, text_lang_ids)
    text_lang_ids = np.asarray(text_lang_ids)
    return text_lang_ids, text_langs  # , new_prefix_phn_list
