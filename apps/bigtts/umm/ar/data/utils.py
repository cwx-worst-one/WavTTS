import math
import string
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
