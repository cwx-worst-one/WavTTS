import math
import string
from typing import Dict, List, Union

import torch
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


def split_prompt_with_pad(length: int, croplen: int, minlen: int):
    """
    split the length into a list of indices
    if the length is less than minlen, return empty tensor
    if the length is less than croplen, repeat pad
    if the length is greater than croplen, extend left
    if the length is greater than croplen and the redundance is less than minlen, extend right

    Args:
        length: the length of the text
        croplen: the length of the crop
        minlen: the minimum length of the text

    Returns:
        the indices of the prompt after split
    """
    redundance = length % croplen
    if length <= minlen:
        return torch.Tensor(())
    if redundance <= minlen:
        return torch.arange(length - redundance)

    x_index = torch.arange(length)
    if redundance == 0:
        return x_index
    if length < croplen:
        # repeat pad
        return x_index.repeat(croplen // length + 1)[-croplen:]
    else:
        # extend left
        return torch.cat((x_index[:length - redundance], x_index[-croplen:]))
