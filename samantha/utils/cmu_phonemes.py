import json
import string
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

import samantha.utils.hdfs_helper as hh
from samantha.utils.distributed import rank_zero_first

phoneme_vocab_size = 80  # 69 phonemes + x special tokens
phoneme_padding_value = 70
phoneme_unk_value = 71
phoneme_newline_value = 72

raw_dictionary_hdfs_path = (
    "hdfs://harunava/home/byte_speech_sv/andrew.shaw/l2s/data/cmu/cmu_dict.txt"
)
dictionary_hdfs_path = "hdfs://harunava/home/byte_speech_sv/andrew.shaw/l2s/data/cmu/diction_word_phoneme.json"

# CN paths
# raw_dictionary_hdfs_path = "hdfs://haruna/home/byte_speech_sv/andrew.shaw/l2s/data/cmu/cmu_dict.txt"
# dictionary_hdfs_path = "hdfs://haruna/home/byte_speech_sv/andrew.shaw/l2s/data/cmu/diction_word_phoneme.json"


def map_raw_dict(hdfs_path, cache_dir):
    "For reprocessing dictionary with updated vocabulary"
    cache_dir = Path(cache_dir)
    with rank_zero_first():
        cmu_dict_fp = cache_dir / "cmu_dict.txt"
        if not cmu_dict_fp.exists():
            hh.get(hdfs_path, cmu_dict_fp)
    with open(cmu_dict_fp) as f:
        file_dict = f.read()
    file_dict = file_dict.split("\n")

    diction_word_phoneme = {}
    diction_phoneme_id = {}
    idx = 0
    for temp in tqdm(file_dict):
        arr = temp.split(" ")
        word = arr[0].lower()
        diction_word_phoneme[word] = []
        for x in arr[1:]:
            x = x.lower()
            if x == "" or x == " ":
                continue
            if x not in diction_phoneme_id:
                diction_phoneme_id[x] = idx
                idx += 1
            phoneme_id = diction_phoneme_id[x]
            diction_word_phoneme[word].append(phoneme_id)
    print("len diction:", len(diction_phoneme_id))
    print("len word phoneme:", len(diction_word_phoneme))
    return diction_word_phoneme


@lru_cache(maxsize=1)
def init_dict(hdfs_path, cache_dir):
    phoneme_dir = Path(cache_dir)
    diction_fp = phoneme_dir / "diction_word_phoneme.json"
    with rank_zero_first():
        if not diction_fp.exists():
            hh.get(hdfs_path, diction_fp)
    with open(diction_fp, "r") as f:
        return json.load(f)


_punc_translator_space = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)
_punc_translator = str.maketrans("", "", string.punctuation)


def remove_punctuation(text, replace_with_space=True):
    if replace_with_space:
        return text.translate(_punc_translator_space)
    else:
        return text.translate(_punc_translator)


def text_to_words(text):
    words = text.lower().split(" ")
    return [w.strip() for w in words if w.strip()]


def word_to_phonemes(dictionary, word, allow_unknown=True):
    if word == "<n>":
        return [phoneme_newline_value]
    if word in dictionary:
        return dictionary[word]
    norm_word = remove_punctuation(word, replace_with_space=False)
    if norm_word == word:
        if allow_unknown:
            return [phoneme_unk_value]
        raise Exception(
            f"Could not find word {word}. Must allow_unknown or add word to dictionary"
        )
    if norm_word in dictionary:
        return dictionary[norm_word]
    norm_word_space = remove_punctuation(word, replace_with_space=True).split(" ")
    phonemes = [word_to_phonemes(dictionary, n, allow_unknown) for n in norm_word_space]
    return sum(phonemes, [])


class CMUPhonemeTokenizer(object):
    def __init__(
        self,
        allow_unknown=False,
        hdfs_path=dictionary_hdfs_path,
        cache_dir=".module_cache",
    ):
        self.allow_unknown = allow_unknown
        self.pad_id = phoneme_padding_value
        self.unk_id = phoneme_unk_value
        self.vocab_size = phoneme_vocab_size
        self.dictionary = init_dict(hdfs_path, cache_dir)

    def __call__(self, text):
        words = text_to_words(text)
        phonemes = [
            word_to_phonemes(self.dictionary, w, self.allow_unknown) for w in words
        ]
        phonemes = sum(phonemes, [])
        return {"input_ids": phonemes}
