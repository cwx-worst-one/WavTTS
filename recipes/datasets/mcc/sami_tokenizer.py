import re
import string
from typing import Tuple, Optional, List, NamedTuple

import torch
from zhon.hanzi import punctuation
from confusables import confusable_characters

punctuation_all = punctuation + string.punctuation
import os
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

try:
    from sami_tts_api.engine import TtsEngine, generate_tts_config
except Exception as e:
    print(f"[Warning] Failed loading sami_tts_api: {e}")

import contextlib
import json

### phones
# silence symbol
sil_symbols = ["sil", "sp", "pau"]

# punc
punctuation_all = list(punctuation + string.punctuation)
special_symbols = ["......", "...", "……", "--", "——"]

# break_symbols: silence symbol + punc
sil_punc_symbols = sil_symbols + punctuation_all + special_symbols

# en
EN_consonant = [
    "E0b",
    "E0ch",
    "E0d",
    "E0dh",
    "E0f",
    "E0g",
    "E0h",
    "E0hh",
    "E0jh",
    "E0k",
    "E0l",
    "E0m",
    "E0n",
    "E0p",
    "E0r",
    "E0s",
    "E0sh",
    "E0t",
    "E0th",
    "E0v",
    "E0w",
    "E0y",
    "E0z",
    "E0zh",
]  # 24

EN_vowel = [
    "E0aa",
    "E0ae",
    "E0ah",
    "E0ao",
    "E0aw",
    "E0ax",
    "E0ay",
    "E0eh",  # cmu_dict
    "E0er",
    "E0ey",
    "E0ih",
    "E0iy",
    "E0ng",
    "E0ow",
    "E0oy",
    "E0uh",
    "E0uw",  # cmu_dict
    "E0en",
]  # 18

# zh
ZH_consonant = [
    "C0b",
    "C0c",
    "C0ch",
    "C0d",
    "C0f",
    "C0g",
    "C0h",
    "C0j",
    "C0k",
    "C0l",
    "C0m",
    "C0n",
    "C0p",
    "C0q",
    "C0r",
    "C0s",
    "C0sh",
    "C0t",
    "C0x",
    "C0z",
    "C0zh",
]

ZH_vowel = [
    "C0a",
    "C0ai",
    "C0air",
    "C0an",
    "C0ang",
    "C0angr",
    "C0anr",
    "C0ao",
    "C0aor",
    "C0ar",
    "C0e",
    "C0ei",
    "C0eir",
    "C0en",
    "C0eng",
    "C0engr",
    "C0enr",
    "C0er",
    "C0i",
    "C0ia",
    "C0ian",
    "C0iang",
    "C0iangr",
    "C0ianr",
    "C0iao",
    "C0iaor",
    "C0iar",
    "C0ie",
    "C0ier",
    "C0ii",
    "C0iii",
    "C0in",
    "C0ing",
    "C0ingr",
    "C0inr",
    "C0io",
    "C0iong",
    "C0iongr",
    "C0iou",
    "C0iour",
    "C0ir",
    "C0ng",
    "C0o",
    "C0or",
    "C0ong",
    "C0ongr",
    "C0ou",
    "C0our",
    "C0u",
    "C0ua",
    "C0uai",
    "C0uair",
    "C0uan",
    "C0uang",
    "C0uangr",
    "C0uanr",
    "C0uar",
    "C0uei",
    "C0ueir",
    "C0uen",
    "C0ueng",
    "C0uengr",
    "C0uenr",
    "C0uer",
    "C0uo",
    "C0uor",
    "C0ur",
    "C0v",
    "C0van",
    "C0vanr",
    "C0ve",
    "C0ver",
    "C0vn",
    "C0vnr",
    "C0vr",
    "C0iir",
    "C0iiir",
]

sep_strs = ["zh_word_sep", "en_word_sep", "syl_sep"]

# special tags
# NOTE (Yilin): Adding special tags changes the vocab, which might affect the
# compatiblity with previous models.
# For singer<#> tags:
# Yilin: There is no upper limit of singer number. To make
# the tags more deterministic, I set 20 singers as the max
# value (It is probably a bad idea to have too many singers).
# Any sample that has a singer tag other than these should
# be discarded.
singer_tags = ["singer" + str(i) for i in range(20)] + ["合", "女", "男"]

# NOTE: **Always update the regex pattern whenver the tags are updated**
# Pattern: "<singer_tag>:<text>"
# The pattern should match all the singer tags in the dataset
# What we select eventually is a subset `singer_tags`.
# Filtering should be done after matching.
singer_pattern = r'^(singer\d+|男|女|合)?:\s*((?:.|\n)*)'

section_tags = ['silence', 'chorus', 'verse', 'bridge', 'inst', 'outro', 'intro']
section_pattern = r'^\[(.*?)\]\s*((?:.|\n)*)'


all_phones = (
    sil_punc_symbols + EN_consonant + EN_vowel + ZH_consonant + ZH_vowel + sep_strs
    # special tags
    + singer_tags + section_tags
)

# Yilin: A full stop (。) will be added by the TTS frontend. It's not
# necessary to use the line break symbol.
# def get_line_break_id():
#     return np.array([len(all_phones) + 1000])

### tones
all_tones = ([str(i) for i in range(15)] + sep_strs
    # special tags
    + singer_tags + section_tags
)

# 0 for padding, 1 for eos
offset = 2

phone_to_int = dict()
for i, phone in enumerate(all_phones):
    if phone not in phone_to_int:
        phone_to_int[phone] = i + offset

tone_to_int = dict()
for i, tone in enumerate(all_tones):
    if tone not in tone_to_int:
        tone_to_int[tone] = i + offset

# ------------------------------------------
# Special tag utils
# ------------------------------------------

def _extract_prefix_tag(text: str, pattern: str) -> Tuple[Optional[str], str]:
    match = re.search(pattern, text)
    if match is None:
        return None, text
    tag, rest_text = match.group(1), match.group(2)
    if tag is not None:
        tag = tag.strip()
    if rest_text is None:
        raise ValueError("Invalid text or regex pattern.")
    return tag, rest_text


def extract_singer_tag(text: str) -> Tuple[Optional[str], str]:
    """Extract the singer tag from the input text.
    :param text: A string that might or might not contain a leading singer tag, followed by a colon.
    :return: The singer tag (None if empty) and the rest of the text without colon and leading whitespace.
    """
    return _extract_prefix_tag(text, singer_pattern)


def add_singer_tag(singer_tag: Optional[str], text: str) -> str:
    if singer_tag:
        return f"{singer_tag}:{text}"
    return text


def extract_section_tag(text: str) -> Tuple[Optional[str], str]:
    """Extract the section tag from the input text.
    :param text: A string that might or might not contain a leading section tag, enclosed by [].
    :return: The section tag (None if empty) and the rest of the text without colon and leading whitespace.
    """
    return _extract_prefix_tag(text, section_pattern)


def add_section_tag(section_tag: Optional[str], text: str) -> str:
    if section_tag:
        sep = " " if text else ""
        return f"[{section_tag}]{sep}{text}"
    return text

# ------------------------------------------
# Segment parsing
# ------------------------------------------

class Phrase(NamedTuple):
    phonemes: Optional[str] = None
    text: Optional[str] = None
    singer_tag: Optional[str] = None
    section_tag: Optional[str] = None
    time_span: Optional[Tuple[int, int]] = None

    @classmethod
    def parse(
        cls, 
        phonemes: Optional[str] = None,
        text: Optional[str] = None,
        singer_tag: Optional[str] = None, 
        section_tag: Optional[str] = None,
        time_span: Optional[Tuple[int, int]] = None,
    ):
        """ Auto-detect singer tag and section tag from the given phonemes or text
        :param singer_tag: Override the detected singer_tag by this value
        :param section_tag: Override the detected section_tag by this value
        """
        def strip(text: Optional[str]) -> Optional[str]:
            if text is None:
                return None
            s = text.strip()
            if not s:
                return None
            return s

        if phonemes is None:
            _section_tag_p, _singer_tag_p, rest_phonemes = None, None, None
        else:
            _section_tag_p, rest_phonemes = extract_section_tag(phonemes)
            _singer_tag_p, rest_phonemes = extract_singer_tag(rest_phonemes)

        if text is None:
            _section_tag_t, _singer_tag_t, rest_text = None, None, None
        else:
            _section_tag_t, rest_text = extract_section_tag(text)
            _singer_tag_t, rest_text = extract_singer_tag(rest_text)

        # Use extracted tags if tags are not forced
        if section_tag is None:
            section_tag = _section_tag_p if _section_tag_t is None else _section_tag_t
        if singer_tag is None:
            singer_tag = _singer_tag_p if _singer_tag_t is None else _singer_tag_t

        if rest_text is not None:
            rest_text = norm_chinese_text(rest_text).strip()

        return cls(
            phonemes=strip(rest_phonemes),
            text=strip(rest_text),
            singer_tag=singer_tag,
            section_tag=section_tag,
            time_span=time_span,
        )

    @property
    def prefix_tags(self) -> List[Optional[str]]:
        """Tags that should be prepended for each sentence (empty tags are filtered out)"""
        return list(filter(None, [self.section_tag, self.singer_tag]))

    @property
    def has_utterance(self) -> bool:
        return bool(self.phonemes or self.text)

    @property
    def start(self) -> Optional[int]:
        return None if self.time_span is None else self.time_span[0]
    
    @property
    def end(self) -> Optional[int]:
        return None if self.time_span is None else self.time_span[1]

    @property
    def duration(self) -> Optional[int]:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def format_phonemes(self) -> str:
        return self._format_str(self.phonemes)

    def format_text(self) -> str:
        return self._format_str(self.text)

    def _format_str(self, _str) -> str:
        if _str is None:
            _str = ""
        return add_section_tag(self.section_tag, add_singer_tag(self.singer_tag, _str))


def convert_v3_to_v1(tacolab):
    tacolab_v1 = []
    # en, zh
    if (
        tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
        or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
    ):
        tacolab = tacolab[1:]
    for x in tacolab:
        x_split = x.split("\t")
        if len(x_split) == 7:
            phone, tone, ws, pw, stype, word, _ = x_split
        elif len(x_split) == 6:
            phone, tone, ws, pw, stype, word = x_split
        else:
            print("Wrong tacolab", x_split)
            return None
        tacolab_v1.append("\t".join([phone, tone, "0.0 0.0 0.0 1.0", ws, pw]))
    return tacolab_v1


def is_english_spanish_char(char):
    special_Spanish_chars_list = [
        "á",
        "é",
        "í",
        "ó",
        "ú",
        "Á",
        "É",
        "Í",
        "Ó",
        "Ú",
        "ñ",
        "Ñ",
        "¡",
        "¿",
        "ü",
        "Ü",
    ]
    if (
        ("\u0041" <= char <= "\u005a")
        or ("\u0061" <= char <= "\u007a")
        or char in special_Spanish_chars_list
    ):
        return True
    else:
        return False


def get_lang(tacolab):
    if len(tacolab[0].split("\t")) != 5:
        if (
            tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
            or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
        ):
            tacolab = tacolab[1:]
    prefix_phn_list = [x.split("\t")[0][:2] for x in tacolab]
    if "C0" in prefix_phn_list:
        if "E0" in prefix_phn_list:
            lang = "zh_en"
        else:
            lang = "zh"
    else:
        lang = "en"
    return lang


def get_lang_by_text(text):
    text = text.replace("'", "")
    # en, zh
    len_en_word = 0
    len_zh_char = 0
    i = 0
    while i < len(text):
        x = text[i]
        if x in punctuation_all:  # punc
            i += 1
            continue
        elif "\u4e00" <= x <= "\u9fff":  # zh
            len_zh_char += 1
            i += 1
        elif is_english_spanish_char(x):  # en with little spanish
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
        else:  # blank or digit
            if not (text[i] == " " or text[i].isdigit()):
                return None
            i += 1

    lang = "en"
    if len_zh_char > len_en_word:
        lang = "zh"

    return lang


def _get_special_tag_tokens(tag: str) -> Tuple[int, int, str, str]:
    return phone_to_int[tag], tone_to_int[tag], tag, tag


def convert_labels_to_text_id_zh(tacolab: List[str]) -> Tuple[List[int], List[int], List[str], List[str]]:
    phone_ids = []
    tone_ids = []
    phones = []
    tones = []
    assert len(tacolab[0].split("\t")) == 7, (
        len(tacolab[0].split("\t")),
        tacolab[0],
    )
    if tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit":
        tacolab = tacolab[1:]
    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        x_split = x.split("\t")
        phone, tone, ws, pw, stype, word, unit = x_split
        assert phone in phone_to_int, f"{phone} not in phone set"
        assert tone in tone_to_int, f"{tone} not in tone set"

        phone_ids.append(phone_to_int[phone])
        tone_ids.append(tone_to_int[tone])
        phones.append(phone)
        tones.append(tone)
        if phone[:2] == "C0":
            if unit in ["S", "E"]:
                phone_ids.append(phone_to_int["syl_sep"])
                tone_ids.append(tone_to_int["syl_sep"])
                phones.append("syl_sep")
                tones.append("syl_sep")
                if ws in ["S", "E"]:
                    phone_ids.append(phone_to_int["zh_word_sep"])
                    tone_ids.append(tone_to_int["zh_word_sep"])
                    phones.append("zh_word_sep")
                    tones.append("zh_word_sep")
        elif phone[:2] == "E0":
            if pw != "0":
                phone_ids.append(phone_to_int["en_word_sep"])
                tone_ids.append(tone_to_int["en_word_sep"])
                phones.append("en_word_sep")
                tones.append("en_word_sep")

    return phone_ids, tone_ids, phones, tones


def convert_labels_to_text_id_zh_en(tacolab: List[str]) -> Tuple[List[int], List[int], List[str], List[str]]:
    phone_ids = []
    tone_ids = []
    phones = []
    tones = []
    assert (
        len(tacolab[0].split("\t")) == 7 or len(tacolab[0].split("\t")) == 6
    ), (len(tacolab[0].split("\t")), tacolab[0])
    if (
        tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
        or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
    ):
        tacolab = tacolab[1:]
    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        x_split = x.split("\t")
        if len(x_split) == 7:
            phone, tone, ws, pw, stype, word, unit = x_split
        elif len(x_split) == 6:
            phone, tone, ws, pw, stype, word = x_split
        else:
            print("Wrong tacolab", x_split)
            return None

        assert phone in phone_to_int, f"{phone} not in phone set"
        assert tone in tone_to_int, f"{tone} not in tone set"

        phone_ids.append(phone_to_int[phone])
        tone_ids.append(tone_to_int[tone])
        phones.append(phone)
        tones.append(tone)
        if phone[:2] == "C0":
            if unit in ["S", "E"]:
                phone_ids.append(phone_to_int["syl_sep"])
                tone_ids.append(tone_to_int["syl_sep"])
                phones.append("syl_sep")
                tones.append("syl_sep")
                if ws in ["S", "E"]:
                    phone_ids.append(phone_to_int["zh_word_sep"])
                    tone_ids.append(tone_to_int["zh_word_sep"])
                    phones.append("zh_word_sep")
                    tones.append("zh_word_sep")
        elif phone[:2] == "E0":
            if pw != "0":
                phone_ids.append(phone_to_int["en_word_sep"])
                tone_ids.append(tone_to_int["en_word_sep"])
                phones.append("en_word_sep")
                tones.append("en_word_sep")

    return phone_ids, tone_ids, phones, tones


def convert_labels_to_text_id_en(tacolab: List[str]) -> Tuple[List[int], List[int], List[str], List[str]]:
    phone_ids = []
    tone_ids = []
    phones = []
    tones = []
    if len(tacolab[0].split("\t")) != 5:
        tacolab = convert_v3_to_v1(tacolab)
    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        phone, tone, _, ws, pw = x.split("\t")
        assert phone in phone_to_int, f"{phone} not in phone set"
        assert tone in tone_to_int, f"{tone} not in tone set"
        phone_ids.append(phone_to_int[phone])
        tone_ids.append(tone_to_int[tone])
        phones.append(phone)
        tones.append(tone)
        if pw != "0":
            phone_ids.append(phone_to_int["en_word_sep"])
            tone_ids.append(tone_to_int["en_word_sep"])
            phones.append("en_word_sep")
            tones.append("en_word_sep")

    return phone_ids, tone_ids, phones, tones


_label_conversion_fns = {
    "zh": convert_labels_to_text_id_zh,
    "zh_en": convert_labels_to_text_id_zh_en,
    "en": convert_labels_to_text_id_en,
}


def convert_labels_to_text_id(tacolab: Optional[List[str]], prefix_tags: Optional[List[str]] = None):
    try:
        if tacolab is not None:
            lang = get_lang(tacolab)
            phone_ids, tone_ids, phones, tones = _label_conversion_fns[lang](tacolab)
        else:
            phone_ids, tone_ids, phones, tones = [], [], [], []

        # prepend prefix tags (not very clean)
        if prefix_tags is not None:
            for tag in reversed(prefix_tags):
                _phone_id, _tone_id, _phone, _tone = _get_special_tag_tokens(tag)
                phone_ids = [_phone_id] + phone_ids
                tone_ids = [_tone_id] + tone_ids
                phones = [_phone] + phones
                tones = [_tone] + tones

        return np.stack([phone_ids, tone_ids]), phones, tones
    except ValueError as ve:
        print(ve)
        return None
    except Exception as e:
        print(e)
        return None


def is_chinese_char(ch):
    return ch >= b'\\u4e00' and ch <= b'\\u9fff'


def norm_chinese_char(ch):
    candidates = confusable_characters(ch)
    for c in candidates:
        codepoint = c.encode('unicode_escape')
        if is_chinese_char(codepoint):
            return c
    return None


def norm_chinese_text(s):
    res = ''
    for ch in s:
        normed = norm_chinese_char(ch)
        if not normed or normed == ch:
            res += ch
            continue
        res += normed
    return res


def parse_raw_text(text_filepath):
    text_dict = OrderedDict()
    f = open(text_filepath)
    lines = f.readlines()
    print("lines: ", len(lines))
    for index, line in enumerate(lines):
        metas = line.strip().split("\t")
        if len(metas) == 2:
            text_dict[metas[0]] = metas[1]
        else:
            text_dict[f"{index:08}"] = metas[0]
    return text_dict


class SamiOfflineTokenizer:
    """SamiOfflineTokenizer expects phonemes to already be extracted from the text."""
    def __init__(self) -> None:
        pass

    def __call__(self, text_batch, line_break=" <n> ", **kwds):
        text_ids = []
        if isinstance(text_batch, str):
            text_batch = [text_batch]

        for sil in text_batch:
            lines = sil.split(line_break)
            text_tokens = []
            for phonemes in lines:
                phrase = Phrase.parse(phonemes=phonemes)
                if not any([phrase.phonemes, phrase.prefix_tags]):
                    continue
                labels = list(filter(lambda x: x != "", phrase.phonemes.split("\n"))) if phrase.phonemes else None
                text_id, _, _ = convert_labels_to_text_id(labels, phrase.prefix_tags)
                text_tokens.append(torch.from_numpy(text_id[0]).long())
            text_tokens = np.concatenate(text_tokens, axis=0)
            text_ids.append(torch.from_numpy(text_tokens).long())
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                text_ids, batch_first=True, padding_value=0
            ),
            "length": torch.tensor([len(t) for t in text_ids]).long()
        }


class SamiTokenizer(SamiOfflineTokenizer):
    def __init__(
        self,
        lib_path="/opt/tiger/sami_engine_cleaned/libs/libsami.so",
        fe="/opt/tiger/sami_tts_api/models/tts_chinese_frontend_model__42.0.model",
        fe_task="tts_chinese_frontend_model",
    ) -> None:
        super().__init__()
        self.cfg = generate_tts_config()
        self.engine = TtsEngine(lib_path=lib_path, fe=fe)
        self.ex = self.engine.create_fe_executor(task_type=fe_task)

    def __call__(self, text_batch, line_break=" <n> ", **kwds):
        if isinstance(text_batch, str):
            text_batch = [text_batch]
        g2p_result = []
        for text in text_batch:
            lines = text.split(line_break)
            with contextlib.redirect_stdout(None):
                phrases = [Phrase.parse(text=line) for line in lines]
                phrases = [
                    phrase._replace(phonemes=(self.ex.run(phrase.text, config=self.cfg)[0] if phrase.text else None)) 
                    for phrase in phrases
                ]
                ls = line_break.join([phrase.format_phonemes() for phrase in phrases])
            g2p_result.append(ls)
        return super().__call__(g2p_result, line_break, **kwds)