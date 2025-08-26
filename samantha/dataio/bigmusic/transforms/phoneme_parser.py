import json
import logging
import re
import string
from pathlib import Path
from typing import Optional

from ToJyutping import ToJyutping  # 3.2.0

from ..tokenizers.sami_phoneme_tokenizer import PhnStrParser

logger = logging.getLogger(__file__)
_VOCAB_DIR = (
    Path(__file__).parent.absolute() / "../tokenizers/sami_phoneme_tokenizer/vocabs"
)


def is_punc(ch):
    return ch in string.punctuation


def is_chinese_char(ch):
    if is_punc(ch):
        return True
    return (ch.encode("unicode_escape") >= b"\\u4e00") and (
        ch.encode("unicode_escape") <= b"\\u9fff"
    )


def is_english_char(ch):
    if is_punc(ch):
        return True
    return ch.encode("utf-8").isalpha()


def get_vocab_from_version(version: str = "v2"):
    vocab = {}
    version_file = _VOCAB_DIR / f"builder.{version}.json"
    with open(version_file, "r") as f:
        vocab = json.load(f)
    return vocab


def convert_phonemes(text, phoneme, lang, version="v2"):
    try:
        import ToJyutping  # 3.2.0
    except ImportError:
        raise ImportError(
            "Failed to import ToJyutping, please install it for `convert_phonemes`."
        )

    vocab = get_vocab_from_version(version)
    CANTO_CONSONANTS, CANTO_VOWELS = [], []
    for item in vocab.get("phn_vocabs", []):
        if item.get("language") == "cant":
            CANTO_CONSONANTS = item.get("consonants", [])
            CANTO_VOWELS = item.get("vowels", [])

    if isinstance(lang, str):
        lang = lang.split(",")
    if "Cantonese" not in lang:
        return phoneme
    phoneme_list = phoneme.split("\n")
    merged_phoneme_list = []
    last_prefix = None
    for ph in phoneme_list:
        x_split = ph.split("\t")
        if len(x_split) == 7:
            phone, tone, _, _, _, word, _ = x_split
        elif len(x_split) == 6:
            phone, tone, _, _, _, word = x_split
        else:
            raise ValueError(f"Wrong tacolab {x_split}")
        now_prefix = phone[:2] if phone not in ["sil", "。"] else phone
        if now_prefix not in ["C0", "E0", "JP", "sil", "。"]:
            continue
        is_zh = all([is_chinese_char(ch) for ch in word])
        is_en = all([is_english_char(ch) for ch in word])
        if not word:
            if now_prefix not in ["sil", "。"]:
                now_prefix = last_prefix
        elif is_zh:
            now_prefix = "C0"
        elif is_en:
            now_prefix = "E0"
        if not last_prefix:
            if now_prefix == "C0":
                merged_phoneme_list.append(
                    {"prefix": now_prefix, "phoneme": "", "word": word}
                )
            else:
                merged_phoneme_list.append(
                    {"prefix": now_prefix, "phoneme": ph, "word": word}
                )
        elif last_prefix == now_prefix:
            if now_prefix == "C0":
                merged_phoneme_list[-1]["word"] += word
            elif now_prefix == "E0":
                merged_phoneme_list[-1]["phoneme"] += "\n" + ph
                merged_phoneme_list[-1]["word"] += " " + word
            else:
                merged_phoneme_list[-1]["phoneme"] += "\n" + ph
                merged_phoneme_list[-1]["word"] += word
            merged_phoneme_list[-1]["phoneme"] = merged_phoneme_list[-1][
                "phoneme"
            ].strip()
            merged_phoneme_list[-1]["word"] = merged_phoneme_list[-1]["word"].strip()
        else:
            merged_phoneme_list.append(
                {"prefix": now_prefix, "phoneme": ph, "word": word}
            )
        last_prefix = now_prefix

    converted_phoneme_list = []
    converted_phoneme_list.append({"prefix": "sil", "phoneme": "sil\t0\tS\t0\tO\t\tS"})
    idx = 0
    pinyin = []
    for ch, ph in ToJyutping.get_jyutping_list(text):
        if ph:
            for sub_ph in ph.split(" "):
                pinyin.append((ch, sub_ph))
        else:
            pinyin.append((ch, ph))
    for ch, ph in pinyin:
        if (not ch.strip()) and (not ph):
            continue
        _ph = ""
        now_prefix = ""
        if ph:
            tone = ph[-1]  # 1~6
            if "C1" + ph[0] in CANTO_CONSONANTS and "C1" + ph[1:-1] in CANTO_VOWELS:
                consonant = "C1" + ph[0]
                vowel = "C1" + ph[1:-1]
            elif "C1" + ph[:2] in CANTO_CONSONANTS and "C1" + ph[2:-1] in CANTO_VOWELS:
                consonant = "C1" + ph[:2]
                vowel = "C1" + ph[2:-1]
            elif "C1" + ph[:-1] in CANTO_CONSONANTS or "C1" + ph[:-1] in CANTO_VOWELS:
                consonant = "C1" + ph[:-1]
                vowel = ""
            else:
                consonant = ""
                vowel = ""
                tone = ""

            if consonant:
                _ph = "\t".join([consonant, tone, "B", "0", "O", ch, "B"])
                now_prefix = "C1"
            if vowel:
                _ph += "\n" + "\t".join([vowel, tone, "B", "0", "O", ch, "E"])
                now_prefix = "C1"
        if idx == 0:
            converted_phoneme_list.append({"prefix": now_prefix, "phoneme": _ph})
        else:
            if now_prefix == converted_phoneme_list[-1]["prefix"]:
                converted_phoneme_list[-1]["phoneme"] += "\n" + _ph
                converted_phoneme_list[-1]["phoneme"] = converted_phoneme_list[-1][
                    "phoneme"
                ].strip()
            else:
                converted_phoneme_list.append({"prefix": now_prefix, "phoneme": _ph})
        idx += 1
    converted_phoneme_list.append(
        {"prefix": "。", "phoneme": "。\t0\tS\t4\tE_DECL\t\tS"}
    )
    if len(merged_phoneme_list) != len(converted_phoneme_list):
        return phoneme
    converted_phoneme = ""
    for idx in range(len(merged_phoneme_list)):
        item1 = merged_phoneme_list[idx]
        item2 = converted_phoneme_list[idx]
        if item1["prefix"] in ["sil", "。"]:
            converted_phoneme += item1["phoneme"] + "\n"
        elif item1["prefix"] == "C0":
            converted_phoneme += item2["phoneme"] + "\n"
        else:
            converted_phoneme += item1["phoneme"] + "\n"
    return converted_phoneme.strip()


# NOTE: **Always update the regex pattern whenver the tags are updated**
# Pattern: "<singer_tag>:<text>"
# The pattern should match all the singer tags in the dataset
# What we select eventually is a subset `singer_tags`.
# Filtering should be done after matching.
SINGER_PATTERN = r"^(singer\d+|男|女|合|中|童|念|多|伴|对)?:\s*((?:.|\n)*)"
SECTION_PARENS = ["[]", "【】", "()", "（）", "<>", "《》", "{}", "「」"]
SECTION_PATTERNS = [
    re.compile(f"^\\{paren[0]}(.*?)\\{paren[1]}\\s*((?:.|\n)*)")
    for paren in SECTION_PARENS
]
STANDARD_SECTION_PATTERN = SECTION_PATTERNS[0]

# ------------------------------------------
# Special tag utils
# ------------------------------------------


def _extract_prefix_tag(text: str, pattern: str) -> tuple[Optional[str], str]:
    match = re.search(pattern, text)
    if match is None:
        return None, text
    tag, rest_text = match.group(1), match.group(2)
    if tag is not None:
        tag = tag.strip()
    if rest_text is None:
        raise ValueError("Invalid text or regex pattern.")
    return tag, rest_text


def extract_singer_tag(text: str) -> tuple[Optional[str], str]:
    """Extract the singer tag from the input text.
    :param text: A string that might or might not contain a leading singer tag, followed by a colon.
    :return: The singer tag (None if empty) and the rest of the text without colon and leading whitespace.
    """
    return _extract_prefix_tag(text, SINGER_PATTERN)


def add_singer_tag(singer_tag: Optional[str], text: str) -> str:
    if singer_tag:
        return f"{singer_tag}:{text}"
    return text


def norm_section_tag(section_tag: str) -> Optional[str]:
    # Tag validation happens in Phrase.
    norm_tag = section_tag.lower().strip()
    # Map pre-chorus to bridge
    if norm_tag in ["pre_chorus", "prechorus", "pre-chorus"]:
        #    return "bridge"
        return "pre-chorus"
    if norm_tag.find("主歌") != -1 or norm_tag.find("verse") != -1:
        return "verse"
    if norm_tag.find("副歌") != -1 or norm_tag.find("chorus") != -1:
        return "chorus"
    return (
        norm_tag
        if norm_tag
        in ["intro", "outro", "inst", "verse", "chorus", "bridge", "pre-chorus"]
        else None
    )


def extract_section_tag(
    text: str, normalize_tag: bool = False
) -> tuple[Optional[str], str]:
    """Extract the section tag from the input text.
    :param text: A string that might or might not contain a leading section tag.
    :param normalize_tag:
        True:
            1. Other parentheses in `section_parens` are also allowed.
            2. Tag is normalized to lower case. If the normalized tag is not allowed, the tag is discarded.
        False: The section name must be enclosed by "[]".
    :return: The section tag (None if empty) and the rest of the text without colon and leading whitespace.
    """
    if not normalize_tag:
        return _extract_prefix_tag(text, STANDARD_SECTION_PATTERN)
    for _section_pattern in SECTION_PATTERNS:
        tag, rest_text = _extract_prefix_tag(text, _section_pattern)
        if tag is not None:
            section_tag = norm_section_tag(tag)
            # if it is not norm section tag, keep prefix_tag as lyrics
            return section_tag, rest_text if section_tag else tag + rest_text
    return None, text


def add_section_tag(section_tag: Optional[str], text: str) -> str:
    if section_tag:
        sep = " " if text else ""
        return f"[{section_tag}]{sep}{text}"
    return text


def norm_instrument_tags(instrument_tags_str: str) -> Optional[str]:
    instrument_tags = instrument_tags_str.split(",")
    instrument_tags = [x.strip() for x in instrument_tags]
    # YILIN NOTE: Temporarily remove the out-of-vocab filtering
    # instrument_tags = [x for x in instrument_tags if x in TAG_VOCAB_INSTRUMENT]

    return instrument_tags if instrument_tags else None


def extract_instrument_tags(text: str) -> tuple[Optional[list[str]], str]:
    """Extract the instrument tags from the input text.
    :param text: A string that might or might not contain instrument tags.
    :return: The instrument tags (None if empty) and the rest of the text without colon and leading whitespace.
    """
    match = re.search(r"\[(.*?)\]", text)
    if match:
        instruments = match.group(1)
        rest_text = re.sub(r"\[.*?\]", "", text, count=1).strip()
        instrument_tags = norm_instrument_tags(instruments)
        if instrument_tags:
            return instrument_tags, rest_text
        else:
            return None, instruments + rest_text
    return None, text


def add_instrument_tags(instrument_tags: Optional[list[str]], text: str) -> str:
    if instrument_tags:
        sep = " " if text else ""
        return f"[{','.join(instrument_tags)}]{sep}{text}"
    return text
