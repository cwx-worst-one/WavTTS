import logging
import re
from typing import Optional

from ..tokenizers.sami_phoneme_tokenizer import PhnStrParser

logger = logging.getLogger(__file__)


def convert_phonemes(phone: str, lang: str):
    """
    sil     0       S       0       O               S
    C0m     4       B       0       O       蔓家    B
    C0an    4       B       0       O       蔓家    E
    C0j     1       E       1       O               B
    C0ia    1       E       1       O               E
    C0uan   6       B       0       O       挽手    S
    C0sh    3       E       0       O               B
    C0ou    3       E       0       O               E
    C0l     2       S       4       O       喽      B
    C0ou    2       S       4       O       喽      E
    。      0       S       4       O               S


    sil     0       S       0       O               S
    C1m  19       B  0       O       蔓家    B
    C1an 19       B  0       O       蔓家    E
    C1j  16       E  1       O               B
    C1ia 16       E  1       O               E
    C1uan        21       B  0       O       挽手    S
    C1sh 18       E  0       O               B
    C1ou 18       E  0       O               E
    C1l  17       S  4       O       喽      B
    C1ou 17       S  4       O       喽      E
    。      0       S       4       O               S
    """
    if isinstance(lang, str):
        lang = lang.split(",")
    if "Cantonese" in lang:
        new_phone_list = []
        phone_list = phone.split("\n")
        for item in phone_list:
            item = PhnStrParser.parse(item)
            if item.phn == "C0":
                item.phn = "C1"
            new_phone_list.append(item.format())
        return "\n".join(phone_list)
    return phone


# NOTE: **Always update the regex pattern whenver the tags are updated**
# Pattern: "<singer_tag>:<text>"
# The pattern should match all the singer tags in the dataset
# What we select eventually is a subset `singer_tags`.
# Filtering should be done after matching.
SINGER_PATTERN = r"^(singer\d+|男|女|合)?:\s*((?:.|\n)*)"
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
        return "bridge"
    if norm_tag.find("主歌") != -1 or norm_tag.find("verse") != -1:
        return "verse"
    if norm_tag.find("副歌") != -1 or norm_tag.find("chorus") != -1:
        return "chorus"
    return (
        norm_tag
        if norm_tag in ["intro", "outro", "inst", "verse", "chorus", "bridge"]
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
