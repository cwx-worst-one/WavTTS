import copy
import json
import operator
import random
import re
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Optional, Tuple

from ..utils.dcbase import DCBase

__all__ = ["parse_freeform_text", "format_freeform_text"]

# NOTE: only use lower case letters
_BLOCKED_KEYWORDS = [
    "empty",
    "non",
    "no",
    "unkown",
    "nonvocal",
    "other",
    "adult",
    "sinking",
    "english",
    "chinese",
    "cantonese",
    "japanese",
    "sichuanese",
]


def parse_freeform_text(meta: dict, style_tags: dict) -> dict:
    """
    keyword_dropout: dropout rate for keywords
    """
    keywords, descriptions = _infer_freeform_text_from_meta(meta)
    style_tags = _infer_freeform_text_from_style_text(style_tags)
    return FreeformText(
        keywords=keywords, style_tags=style_tags, descriptions=descriptions
    ).to_dict()


def format_freeform_text(
    freeform_text: dict,
    long_description_rate: float,
    keyword_dropout_rate: float,
    style_tags_dropout_rate: float,
) -> str:
    def split_text(text) -> list:
        """Splits the input text by special characters and spaces."""
        return re.findall(r"\w+", text)

    def is_tag_valid(tag: str) -> bool:
        subwords = split_text(tag)
        if not subwords:
            return False
        return all(word.lower() not in _BLOCKED_KEYWORDS for word in subwords)

    freeform_text = FreeformText.from_dict(freeform_text)
    if random.random() < long_description_rate and freeform_text.descriptions:
        return random.choice(freeform_text.descriptions)

    keywords_all = []
    if random.random() < 1.0 - keyword_dropout_rate:
        keywords_all += freeform_text.keywords
    if random.random() < 1.0 - style_tags_dropout_rate:
        keywords_all += freeform_text.style_tags

    # filter, shuffle, trim
    keywords_all = [kw for kw in keywords_all if is_tag_valid(kw)]
    keywords_all = list(dict.fromkeys(keywords_all))
    random.shuffle(keywords_all)
    if random.random() < 0.5:  # default: 0.5 chance to trim keywords
        keywords_all = keywords_all[: random.randint(1, 10)]
    return ", ".join(keywords_all)


@dataclass
class FreeformText(DCBase):
    keywords: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.keywords and not self.descriptions


def _infer_freeform_text_from_style_text(style_tags: dict) -> List[str]:
    def take_first(lst: Optional[list[str]]) -> str:
        return lst[0] if lst else ""

    style_tags = copy.deepcopy(
        style_tags
    )  # make a copy just in case of any in-place modification

    # put all tags into a list, but concatenate key and mode as one tag
    keywords = reduce(
        operator.add, [v for k, v in style_tags.items() if k not in ["key", "mode"]]
    )
    keywords.append(
        take_first(style_tags.get("key")) + " " + take_first(style_tags.get("mode"))
    )
    return keywords


def _infer_freeform_text_from_meta(meta: dict) -> Tuple[List[str], List[str]]:

    standard_music_meta_nonbpe = meta.get("standard_music_meta_nonbpe", {})

    freeform_text_raw = standard_music_meta_nonbpe.get("freeform_text_raw", {})
    freeform_text_llm = standard_music_meta_nonbpe.get("freeform_text_llm", {})
    keywords = []
    keywords = freeform_text_raw.get("freeform_text_short", [])
    if not keywords:
        keywords = freeform_text_llm.get("freeform_text_short", [])
    descriptions = []
    descriptions = freeform_text_raw.get("freeform_text_long", [])
    if not descriptions:
        descriptions = freeform_text_llm.get("freeform_text_long", [])

    return keywords, descriptions
