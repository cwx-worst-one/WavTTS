import copy
import json
import operator
import random
import re
from dataclasses import dataclass, field
from functools import reduce
from typing import Optional

from ..utils.dcbase import DCBase

__all__ = ["parse_freeform_text", "format_freeform_text"]


def parse_freeform_text(meta: dict, style_tags: dict) -> dict:
    """
    keyword_dropout: dropout rate for keywords
    """
    freeform_text = _infer_freeform_text_from_meta(meta)
    if not freeform_text.is_empty():
        return freeform_text.to_dict()
    return _infer_freeform_text_from_style_text(style_tags).to_dict()


def format_freeform_text(freeform_text: dict, keyword_dropout_rate: float = 0.2) -> str:
    def split_text(text) -> list:
        """Splits the input text by special characters and spaces."""
        return re.findall(r"\w+", text)

    def is_tag_valid(tag: str) -> bool:
        subwords = split_text(tag)
        if not subwords:
            return False
        return all(word.lower() not in _BLOCKED_KEYWORDS for word in subwords)

    freeform_text = FreeformText.from_dict(freeform_text)
    if freeform_text.descriptions:
        return random.choice(freeform_text.descriptions)

    # filter, dropout, shuffle
    keywords = freeform_text.keywords
    keywords = [kw for kw in keywords if is_tag_valid(kw)]
    keywords = [kw for kw in keywords if random.random() < (1.0 - keyword_dropout_rate)]
    keywords = list(dict.fromkeys(keywords))  # remove duplicates
    if not keywords:
        return ""
    random.shuffle(keywords)
    return ", ".join(keywords)


@dataclass
class FreeformText(DCBase):
    keywords: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.keywords and not self.descriptions


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


def _infer_freeform_text_from_style_text(style_tags: dict) -> FreeformText:
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
    return FreeformText(keywords=keywords, descriptions=[])


def _infer_freeform_text_from_meta(meta: dict) -> FreeformText:
    # YILIN NOTE: This is an extremely simplified version. No remapping whatsoever.
    fns = [
        _extract_keywords_apm,
        _extract_keywords_everynoise,
        _extract_keywords_rym,
        _extract_keywords_sstk,
        _extract_keywords_wyy,
        _extract_keywords_llm,
    ]
    results = [fn(meta) for fn in fns]
    keywords = reduce(operator.add, [r.get("keywords", []) for r in results])
    descriptions = [desc for desc in [r.get("description") for r in results] if desc]
    return FreeformText(keywords=keywords, descriptions=descriptions)


# YILIN NOTE: The functions below are copied from zh_meta and refactored


def _json_loads_as_list(json_str: str) -> list[str]:
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError:
        return []


def _is_platform_blocked(meta: dict) -> bool:
    return meta.get("raw", {}).get("platform") in ["dq", "wyy", "mcc"]


def _is_library_blocked(meta: dict) -> bool:
    library = meta.get("library")
    return library not in ["pond5", "shutterstock"]


def _are_platform_and_library_blocked(meta: dict) -> bool:
    return _is_platform_blocked(meta) or _is_library_blocked(meta)


def _extract_keywords_sstk(meta: dict) -> dict:
    if _is_platform_blocked(meta):
        return {}

    instruments = meta.get("instruments", "")
    if not instruments:
        instruments = meta.get("raw", {}).get("instruments", "")
    instruments = instruments.split(", ") if instruments else []
    instruments = [x for x in instruments if x != "\\N"]
    keywords = meta.get("keywords", "")
    if not keywords:
        keywords = meta.get("raw", {}).get("keywords", "")
    keywords = keywords.split(", ") if keywords else []
    all_tags = [tag for tag in instruments + keywords if tag]
    keywords = list(dict.fromkeys(all_tags))

    description = meta.get("description")
    # YILIN NOTE: Get rid of tempo for simplification.
    # tempo = meta.get("bpm")
    # try:
    #     tempo = float(tempo)
    # except (ValueError, TypeError):
    #     tempo = None

    return {"keywords": keywords, "description": description}


def _extract_keywords_everynoise(meta: dict) -> dict:
    """meta.raw.genres, meta.everynoise_genre, meta.everynoise_trending.vantage/genre"""
    if _are_platform_and_library_blocked(meta):
        return {}
    raw_genres = meta.get("raw", {}).get("genres", [])
    everynoise_genre = meta.get("everynoise_genre", "")
    everynoise_trending_vantage = meta.get("everynoise_trending", {}).get(
        "vantage", ""
    )  # sometimes 'everynoise_trending' exists but is None
    everynoise_trending_genre = meta.get("everynoise_trending", {}).get("genre", "")
    all_tags = [
        tag
        for tag in [
            raw_genres,
            everynoise_genre,
            everynoise_trending_genre,
            everynoise_trending_vantage,
        ]
        if tag
    ]
    flat_list = [
        tag
        for sublist in all_tags
        for tag in (sublist if isinstance(sublist, list) else [sublist])
    ]
    return {"keywords": flat_list}


def _extract_keywords_wyy(meta: dict) -> dict:
    if _are_platform_and_library_blocked(meta):
        return {}
    tags = _json_loads_as_list(
        meta.get("raw", {}).get("tags", "[]")
    )  # YILIN NOTE: the original code uses ast.literal_eval, which is not safe
    category = meta.get("raw", {}).get("category", "")
    song_tag = meta.get("raw", {}).get("song_tag", "")
    song_biz_tag = meta.get("raw", {}).get("song_biz_tag", "")
    song_tag = song_tag.split("-", 1) if song_tag else []
    song_biz_tag = song_biz_tag.split(",") if song_biz_tag else []
    category = category.split(",") if category else []
    return {"keywords": reduce(operator.add, [tags, category, song_tag, song_biz_tag])}


def _extract_keywords_apm(meta: dict) -> dict:
    if _are_platform_and_library_blocked(meta):
        return {}
    raw = meta.get("raw", {})
    facets_list = _json_loads_as_list(raw.get("facets_list", "[]"))
    keywords = [
        item
        for d in facets_list
        for value in d.values()
        for item in value
        if item is not None
    ]
    term = raw.get("term")
    if term:
        keywords.extend(term.split(","))
    return {
        "keywords": keywords,  # YILIN NOTE: There was a list of blocked keywords, removed.
        "description": raw.get("description"),
    }


def _extract_keywords_rym(meta: dict) -> dict:
    if _are_platform_and_library_blocked(meta):
        return {}
    track_genres = meta.get("raw", {}).get("track_genres", [])
    album_genres = meta.get("raw", {}).get("album_genres", [])
    album_genres = (
        reduce(operator.add, album_genres) if album_genres else []
    )  # flatten the list
    album_descriptors = meta.get("raw", {}).get("album_descriptors", [])
    return {
        "keywords": reduce(
            operator.add, [track_genres, album_genres, album_descriptors]
        )
    }


def _extract_keywords_llm(meta: dict) -> dict:
    if _are_platform_and_library_blocked(meta):
        return {}
    freeform_list = []
    for x in meta.get("music_info_by_llm", {}).get("tags", []):
        x = x.strip()
        if x:
            freeform_list.append(x)
    for x in meta.get("music_info_by_llm", {}).get("genres", []):
        x = x.strip()
        if x:
            freeform_list.append(x)
    return {"keywords": freeform_list}
