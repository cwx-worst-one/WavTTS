"""
Tools for genius format lyrics conversion, filtering, and processing
"""

import random
import re
from collections import Counter
from typing import Optional, Union

# -----------------------------------
# Convert lyrics to genius format
# -----------------------------------


def augment_lyrics_to_genius_style(
    genre: Union[str, list[str]], lyrics: str, rnd=random
) -> str:
    if isinstance(genre, str):
        genre = [genre]

    if "Hip Hop/Rap" in genre:
        return _process_genre_hiphop(lyrics)
    elif "Electronic" in genre:
        return _process_genre_electronic(lyrics)
    elif "Pop" in genre:
        return _process_genre_pop(lyrics, rnd)

    return _process_genre_default(lyrics)


def _sublist(ls1, ls2):
    """
    >>> sublist([], [1,2,3])
    True
    >>> sublist([1,2,3,4], [2,5,3])
    True
    >>> sublist([1,2,3,4], [0,3,2])
    False
    >>> sublist([1,2,3,4], [1,2,5,6,7,8,5,76,4,3])
    False
    """

    def get_all_in(one, another):
        for element in one:
            if element in another:
                yield element

    for x1, x2 in zip(get_all_in(ls1, ls2), get_all_in(ls2, ls1)):
        if x1 != x2:
            return False

    return True


_section_tag_pattern = r"\[([^\]]+)\]"


def _find_all_section_tags(lyrics: str) -> list[str]:
    """
    Find all section tags in the lyrics.

    Args:
        lyrics (str): The input lyrics.

    Returns:
        list[str]: A list of section tags found in the lyrics.
    """
    return re.findall(_section_tag_pattern, lyrics)


def _process_tag_inst(lyrics: str) -> str:
    def get_random_inst_replacement():
        return random.choice(
            ["instrumental", "interlude", "instrumental break", "guitar solo"]
        )

    new_lines = []
    for line in lyrics.split("\n"):
        if line == "[inst]":
            new_lines.append(f"[{get_random_inst_replacement()}]")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def _process_tag_prechorus(lyrics: str) -> str:
    """Make the tag pre-chorus if it is supposed to be pre-chorus in the first place"""
    section_tags = _find_all_section_tags(lyrics)
    n_bridges = Counter(section_tags)["bridge"]
    if n_bridges <= 1:
        return lyrics

    new_lines = []
    n_bridges_seen = 0
    for line in lyrics.split("\n"):
        if line == "[bridge]":
            if n_bridges_seen < n_bridges - 1:
                new_lines.append("[pre-chorus]")
            else:
                new_lines.append(line)
            n_bridges_seen += 1
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def _process_tag_refrain(lyrics: str, rnd=random) -> str:
    """Randomly add a refrain section after verse"""

    def generate_refrain(line: str) -> list[str]:
        if rnd.random() < 0.5:
            return []
        return ["[refrain]"] + rnd.choice(
            [[line], [line] * 2, ["oh oh oh oh oh oh"], ["oh oh oh oh oh oh"] * 2]
        )

    new_lines = []
    last_bracket_tag = None
    last_lyric_line = None
    for line in lyrics.split("\n"):
        if _find_all_section_tags(line):
            if (
                line != last_bracket_tag and last_bracket_tag == "[verse]"
            ):  # last one was verse
                refrain_lines = generate_refrain(last_lyric_line)
                new_lines.extend(refrain_lines)
            new_lines.append(line)
            last_bracket_tag = line
        else:
            new_lines.append(line)
            last_lyric_line = line

    return "\n".join(new_lines)


def _process_tag_post_chorus(lyrics: str, rnd=random) -> str:
    """Randomly add post-chorus after chorus"""

    def generate_post_chorus() -> list[str]:
        return ["[post-chorus]"] + rnd.choice(
            [["oh oh oh oh oh oh"] * 2, ["oh oh oh oh oh oh"] * 4]
        )

    if "[chorus]" not in lyrics:
        return lyrics

    if rnd.random() < 0.7:
        return lyrics

    new_lines = []
    last_bracket_tag = None
    post_chorus_lines = generate_post_chorus()
    for line in lyrics.split("\n"):
        if _find_all_section_tags(line):
            if (
                line != last_bracket_tag and last_bracket_tag == "[chorus]"
            ):  # last one was chorus
                new_lines.extend(post_chorus_lines)
            new_lines.append(line)
            last_bracket_tag = line
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def _process_tags_common(lyrics: str) -> str:
    lyrics = _process_tag_prechorus(lyrics)
    lyrics = _process_tag_inst(lyrics)
    return lyrics


def _process_genre_hiphop(lyrics: str) -> str:
    lyrics = lyrics.replace("[chorus]", "[hook]")
    return _process_tags_common(lyrics)


def _process_genre_electronic(lyrics: str) -> str:
    section_tags = _find_all_section_tags(lyrics)
    if _sublist(["chorus", "inst"], section_tags):
        lyrics = lyrics.replace("[chorus]", "[build]")
        lyrics = lyrics.replace("[inst]", "[drop]")
    return _process_tags_common(lyrics)


def _process_genre_pop(lyrics: str, rnd=random) -> str:
    lyrics = _process_tags_common(lyrics)
    lyrics = _process_tag_refrain(lyrics, rnd)
    lyrics = _process_tag_post_chorus(lyrics, rnd)
    return lyrics


def _process_genre_default(lyrics: str) -> str:
    return _process_tags_common(lyrics)


# -----------------------------------------------
# Genius Lyrics Filtering and Processing
# -----------------------------------------------

# Filter out the lyrics if they can't be processed
# Keep and process lyrics that can be processed


def clean_genius_lyrics(
    lyrics: str, pre_filter: bool = True, translate_bracket_tag: bool = True
) -> Optional[str]:
    """
    Process the lyrics by translating the bracket tags.
    Args:
        lyrics (str): The lyrics to process.
        pre_filter (bool): Whether to pre-filter the lyrics. Defaults to True.
        translate_bracket_tag (bool): Whether to translate the bracket tags. Defaults to True.
            Bracket tags must be normalized before translation.
    Returns:
        Optional[str]: The processed lyrics, or None if the lyrics are invalid.
    """
    if pre_filter and not _is_genius_lyrics_valid(lyrics):
        return None

    if translate_bracket_tag:
        return _translate_genius_bracket_tags(lyrics)

    return lyrics


def _is_bracket_tag_in_lyrics(lyrics: str) -> bool:
    return bool(_find_all_section_tags(lyrics))


def _has_too_many_words_in_bracket(text: str) -> bool:
    """
    Find all contents inside brackets and check if any contain more than 2 words.
    Returns False if any bracket content has > 2 words, else True.
    """
    # Find all contents inside brackets
    contents = _find_all_section_tags(text)

    for content in contents:
        words = content.strip().split()
        if len(words) > 2:
            return True
    return False


def _is_genius_lyrics_valid(lyrics: str) -> bool:
    # NOTE(Yilin): This is from @yixiao's UDF impl, quite aggressive

    # 歌词长度检查
    if len(lyrics) < 100:
        return False

    # 匹配所有带xN的tag
    x_n_pattern = re.compile(r"(\[x\d+\]|\(x\d+\))")
    if x_n_pattern.search(lyrics):
        return False

    # 使用普通字符串匹配其他固定模式
    if (
        ("[?]" in lyrics)
        or ("[...]" in lyrics)
        or ("[x]" in lyrics)
        or ("作词" in lyrics)
        or ("作曲" in lyrics)
    ):
        return False

    if _has_too_many_words_in_bracket(lyrics):
        return False

    # 检查是否包含有效的分段标签
    return _is_bracket_tag_in_lyrics(lyrics)


def _translate_genius_bracket_tags(lyrics: str) -> str:
    """
    Replace the section tags in the lyrics.
    """

    def _replacer(match):
        tag = match.group(1)  # content inside [ ]
        return f"[{SECTION_TAG_MAP.get(tag.lower(), tag)}]"

    return re.sub(_section_tag_pattern, _replacer, lyrics)


SECTION_TAG_MAP_MULTILANG = {
    "english": {
        "verse": ["strophe"],
        "pre-chorus": ["prechorus", "pre chorus"],
        "post-chorus": ["postchorus", "post chorus"],
    },
    "russian": {
        "intro": ["интро", "вступление"],
        "verse": ["куплет"],
        "chorus": ["припев"],
        "pre-chorus": ["предприпев"],
        "post-chorus": ["постприпев"],
        "refrain": ["рефрен"],
        "hook": ["ху́к"],
        "bridge": ["бридж"],
        "interlude": ["интерлюдия"],
        "solo": ["соло"],
        "build": ["нарастание"],
        "drop": ["дроп"],
        "outro": ["аутро", "завершение"],
    },
    "portuguese": {
        "intro": ["introdução"],
        "verse": ["verso", "estrofe"],
        "chorus": ["refrão"],
        "pre-chorus": ["pré-refrão"],
        "post-chorus": ["pós-refrão"],
        "refrain": ["refrão"],
        "hook": ["gancho"],
        "bridge": ["ponte"],
        "interlude": ["interlúdio"],
        "solo": ["solo"],  # widely used as-is
        "build": ["subida", "crescendo"],
        "drop": ["drop"],  # borrowed as-is
        "outro": ["saída", "final"],
    },
    "spanish": {
        "intro": ["introducción"],
        "verse": ["verso", "estrofa"],
        "chorus": ["estribillo", "coro", "refrán"],
        "pre-chorus": ["pre-coro", "pre-estribillo"],
        "post-chorus": ["post-coro", "post-estribillo"],
        "refrain": ["estribillo", "refrán"],
        "hook": ["gancho"],
        "bridge": ["puente"],
        "interlude": ["interludio"],
        "solo": ["solo"],
        "build": ["subida", "crescendo"],
        "drop": ["drop"],
        "outro": ["coda", "final", "outro"],
    },
    "italian": {
        "intro": ["introduzione"],
        "verse": ["strofa"],
        "chorus": ["ritornello", "coro"],
        "pre-chorus": ["pre-ritornello"],
        "post-chorus": ["post-ritornello"],
        "refrain": ["ritornello"],
        "hook": ["gancio"],
        "bridge": ["ponte"],
        "interlude": ["interludio"],
        "solo": ["assolo"],
        "build": ["crescendo"],
        "drop": ["drop"],
        "outro": ["coda", "finale", "outro"],
    },
    "polish": {
        "intro": ["wstęp"],
        "verse": ["zwrotka"],
        "chorus": ["refren"],
        "pre-chorus": ["przed-refren"],
        "post-chorus": ["po-refren"],
        "refrain": ["refren"],
        "hook": ["haczyk"],
        "bridge": ["most"],
        "interlude": ["interludium"],
        "solo": ["solo"],
        "build": ["narastanie"],
        "drop": ["drop"],
        "outro": ["zakończenie", "outro"],
    },
    "turkish": {
        "intro": ["giriş"],
        "verse": ["kıta"],
        "chorus": ["nakarat"],
        "pre-chorus": ["ön-nakarat"],
        "post-chorus": ["son-nakarat"],
        "refrain": ["nakarat"],
        "hook": ["nakarat bölümü"],
        "bridge": ["köprü"],
        "interlude": ["ara"],
        "solo": ["solo"],
        "build": ["yükseliş"],
        "drop": ["drop"],
        "outro": ["bitiş", "outro"],
        "part": ["bölüm"],
    },
    "french": {
        "intro": ["introduction"],
        "verse": ["couplet"],
        "chorus": ["refrain"],
        "pre-chorus": ["pré-refrain"],
        "post-chorus": ["post-refrain"],
        "refrain": ["refrain"],
        "hook": ["accroche"],
        "bridge": ["pont"],
        "interlude": ["interlude"],
        "solo": ["solo"],
        "build": ["montée"],
        "drop": ["drop"],
        "outro": ["final", "outro"],
    },
    "arabic": {
        "intro": ["مقدمة"],
        "verse": ["مقطع", "المقطع الأول", "المقطع الثاني"],
        "chorus": ["اللازمة", "كورَس"],
        "pre-chorus": ["قبل اللازمة"],
        "post-chorus": ["بعد اللازمة"],
        "refrain": ["لازمة"],
        "hook": ["هوك", "جاذب"],
        "bridge": ["الجسر"],
        "interlude": ["فاصل"],
        "solo": ["صولو"],
        "build": ["تصاعد"],
        "drop": ["دروب"],
        "outro": ["خاتمة"],
    },
    "hebrew": {
        "intro": ["פתיחה"],
        "verse": ["בית"],
        "chorus": ["פזמון"],
        "pre-chorus": ["קדם-פזמון"],
        "post-chorus": ["אחרי-פזמון"],
        "refrain": ["פזמון חוזר"],
        "hook": ["הוק", "קרס"],
        "bridge": ["גשר"],
        "interlude": ["קטע ביניים"],
        "solo": ["סולו"],
        "build": ["בנייה"],
        "drop": ["דרופ"],
        "outro": ["סיום"],
    },
    "swedish": {
        "intro": ["intro"],
        "verse": ["vers"],
        "chorus": ["refräng"],
        "pre-chorus": ["för-refräng"],
        "post-chorus": ["efter-refräng"],
        "refrain": ["refräng"],
        "hook": ["hook"],
        "bridge": ["brygga"],
        "interlude": ["mellanspel"],
        "solo": ["solo"],
        "build": ["uppbyggnad"],
        "drop": ["drop"],
        "outro": ["avslutning", "outro"],
    },
    "serbo-croatian": {
        "intro": ["uvod"],
        "verse": ["strofa"],
        "chorus": ["refren"],
        "pre-chorus": ["pred-refren"],
        "post-chorus": ["posle-refren"],
        "refrain": ["refren"],
        "hook": [],
        "bridge": ["most"],
        "interlude": ["međudel"],
        "solo": ["solo"],
        "build": ["gradacija"],
        "drop": ["drop"],
        "outro": ["završetak", "outro"],
    },
    "azerbaijani": {
        "intro": ["giriş"],
        "verse": ["bənd"],
        "chorus": ["nəqərat"],
        "pre-chorus": ["ön-nəqərat"],
        "post-chorus": ["sonrakı-nəqərat"],
        "refrain": ["nəqərat"],
        "hook": [],
        "bridge": ["körpü"],
        "interlude": ["ara musiqisi"],
        "solo": ["solo"],
        "build": ["yüksəliş"],
        "drop": ["drop"],
        "outro": ["sonluq", "outro"],
    },
    "romanian": {
        "intro": ["introducere"],
        "verse": ["strofa"],
        "chorus": ["refren"],
        "pre-chorus": ["pre-refren"],
        "post-chorus": ["post-refren"],
        "refrain": ["refren"],
        "hook": [],
        "bridge": ["punte"],
        "interlude": ["interludiu"],
        "solo": ["solo"],
        "build": ["creștere"],
        "drop": ["drop"],
        "outro": ["încheiere", "outro"],
    },
    "chinese": {
        "intro": ["前奏", "引子", "序曲"],
        "verse": ["主歌"],
        "chorus": ["副歌", "合唱"],
        "pre-chorus": ["前副歌"],
        "post-chorus": ["后副歌"],
        "refrain": ["叠句"],
        "hook": ["洗脑段"],
        "bridge": ["桥段"],
        "interlude": ["间奏"],
        "solo": ["独奏"],
        "build": ["渐强", "铺垫"],
        "drop": ["掉落", "爆点"],
        "outro": ["尾奏", "结束"],
    },
    "danish": {
        "intro": ["intro"],
        "verse": ["vers"],
        "chorus": ["omkvæd"],
        "pre-chorus": ["før-omkvæd"],
        "post-chorus": ["efter-omkvæd"],
        "refrain": ["refræn"],
        "hook": [],
        "bridge": ["bro"],
        "interlude": ["mellemspil"],
        "solo": ["solo"],
        "build": ["opbygning"],
        "drop": ["drop"],
        "outro": ["afslutning", "outro"],
    },
}


def _build_reversed_map() -> dict[str, str]:
    m = {}
    for d in SECTION_TAG_MAP_MULTILANG.values():
        for english_tag, lang_tags in d.items():
            for lang_tag in lang_tags:
                m[lang_tag] = english_tag
    return m


SECTION_TAG_MAP = _build_reversed_map()
