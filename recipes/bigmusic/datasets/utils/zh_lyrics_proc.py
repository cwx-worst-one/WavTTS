"""
The processing approach is designed primarily for Chinese lyrics, assuming each word consists of
a single syllable. This approach simplifies the logic, and does not require third-party NLP packages.
However, it is not suitable for English, so certain processing steps, such as line splitting and
concatenation, are skipped when handling English lyrics. Code related to English-specific
processing is commented as "EN Patch: ...".
"""


import bisect
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from itertools import accumulate 
import logging
import math
import operator
import random
import string
from typing import List, Optional, Tuple
import unicodedata

import numpy as np

from recipes.datasets.mcc.sami_tokenizer import (
    Phrase,
    is_chinese_char,
    add_section_tag,
    section_parens as SECTION_PARENS,
)

logger = logging.getLogger(__file__)


VOCAL_SECTION_TAGS = ["chorus", "verse", "bridge"]


class ZhLyricError(Exception):
    pass


class Lang(Enum):
    ZH = auto()
    EN = auto()


@dataclass
class LyricsProcConfig:
    line_range: Tuple[int, int]
    slb_range: Tuple[int, int]

GLOBAL_MAX_N_LINES = 10000  # A very high value that ensures words will not be deleted. The frontend is responsible for the length constraint (200).
GENRE_CONFIGS = {
    "Pop": LyricsProcConfig(line_range=(5, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
    "Hip Hop/Rap": LyricsProcConfig(line_range=(8, GLOBAL_MAX_N_LINES), slb_range=(8, 15)),
    "Chinese Style": LyricsProcConfig(line_range=(6, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
    "Electronic": LyricsProcConfig(line_range=(5, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
    "DJ": LyricsProcConfig(line_range=(5, GLOBAL_MAX_N_LINES), slb_range=(4, 15)),
    "Rock": LyricsProcConfig(line_range=(8, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
    "Folk": LyricsProcConfig(line_range=(5, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
    "R&B/Soul": LyricsProcConfig(line_range=(5, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
    "MC": LyricsProcConfig(line_range=(10, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
    "empty": LyricsProcConfig(line_range=(5, GLOBAL_MAX_N_LINES), slb_range=(5, 15)),
}

def get_genre_config(genre: str) -> LyricsProcConfig:
    for k in GENRE_CONFIGS:
        if k in genre:
            return GENRE_CONFIGS[k]
    return GENRE_CONFIGS["empty"]


def _is_chinese_char(c: str) -> bool:
    return is_chinese_char(c.encode("unicode_escape"))

@dataclass
class Syllable:
    phonemes: List[str]


@dataclass
class Lyric:
    text: Optional[str] = None
    syllables: Optional[List[Syllable]] = None
    punc: Optional[str] = None

    def __post_init__(self):
        if (self.text is None and self.punc is None) or (self.text is not None and self.punc is not None):
            raise ZhLyricError("Invalid lyric.")

        if not self.text:
            self.text = None
        if not self.syllables:
            self.syllables = None
        if not self.punc:
            self.punc = None

        if self.text is not None:
            self.text = self.text.strip()
        if self.punc is not None:
            self.punc = self.punc.strip()

    def __str__(self) -> str:
        text = "" if self.text is None else self.text
        punc = "" if self.punc is None else self.punc
        return text + punc

    @property
    def n_syllables(self):
        if self.syllables:
            return len(self.syllables)
        if not self.has_utterance:
            return 0
        return 1

    @property
    def lang(self) -> Optional[Lang]:
        if not self.has_utterance:
            return None
        if any(_is_chinese_char(c) for c in self.text):
            return Lang.ZH
        return Lang.EN

    @property
    def has_utterance(self) -> bool:
        return self.text is not None


class Line(list):
    def __init__(self, lyrics: List[Lyric]):
        super().__init__(lyrics)

    def __getitem__(self, index_or_slice):
        lst = list.__getitem__(self, index_or_slice)
        if isinstance(index_or_slice, int):
            return lst
        return self.__class__(lst)

    def __add__(self, other):
        return self.__class__(list.__add__(self, other))

    def __mul__(self, other):
        return self.__class__(list.__mul__(self, other))

    def __str__(self) -> str:
        words = []
        for lyric, next_lyric in zip(self, self[1:] + [None]):
            words.append(str(lyric))
            if next_lyric is None:
                continue
            # append a space if they are in different langauges or both are English
            if lyric.lang != next_lyric.lang or lyric.lang == Lang.EN or next_lyric.lang is None:
                words.append(" ")
        return "".join(words).strip()

    @property
    def n_syllables(self):
        return sum(lyric.n_syllables for lyric in self)

    @property
    def has_utterance(self) -> bool:
        return len(self) > 0 and any(lyric.has_utterance for lyric in self)

    @property
    def lang(self) -> Optional[Lang]:
        if not self.has_utterance:
            return None
        if any(lyric.lang == Lang.ZH for lyric in self):
            return Lang.ZH
        return Lang.EN

    @classmethod
    def parse(cls, text: str):
        """Parse the text into words. (Assume each English word only takes up one syllable for simplicity.)"""
        def replace_space_in_zh_with_comma(text: str) -> str:
            lst = []
            subwords = list(filter(lambda w: len(w) > 0, text.split(" ")))
            last_is_zh = False
            for subword in subwords: 
                current_start_is_zh = _is_chinese_char(subword[0])
                current_end_is_zh = _is_chinese_char(subword[-1])
                if lst and last_is_zh and current_start_is_zh:
                    lst[-1] = lst[-1] + "，" + subword
                else:
                    lst.append(subword)
                last_is_zh = current_end_is_zh
            return " ".join(lst)
        
        def parse_subword(subword: str) -> List[Lyric]:
            lyric_list = []
            char_buffer = []
            for c in subword:
                if c in ["'", "’"]:  # type: quote
                    char_buffer.append(c)
                elif _has_only_latin_letters(c):  # type: English letter
                    # push and clear buffer
                    if char_buffer and char_buffer[-1].isdigit():
                        lyric_list.append(Lyric(text="".join(char_buffer)))
                        char_buffer = []
                    char_buffer.append(c)
                elif c.isdigit():  # type: digit
                    # push and clear buffer
                    if char_buffer and _has_only_latin_letters(char_buffer[-1]):
                        lyric_list.append(Lyric(text="".join(char_buffer)))
                        char_buffer = []
                    char_buffer.append(c)
                else:  # type: Chinese character or punctuation mark
                    # push and clear buffer
                    if char_buffer:
                        lyric_list.append(Lyric(text="".join(char_buffer)))
                        char_buffer = []
                    if _is_chinese_char(c):
                        lyric_list.append(Lyric(text=c))
                    else:  # must be punc
                        lyric_list.append(Lyric(punc=c))
            if char_buffer:
                lyric_list.append(Lyric(text="".join(char_buffer)))
            return lyric_list

        text = _normalize_text(text)
        text = replace_space_in_zh_with_comma(text)
        subwords = list(filter(lambda w: len(w) > 0, text.split(" ")))
        lyric_list = reduce(operator.add, [parse_subword(subword) for subword in subwords])
        return cls(lyric_list)

    def split(self):  # -> Tuple[Line, Line]
        """Split the line into two"""
        mid = len(self) // 2  # Default split point TODO: replace it with NLP
        punc_ind = np.array([i for i, lyric in enumerate(self) if not lyric.has_utterance])
        if len(punc_ind) > 0:  # break at punctuation if there's any
            diffs = [
                self[:punc_idx+1].without_punc().n_syllables -   # left
                self[punc_idx+1:].without_punc().n_syllables     # right
                for punc_idx in punc_ind
            ]
            # break at a midpoint that makes left and right have the minimum syllable number difference
            mid = punc_ind[np.argmin(np.abs(diffs))] + 1  # include the punc in the left split
        return self[:mid], self[mid:]

    def without_punc(self):  # -> self
        return self.__class__([lyric for lyric in self if lyric.text])


class SectionLyrics(list):
    def __init__(self, lines: List[Line], section_tag: Optional[str] = None):
        super().__init__(lines)
        self.section_tag = section_tag

    def __getitem__(self, index_or_slice):
        lst = list.__getitem__(self, index_or_slice)
        if isinstance(index_or_slice, int):
            return lst
        return self.__class__(lst, section_tag=self.section_tag)

    def __add__(self, other):
        if self.section_tag != other.section_tag:
            raise ZhLyricError("Two paragraphs' section_tags do not match")
        return self.__class__(list.__add__(self, other), section_tag=self.section_tag)

    def __mul__(self, other):
        if self.section_tag != other.section_tag:
            raise ZhLyricError("Two paragraphs' section_tags do not match")
        return self.__class__(list.__mul__(self, other), section_tag=self.section_tag)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self).__repr__()}, section_tag={self.section_tag.__repr__()})"

    def __str__(self) -> str:
        return "\n".join([add_section_tag(self.section_tag, "")] + [str(line) for line in self]).strip()

    @property
    def n_syllables(self):
        return sum(line.n_syllables for line in self)

    @property
    def has_utterance(self) -> bool:
        return len(self) > 0 and any(line.has_utterance for line in self)

    @property
    def lang(self) -> Optional[Lang]:
        if not self.has_utterance:
            return None
        if any(line.lang == Lang.ZH for line in self):
            return Lang.ZH
        return Lang.EN

    @classmethod
    def parse(cls, raw_lines: List[str], section_tag: Optional[str] = None):
        return cls([Line.parse(raw_line) for raw_line in raw_lines], section_tag=section_tag)
    
    def process(self, genre: str):  # -> Paragraph
        config = get_genre_config(genre)
        paragraph = self.__class__(_process_section_lyrics(self, config.slb_range), section_tag=self.section_tag)
        # ASR lyrics does not have many punctuations, remove them for now
        return paragraph.without_punc()

    def without_punc(self):  # -> SectionLyrics
        lines = [line.without_punc() for line in self]
        lines = [l for l in lines if len(l) > 0]  # remove empty lines
        return self.__class__(lines, section_tag=self.section_tag)


class SongLyrics(list):
    def __init__(self, paragraphs: List[SectionLyrics]):
        super().__init__(paragraphs)

    def __getitem__(self, index_or_slice):
        lst = list.__getitem__(self, index_or_slice)
        if isinstance(index_or_slice, int):
            return lst
        return self.__class__(lst)

    def __add__(self, other):
        return self.__class__(list.__add__(self, other))

    def __mul__(self, other):
        return self.__class__(list.__mul__(self, other))

    def __str__(self) -> str:
        return "\n".join(str(paragraph) for paragraph in self).strip()

    @property
    def n_lines(self) -> int:
        return sum(len(paragraph) for paragraph in self) if len(self) > 0 else 0

    @property
    def has_utterance(self) -> bool:
        return len(self) > 0 and all(p.has_utterance for p in self)

    @property
    def lang(self) -> Optional[Lang]:
        if not self.has_utterance:
            return None
        if any(paragraph.lang == Lang.ZH for paragraph in self):
            return Lang.ZH
        return Lang.EN

    @classmethod
    def parse(cls, lyrics: str):  # -> Song
        raw_lines = _split_raw_text(lyrics)
        phrases = [Phrase.parse(text=raw_line, normalize_tag=True, normalize_chinese=False) for raw_line in raw_lines]
        phrases = _move_out_section_tags(phrases)
        section_tag_ind = sorted(list(set([0] + [
            idx for idx, phrase in enumerate(phrases) if phrase.section_tag is not None
        ] + [len(phrases)])))
        groups = [phrases[a: b] for a, b in zip(section_tag_ind, section_tag_ind[1:])]
        return cls([
            SectionLyrics.parse(
                raw_lines=[phrase.format_text() for phrase in group if phrase.section_tag is None],
                section_tag=group[0].section_tag,
            ) for group in groups
        ])

    def process(self, genre: str):  # -> Song
        ps = [paragraph.process(genre) for paragraph in self]
        ps = _match_vocal_non_vocal_section_tags(ps)
        ps = _correct_non_vocal_section_tags(ps)
        ps = _assign_vocal_section_tags(ps)
        ps = _merge_vocal_sections(ps)
        # Need to process again since paragraph merging might include short lines
        # that did not get chance to concatenate (e.g. a single short line in a section).
        ps = [paragraph.process(genre) for paragraph in ps]
        ps = _repeat_single_section(ps)
        line_range = get_genre_config(genre).line_range
        ps = _process_song_lyrics_lines(self.__class__(ps), line_range)
        ps = _add_outro(ps)  # always add an outro
        # Using outro might introduce halluciations, replace it with inst instead
        ps = _reassign_last_non_vocal_to_inst(ps)
        return self.__class__(ps)

    def trim_lines_to(self, n_lines: int):  # -> Song
        """Trim the song to n_lines"""
        if self.n_lines <= n_lines:
            return self
        acc_lines = list(accumulate(len(paragraph) for paragraph in self))
        p_idx = bisect.bisect_left(acc_lines, n_lines)
        n_lines_to_remove = acc_lines[p_idx] - n_lines
        return self[:p_idx] + self.__class__([self[p_idx][:-n_lines_to_remove]])


# ====================================================================


_MAX_ITERS = 100  # as a guardrail to avoid being stuck in an infinite loop


def _trim_to_max(paragraph: List[Line], max_n_slbs_per_line: int) -> List[Line]:
    def lang_aware_line_split(line: Line) -> List[Line]:
        """EN Patch: Do not split English line"""
        if line.lang == Lang.EN:
            return [line]
        return line.split()

    last_processed_paragraph = list(paragraph)
    for _ in range(_MAX_ITERS):
        in_loop_paragraph = []
        for line in last_processed_paragraph:
            _lines = list(filter(lambda l: len(l) > 0, lang_aware_line_split(line))) if line.n_syllables > max_n_slbs_per_line else [line]
            in_loop_paragraph.extend(_lines)
        if len(in_loop_paragraph) == len(last_processed_paragraph):
            break  # if there's no change to the length, the entire paragraph should be the same
        last_processed_paragraph = in_loop_paragraph
    else:
        logger.warning("Lyrics Pre-processing: Reached maximum iterations")
    return last_processed_paragraph


def _comb_to_min(paragraph: List[Line], slb_range: Tuple[int, int]) -> List[Line]:
    """For each line that is too short, check two adjancent lines and concatenate with one of them"""
    last_processed_paragraph = list(paragraph)
    if len(last_processed_paragraph) <= 1:
        return last_processed_paragraph
    min_n_slbs_per_line, max_n_slbs_per_line = slb_range
    for _ in range(_MAX_ITERS):
        for idx, line in enumerate(last_processed_paragraph):
            if line.n_syllables >= min_n_slbs_per_line:  # no need to process
                continue
            line_prev = None if idx == 0 else last_processed_paragraph[idx-1]
            line_next = None if idx == len(last_processed_paragraph) - 1 else last_processed_paragraph[idx+1]
            if line_prev is None and line_next is None:
                continue  # this is the only line, no need to process
            if line.n_syllables + min([l.n_syllables for l in [line_prev, line_next] if l is not None]) > max_n_slbs_per_line:
                continue  # unable to process

            # Obtain the indices of two lines that we want to merge
            if line_prev is not None and line_next is not None:
                # Concatenate with the shortest
                if line_prev.n_syllables < line_next.n_syllables:
                    idx_a, idx_b = idx-1, idx
                else:
                    idx_a, idx_b = idx, idx+1
            elif line_prev is not None:
                idx_a, idx_b = idx-1, idx
            else:  # line_next is not None
                idx_a, idx_b = idx, idx+1

            # Concatenate two lines
            line_a: Line = last_processed_paragraph[idx_a]
            line_b: Line = last_processed_paragraph[idx_b]
            if Lang.EN in [line_a.lang, line_b.lang]:
                continue  # EN Patch: No concat for English line(s)
            line_punc = Line([Lyric(punc="，")])  # add a full comma in the middle
            line_comb = (line_a + line_b) if (not line_a[-1].has_utterance or not line_b[0].has_utterance) else (line_a + line_punc + line_b)
            last_processed_paragraph = last_processed_paragraph[:idx_a] + [line_comb] + last_processed_paragraph[idx_b+1:]
            break  # reset, rescan from the start
        else:
            break  # no break means no processing happened in the previous loop, break the outer loop, exit
    else:
        logger.warning("Lyrics Pre-processing: Reached maximum iterations")
    return last_processed_paragraph


def _process_section_lyrics(
    paragraph: List[Line],
    slb_range: Tuple[int, int]
) -> List[Line]:
    _, max_n_slbs_per_line = slb_range
    return _comb_to_min(_trim_to_max(paragraph, max_n_slbs_per_line), slb_range)


# ====================================================================


def _split_text_by_parens(text: str) -> List[str]:
    # The reason is to handle a long line with multiple section tags. The splitting makes each line
    # only has one leading section tag, which can then be parsed by the lyrics parser.
    # The left parentheses do not necessarily indicate a section tag, but we can do this because:
    # 1. The current lyrics input does not support specical characters, including parentheses/brackets
    # 2. It is reasonable to treat a left paren as a line break, even if it's not part of a section tag
    # It is also possible to use regex to match all the occurances of section tags, but let's keep it
    # simple for now.
    left_parens = [p[0] for p in SECTION_PARENS]
    ind = sorted(list(set([0] + [i for i, c in enumerate(text) if c in left_parens] + [len(text)])))
    lines = [text[a:b] for a, b in zip(ind, ind[1:])]
    lines = [l.strip() for l in lines]
    return [l for l in lines if len(l) > 0]


def _split_raw_text(text: str) -> List[str]:
    lines = _split_text_by_parens(text)
    return reduce(operator.add, [line.split("\n") for line in lines])


def _match_vocal_non_vocal_section_tags(paragraphs: List[SectionLyrics]) -> List[SectionLyrics]:
    """Make sure the section tag assignment is correct. Lyrics should always under vocal sections.
    Some sections might have empty section tag after processing.
    """
    ps = []
    for paragraph in paragraphs:
        if paragraph.has_utterance and paragraph.section_tag not in [None] + VOCAL_SECTION_TAGS:
            # Take the inst section tag out as an individual section
            ps.append(SectionLyrics([], section_tag=paragraph.section_tag))
            ps.append(SectionLyrics(paragraph))
        elif not paragraph.has_utterance and paragraph.section_tag in [None] + VOCAL_SECTION_TAGS:
            continue  # Skip empty vocal section
        else:
            ps.append(paragraph)  # Keep
    return ps


def _correct_non_vocal_section_tags(paragraphs: List[SectionLyrics]) -> List[SectionLyrics]:
    """
    Always make sure intro is the first section, outro is the last, inst is in the middle.
    Merge non-vocal sections that have the same name.
    Run this after _match_vocal_non_vocal_section_tags.
    """
    non_vocal_section_ind = [idx for idx, p in enumerate(paragraphs) if not p.has_utterance]
    vocal_ind = [idx for idx, p in enumerate(paragraphs) if p.has_utterance]
    all_ind = list(range(len(paragraphs)))
    # consecutive sections starts from 0
    intro_ind = [inst_idx for inst_idx, p_idx in zip(non_vocal_section_ind, all_ind) if inst_idx == p_idx]
    # consecutive sections starts from -1  (reversed order)
    outro_ind = [inst_idx for inst_idx, p_idx in zip(reversed(non_vocal_section_ind), reversed(all_ind)) if inst_idx == p_idx]
    intro_outro_ind = intro_ind + outro_ind
    # all inst tags
    inst_ind = [idx for idx in non_vocal_section_ind if idx not in intro_outro_ind]
    # remove consecutive inst sections
    inst_ind = [inst_ind[idx] for idx in [0] + (np.where(np.diff(inst_ind) != 1)[0] + 1).tolist()] if inst_ind else []
    # take only one intro section and outro section
    intro_idx = intro_ind[0] if intro_ind else []
    outro_idx = outro_ind[0] if outro_ind else []

    ps = []
    for idx, paragraph in enumerate(paragraphs):
        # vocal
        if idx in vocal_ind:
            ps.append(paragraph)
            continue
        # non vocal
        if idx not in [intro_idx, outro_idx] + inst_ind:
            continue  # skip redundant consecetive non-vocal sections
        if idx == intro_idx:
            section_tag = "intro"
        elif idx == outro_idx:
            section_tag = "outro"
        else:
            section_tag = "inst"
        ps.append(SectionLyrics([], section_tag=section_tag))
    return ps


def _assign_vocal_section_tags(paragraphs: List[SectionLyrics]) -> List[SectionLyrics]:
    """Assign section tags to vocal paragraphs that do not have section tags."""
    if all(p.section_tag is not None for p in paragraphs):
        return paragraphs[:]

    # This can get very complicated. Use a simple algorithm for now (verse chorus alter)
    last_vocal_section_tag = "chorus"
    ps = []
    for paragraph in paragraphs:
        if paragraph.has_utterance:  # vocal
            if paragraph.section_tag is None:
                section_tag = "verse" if last_vocal_section_tag == "chorus" else "chorus"
                ps.append(SectionLyrics(paragraph, section_tag=section_tag))
            else:
                section_tag = paragraph.section_tag
                ps.append(paragraph)
            last_vocal_section_tag = section_tag
        else:  # non-vocal
            ps.append(paragraph)
    return ps


_MIN_N_LINES_PER_SECTION = 4


def _merge_vocal_sections(paragraphs: List[SectionLyrics]) -> List[SectionLyrics]:
    """Combine short vocal sections with the same section_tag.
    This function assumes all the paragraphs have section tags.
    """
    ps = []
    last_section_tag = ""
    for idx, paragraph in enumerate(paragraphs):
        if paragraph.section_tag not in VOCAL_SECTION_TAGS or paragraph.section_tag != last_section_tag:
            ps.append(paragraph)
            last_section_tag = paragraph.section_tag
        # same section_tag
        elif ((len(paragraph) < _MIN_N_LINES_PER_SECTION) or  # current paragraph too short
              (idx > 0 and len(paragraphs[idx-1]) < _MIN_N_LINES_PER_SECTION)):  # last paragraph too short
            if len(ps) == 0:
                ps.append(paragraph)
            else:  # modify the last one
                ps[-1] = ps[-1] + paragraph
        else:  # proper length
            ps.append(paragraph)
    return ps


# Intentionally set this value ower than _MIN_N_LINES_PER_SECTION.
# If there is only one line, repeating line too many times could be repetitive.
_MIN_N_LINES_SINGLE_SECTION = 4


def _repeat_single_section(paragraphs: List[SectionLyrics]) -> List[SectionLyrics]:
    """If there is only one vocal section, repeat the lines in the section to reach _MIN_N_LINES.
    This function is specifically designed to handle short input.
    """
    if len([paragraph for paragraph in paragraphs if paragraph.has_utterance]) != 1:
        return paragraphs
    ps = []
    for paragraph in paragraphs:
        if paragraph.has_utterance and len(paragraph) < _MIN_N_LINES_SINGLE_SECTION:  # add a section with a different section tag
            n_repeats = math.ceil(_MIN_N_LINES_SINGLE_SECTION / len(paragraph))
            new_paragraph = reduce(operator.add, [paragraph for _ in range(n_repeats)])
            ps.append(new_paragraph)
        else:
            ps.append(paragraph)
    return ps


def _process_song_lyrics_lines(
    song: SongLyrics,
    line_range: Tuple[int, int],
) -> SongLyrics:
    """Process lyrics to meet the line range requirement"""
    _MAX_N_REPEATS = 2  # maximum times the end portion of the song should repeat.

    def add_intro(song: SongLyrics) -> SongLyrics:
        if random.random() < 0.5:
            return SongLyrics([SectionLyrics([], section_tag="intro")]) + song
        else:
            return song

    def add_outro(song: SongLyrics) -> SongLyrics:
        return song + SongLyrics([SectionLyrics([], section_tag="outro")])

    def add_intro_and_outro(song: SongLyrics) -> SongLyrics:
        return add_outro(add_intro(song))
    
    def add_non_vocal_sections(song: SongLyrics) -> SongLyrics:
        # NOTE: V3 patch, only add outro to avoid the long intro issue
        # can_add_intro = not song or song[0].has_utterance
        can_add_outro = not song or song[-1].has_utterance
        # Short song, add non vocal sections as many as possible
        # if song.n_lines < _MIN_N_LINES_PER_SECTION:
        #     if can_add_intro and can_add_outro:
        #         return add_intro_and_outro(song)
        #     elif can_add_outro:  # consider add to the end first
        #         return add_outro(song)
        #     elif can_add_intro:
        #         return add_intro(song)
        #     return song[:]
        # Medium song, only add outro
        if can_add_outro:
            return add_outro(song)
        return song[:]

    min_n_lines, max_n_lines = line_range
    # Meet the requirement already
    if min_n_lines <= song.n_lines <= max_n_lines:
        return song
    # Too long
    if song.n_lines > max_n_lines:
        return song.trim_lines_to(max_n_lines)
    # Too short
    prev_iter_song = song
    if len([p for p in prev_iter_song if p.has_utterance]) > 1:  # more than one vocal sections
        for n_paragraphs in range(1, len(song)+1):
            for n_repeats in range(1, _MAX_N_REPEATS+1):
                # Must re-assign and merge non-vocal section tags
                curr_iter_song = song.__class__(
                    _merge_vocal_sections(_correct_non_vocal_section_tags(song + song[-n_paragraphs:] * n_repeats))
                )
                if min_n_lines <= curr_iter_song.n_lines <= max_n_lines:
                    return curr_iter_song
                if curr_iter_song.n_lines > max_n_lines:
                    return add_non_vocal_sections(prev_iter_song)  # use the last trial
                prev_iter_song = curr_iter_song
    # If there is only one vocal section, simply add an intro or outro, otherwise
    # the song could be too repetitive.
    return add_non_vocal_sections(prev_iter_song)


def _add_outro(song: SongLyrics) -> SongLyrics:
    """Add an outro section if the last section is not outro"""
    if not song or song[-1].section_tag == "outro":
        return song[:]
    return song[:] + SongLyrics([SectionLyrics([], section_tag="outro")])


def _reassign_last_non_vocal_to_inst(song: SongLyrics) -> SongLyrics:
    if not song or song[-1].section_tag != "outro":
        return song[:]
    return song[:-1] + SongLyrics([SectionLyrics([], section_tag="inst")])


# ====================================================================


def _move_out_section_tags(phrases: List[Phrase]) -> List[Phrase]:
    """Move out every phrase's section tag, no tag merging (different from the function in sami_tokenizer)"""
    out_phrases = []
    for phrase in phrases:
        if phrase.section_tag:
            out_phrases.append(Phrase(section_tag=phrase.section_tag))
        if phrase.has_utterance:
            out_phrases.append(phrase._replace(section_tag=None))
    return out_phrases


def _normalize_text(text: str) -> str:
    # Convert full-width digits to regular digits
    normalized_text = ''.join(
        chr(ord(char) - 0xFEE0) if '０' <= char <= '９' else char
        for char in text
    )
    # Normalize the text to NFKD form and remove diacritical marks
    normalized_text = unicodedata.normalize('NFKD', normalized_text)
    normalized_text = ''.join(
        char for char in normalized_text if unicodedata.category(char) != 'Mn'
    )
    return normalized_text


def _has_only_latin_letters(name: str) -> bool:
    char_set = string.ascii_letters
    return all(True if x in char_set else False for x in name)
