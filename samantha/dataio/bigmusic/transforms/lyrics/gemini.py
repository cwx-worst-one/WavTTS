import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from samantha.dataio.bigmusic.base_transform import MusicMetaRWTransform


class GeminiLyricsFormatter(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: tuple[str] = (
            "meta.lyrics.utterances",
            "meta.standard_meta.llm_augmented_meta.structure_level_tags",
        ),
        out_key: str = "lyrics",
        keyword_dropout_rate: float = 0.0,
        keyword_norm_rate: float = 0.0,
        tag_dropout_rate: float = 0.0,
        tag_full_dropout_rate: float = 0.0,
        sentence_dropout_rate: float = 0.0,
        desc_dropout_rate: float = 0.0,
        keyword_fields: tuple[str] = (
            "structure_features",
            "basic_musical_information",
            "mood_emotion_dynamic_change",
            "harmonic_progression",
            "instruments",
            "vocal_features",
        ),
        **kwargs,
    ):
        """
        Args:
            keyword_dropout_rate: Dropout rate for **each keyword**.
            keyword_norm_rate: Normalization rate for **each keyword**.
            tag_dropout_rate: Dropout rate for **all the "tags"** (section level keywords).
            sentence_dropout_rate: Dropout rate for **each natural language sentence**.
            desc_dropout_rate: Dropout rate for **all the descriptions**.
        """
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.keyword_dropout_rate = keyword_dropout_rate
        self.keyword_norm_rate = keyword_norm_rate
        self.tag_dropout_rate = tag_dropout_rate
        self.tag_full_dropout_rate = tag_full_dropout_rate
        self.sentence_dropout_rate = sentence_dropout_rate
        self.desc_dropout_rate = desc_dropout_rate
        self.keyword_fields = keyword_fields

    def call(self, item: tuple, **kwargs):
        utterances, structure_level_tags = item
        if not structure_level_tags:
            return None
        return format_gemini_lyrics(
            utterances,
            structure_level_tags,
            keyword_dropout_rate=self.keyword_dropout_rate,
            keyword_norm_rate=self.keyword_norm_rate,
            tag_dropout_rate=self.tag_dropout_rate,
            tag_full_dropout_rate=self.tag_full_dropout_rate,
            sentence_dropout_rate=self.sentence_dropout_rate,
            desc_dropout_rate=self.desc_dropout_rate,
            keyword_fields=self.keyword_fields,
        )


class GeminiLyricsAugmentor(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: tuple[str] = ("prompt", "lyrics"),
        out_key: str = "lyrics",
        **kwargs,
    ):
        super().__init__(in_key, out_key, **kwargs)

    def call(self, item: tuple, **kwargs) -> str:
        prompt, lyrics = item
        return apply_gemini_template_to_lyrics(prompt, lyrics)


def format_gemini_lyrics(
    utterances: list[dict], structure_level_tags: list[dict], **kwargs
) -> str:
    lyrics = _format_sections(
        _group_lyric_lines_by_time(
            _parse_lyric_lines_with_timestamps(utterances), structure_level_tags
        ),
        **kwargs,
    )
    # replace '.,' with '.', which happens in description + keywords
    lyrics = lyrics.replace(".,", ".")
    return lyrics


def _parse_lyric_lines_with_timestamps(utterances: list[dict]) -> list[dict]:
    if not utterances or len(utterances) == 0:
        return []
    return [
        {
            "text": utt["text"],
            "time_span": (utt["start_time"] / 1000, utt["end_time"] / 1000),
        }
        for utt in utterances
    ]


def _group_lyric_lines_by_time(
    lyric_lines: list[dict], structure_level_tags: list[dict]
) -> list[dict]:
    def get_overlap(time_span1, time_span2) -> float:
        return max(
            0, min(time_span1[1], time_span2[1]) - max(time_span1[0], time_span2[0])
        )

    n_sections = len(structure_level_tags)

    ind_register = []
    if lyric_lines and len(lyric_lines) > 0:
        for line in lyric_lines:
            overlaps = [
                get_overlap(line["time_span"], section["section_interval"])
                for section in structure_level_tags
            ]
            # pick the index of the max overlap
            max_overlap_idx = max(range(n_sections), key=lambda x: overlaps[x])
            ind_register.append(max_overlap_idx)

    groups = []
    for i in range(n_sections):
        group = {"section": structure_level_tags[i]}
        if lyric_lines and len(lyric_lines) > 0:
            group["lines"] = [
                lyric_lines[j] for j in range(len(lyric_lines)) if ind_register[j] == i
            ]
        groups.append(group)
    return groups


def _shuffle_list(lst: list) -> list:
    lst = lst[:]
    random.shuffle(lst)
    return lst


def _format_sections(
    groups: list[dict],
    keyword_dropout_rate: float = 0.0,
    keyword_norm_rate: float = 0.0,
    tag_dropout_rate: float = 0.0,
    tag_full_dropout_rate: float = 0.0,
    sentence_dropout_rate: float = 0.0,
    desc_dropout_rate: float = 0.0,
    keyword_fields: tuple[str] = (
        "structure_features",
        "basic_musical_information",
        "mood_emotion_dynamic_change",
        "harmonic_progression",
        "instruments",
        "vocal_features",
    ),
) -> list:

    def format_keywords(kws: list[str], shuffle: bool = True) -> str:
        if shuffle:
            kws = _shuffle_list(kws)
        if keyword_dropout_rate > 0.0:
            kws = [kw for kw in kws if random.random() > keyword_dropout_rate]
        if keyword_norm_rate > 0.0:  # lower by chance
            kws = [
                kw.lower() if random.random() < keyword_norm_rate else kw for kw in kws
            ]
        return ", ".join(kws)

    def format_instruments(instruments: dict[str, list[str]]) -> str:
        instrumental_timbre_and_techniques = instruments.get(
            "instrumental_timbre_and_techniques", []
        )
        if isinstance(instrumental_timbre_and_techniques, dict):
            instrumental_timbre_and_techniques = [
                item
                for sublist in instrumental_timbre_and_techniques.values()
                for item in sublist
            ]
        return format_keywords(
            instruments.get("main_instrument", [])
            + instruments.get("accompaniment_instruments", [])
            + instrumental_timbre_and_techniques
        )

    def format_one_vocal_feature(
        vocal_feature: dict, use_singer_label: bool = False
    ) -> str:
        x = format_keywords(
            vocal_feature.get("gender", [])
            + vocal_feature.get("timbre", [])
            + vocal_feature.get("techniques", [])
        )
        if use_singer_label and x:
            x = f"{vocal_feature['singer_label']}: {x}"
        return x

    def format_vocal_features(vocal_features: list[dict]) -> list[str]:
        if not vocal_features:
            return []
        vfs = [
            format_one_vocal_feature(vocal_feature, use_singer_label=True)
            for vocal_feature in vocal_features
        ]
        return [vf for vf in vfs if vf]  # remove empty strings

    def format_natural_language(sentences: list[str]) -> str:
        sentences = _shuffle_list(sentences)
        sentences = [x for x in sentences if x]
        sentences = [x for x in sentences if random.random() > sentence_dropout_rate]
        if sentences:
            return " ".join(sentences)
        return ""

    def format_section(group: dict, tag_full_dropout: bool) -> str:
        section = group["section"]
        lines = group.get("lines", [])

        # = section tag
        section_tag = f"[{section['section_label']}]"
        final_lines = []

        if random.random() > tag_dropout_rate:
            # = MIR tags
            mir_desc = format_keywords(
                [
                    x
                    for x in [
                        (
                            format_keywords(section.get("structure_features", []))
                            if "structure_features" in keyword_fields
                            else ""
                        ),
                        (
                            format_keywords(
                                section.get("basic_musical_information", [])
                            )
                            if "basic_musical_information" in keyword_fields
                            else ""
                        ),
                        (
                            format_keywords(
                                section.get("mood_emotion_dynamic_change", [])
                            )
                            if "mood_emotion_dynamic_change" in keyword_fields
                            else ""
                        ),
                        (
                            format_keywords(section.get("harmonic_progression", []))
                            if "harmonic_progression" in keyword_fields
                            else ""
                        ),
                        (
                            format_instruments(section.get("instruments", {}))
                            if "instruments" in keyword_fields
                            else ""
                        ),
                    ]
                    if x
                ]
            )
            if mir_desc:
                final_lines.append(f"[{mir_desc}]")

            # = vocal feautres
            vocal_features = (
                format_vocal_features(section.get("vocal_features", None))
                if "vocal_features" in keyword_fields
                else []
            )  # multiple results
            if vocal_features:
                final_lines.extend([f"[{vf}]" for vf in vocal_features])

        if random.random() > desc_dropout_rate:
            # = natural language descriptions
            descs = [
                section.get("section_description", []),
                section.get("melodic_features", {}).get(
                    "motif_contour_and_features", []
                ),
                section.get("instruments", {}).get("instruments_development", []),
            ]
            new_descs = []
            for desc in descs:
                if isinstance(desc, list):
                    new_descs.extend(desc)
                else:
                    new_descs.append(desc)

            desc = format_natural_language(descs)
            if desc:
                final_lines.append(f"[{desc}]")

        # shuffle the order of description lines
        final_lines = _shuffle_list(final_lines)

        if tag_full_dropout:
            final_lines = []

        # = section tag (prepend)
        final_lines = [section_tag] + final_lines

        # = lyrics (append)
        lyrics = "\n".join([line["text"] for line in lines])
        if lyrics:
            final_lines.append(lyrics)

        return "\n".join(final_lines)

    formatted = []
    tag_full_dropout = random.random() < tag_full_dropout_rate
    for group in groups:
        formatted.append(format_section(group, tag_full_dropout))
    return "\n".join(formatted)


# ===================================================
# Inference stage augmentation
# ===================================================


def _keyword_to_genre(keyword: str) -> Optional[str]:
    keyword_map = {
        "folk": ["folk", "民谣"],
        "rock": ["rock", "摇滚"],
        "r&b": ["r&b", "soul", "节奏布鲁斯", "灵魂"],
        "rap": ["hip hop", "rap", "trap", "嘻哈", "说唱", "陷阱"],
        "reggae": ["reggae", "雷鬼"],
        "funk": ["funk", "放克"],
        "electronic": [
            "edm",
            "dance",
            "house",
            "trance",
            "drum and bass",
            "电子",
            "浩室",
        ],
        "jazz": ["jazz", "爵士"],
        "blues": ["blues", "蓝调"],
        "chinese": ["chinese style", "国风"],
        "pop": ["pop", "ballad", "流行", "芭乐"],
    }
    keyword = keyword.lower().strip()  # normalize it first
    for category_tag, kws_to_check in keyword_map.items():
        if any(kw in keyword for kw in kws_to_check):
            return category_tag
    return None


def _keywords_to_genre(keywords: list[str]) -> str:
    for kw in keywords:
        genre = _keyword_to_genre(kw)
        if genre:
            return genre
    return "pop"  # default


def _extract_genre(prompt: str) -> str:
    kws = [kw.strip() for kw in prompt.split(",")][:3]  # take the first 3 keywords
    return _keywords_to_genre(kws)


def _is_keyword_bpm(keyword: str) -> Optional[str]:
    """
    Extracts BPM/tempo from a string.
    Returns the BPM as a string like "123 bpm", or None if not found.
    """
    match = re.search(
        r"\b(?:bpm|tempo)\s*[:~]?\s*(\d+)\b|\b(\d+)\s*[:~]?\s*(?:bpm|tempo)\b",
        keyword.lower(),
    )
    if match:
        bpm = match.group(1) or match.group(2)
        return f"{bpm} bpm"
    return None


def _is_keyword_key(keyword: str) -> Optional[str]:
    """
    Extracts musical key (major/minor) from a string.
    Handles formats like:
    - "a major"
    - "b minor"
    - "a:major"
    - "modulates to c major"
    - "key of d minor"
    - "eb major"
    - "f# minor"
    Returns normalized string like "c major", or None.
    """
    match = re.search(
        r"\b(?:key of\s+)?([a-g](?:#|b|##|bb)?)\s*:?[\s-]*(major|minor)\b",
        keyword.lower(),
    )
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


def _is_keyword_time_signature(keyword: str) -> Optional[str]:
    """
    Extracts time signatures from a string.
    Handles formats like:
    - "3/4"
    - "6 / 8"
    Returns normalized form like "3/4", or None.
    """
    match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", keyword)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return None


def _is_keyword_bars(keyword: str) -> Optional[str]:
    if keyword.endswith(" bars"):
        return keyword
    return None


def _extract_mir(prompt: str) -> tuple[list[str], dict]:
    keywords = [kw.strip() for kw in prompt.split(",")]
    mir_info = {}
    for kw in keywords:
        bpm = _is_keyword_bpm(kw)
        if bpm:
            mir_info["bpm"] = bpm
        ts = _is_keyword_time_signature(kw)
        if ts:
            mir_info["time_signature"] = ts
        key = _is_keyword_key(kw)
        if key:
            mir_info["key"] = key
        bars = _is_keyword_bars(kw)
        if bars:
            mir_info["bars"] = bars
    return mir_info


def _remove_mir_from_keywords(keywords: list[str]) -> list[str]:
    fns = [
        _is_keyword_bpm,
        _is_keyword_time_signature,
        _is_keyword_key,
        _is_keyword_bars,
    ]
    return [kw for kw in keywords if not any(fn(kw) for fn in fns)]


def _extract_template(lyrics: str) -> dict:
    lines = lyrics.split("\n")
    tag_ind = [i for i, line in enumerate(lines) if line.startswith("[")]

    # Considering the section-level keyword dropout, we need to check the indices to make sure
    # every section has its section-level keywords.
    if len(tag_ind) % 2 != 0:
        return {}
    for i, j in zip(tag_ind[::2], tag_ind[1::2]):
        if j - i != 1:
            return {}

    template = defaultdict(list)
    for i, j in zip(tag_ind[::2], tag_ind[1::2]):
        # i + 1 == j
        tag_name = lines[i][1:-1]  # remove the bracket
        if len(tag_name) > 20:  # something is wrong
            return {}
        keywords = [kw.strip() for kw in lines[j][1:-1].split(",")]
        keywords = _remove_mir_from_keywords(keywords)
        template[tag_name].append(keywords)

    return template


def generate_template(table_dir, template_fp) -> dict:
    def write_json(template_base: dict, fp: str):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(template_base, f, ensure_ascii=False, indent=2)

    def is_str_valid(s) -> bool:
        if not isinstance(s, str) or not s:
            return False
        return True

    table_fps = Path(table_dir).glob("*.csv")
    template_base = defaultdict(list)
    for table_fp in table_fps:
        df = pd.read_csv(table_fp)
        records = df.to_records(index=False)
        for record in records:
            lyrics = record.lyrics
            prompt = record.prompt
            if not is_str_valid(lyrics) or not is_str_valid(prompt):
                continue
            template = _extract_template(lyrics)
            if not template:
                continue
            genre = _extract_genre(prompt)
            template_base[genre].append(template)

    write_json(template_base, template_fp)


def _load_json(json_fp):
    with open(json_fp, "r") as f:
        x = json.load(f)
    return x


TEMPLATE_BASE = _load_json(
    Path(__file__).parent / "gemini_section_keyword_templates.json"
)


def apply_gemini_template_to_lyrics(prompt: str, lyrics: str) -> str:
    def sample_default_keywords(template: dict) -> list[str]:
        for key in ["chorus", "verse", "bridge", "pre-chorus"]:
            if key in template:
                return template[key]
        return next(template.values())

    genre = _extract_genre(prompt)
    templates = TEMPLATE_BASE.get(
        genre, "pop"
    )  # just in case the genre is not in the template base

    template = random.choice(templates)
    mir = _extract_mir(prompt)
    mir_keywords = list(mir.values())

    counter = defaultdict(int)

    new_lines = []
    for line in lyrics.split("\n"):
        if not line.startswith("["):
            new_lines.append(line)
        else:
            tag_name = line[1:-1]  # remove the bracket
            idx = max(min(counter[tag_name], len(template.get(tag_name, [])) - 1), 0)
            keywords = template.get(tag_name, sample_default_keywords(template))[idx]
            keywords = keywords + mir_keywords
            new_lines.append(line)
            new_lines.append(f"[{', '.join(keywords)}]")  # additional line
            counter[tag_name] += 1

    return "\n".join(new_lines)
