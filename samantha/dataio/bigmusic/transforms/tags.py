import copy
import re
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Optional, Union

from ..tokenizers.style_tag_tokenizer import (
    StyleTagTokenizer,
    StyleTagTokenizerError,
    StyleTagVocab,
)
from ..utils.dcbase import DCBase

__all__ = ["TagError", "tokenize_tags", "transform_tags", "validate_tags"]


_DEFAULT_SINKING_THRESHOLD = 0.51
_EMPTY_TAG = ""


class TagError(Exception):
    pass


def transform_tags(
    vocab: StyleTagVocab,
    meta: dict,
    insts: Optional[list[str]] = None,
    tempo: Optional[float] = None,
    key_mode: Optional[str] = None,
    standardize: bool = True,
    extend_extra: bool = True,
    source_selection: str = "fill",
) -> dict:
    audio_tags = _parse_audio_tags_into_tags_proto(meta, _parse_audio_tags_from_meta)
    human_tags = _parse_audio_tags_into_tags_proto(meta, _parse_human_label_from_meta)
    musicfm_finegrained_tags = _parse_musicfm_finegrained_tags_into_tags_proto(
        meta, _parse_musicfm_finegrained_tags_from_meta
    )
    llm_tags = _parse_audio_tags_into_tags_proto(meta, _parse_llm_tags_from_meta)
    sa_tags = _parse_sa_tags_into_tags_proto(meta, _parse_sa_tags_from_meta)
    inst_tags = _parse_insts_to_tags_proto(insts)
    tempo_tags = _parse_tempo_to_tags_proto(tempo)
    key_mode_tags = _parse_key_mode_to_tags_proto(key_mode)
    # Arrange tag sources from high priority to low priority
    valid_tags = list(
        filter(
            None,
            [
                human_tags,
                audio_tags,
                musicfm_finegrained_tags,
                llm_tags,
                sa_tags,
                inst_tags,
                tempo_tags,
                key_mode_tags,
            ],
        )
    )
    if not valid_tags:
        raise TagError("No valid tags")
    if source_selection == "fill":
        tags = TagsProto.fill(*valid_tags)
    elif source_selection == "merge":
        tags = TagsProto.merge(*valid_tags)
    else:
        raise TagError(f"Invalid source_selection: {source_selection}")
    if standardize:
        tags.standardize_inplace(vocab)
    if extend_extra:
        tags.extend_extra_inplace()  # this step might introduce OOV tags
    tags, oov = tags.remove_oov_tags(vocab)
    return {"tags": tags.fill_empty_inplace().to_dict(), "oov_tags": oov}


def validate_tags(style_tags: dict) -> None:
    tags_proto = TagsProto.from_dict(style_tags)
    if any(
        lang
        for lang in tags_proto.language
        if lang
        not in [
            "English",
            "Chinese",
            "Cantonese",
            "Japanese",
            "Sichuanese",
            "Instrumental/Non-Vocal",
        ]
    ):
        raise TagError(f"Audio tags contains invalid language: {tags_proto.language}")
    if set(["Chinese", "Cantonese"]).issubset(tags_proto.language):
        raise TagError(f"Audio tags contains invalid language: {tags_proto.language}")


def tokenize_tags(style_tags: dict, tokenizer: StyleTagTokenizer) -> dict:
    """
    Tokenize each category of style tags into a list of integers.
    """

    def map_key(k: str) -> str:
        return {"genre_extra": "genre"}.get(
            k, k
        )  # "genre_extra" and "genre" share the same vocab

    def get_input_id(category: str, tag: str) -> str:
        category = map_key(category)
        if category not in tokenizer.vocab.token_to_id:
            raise TagError(f"Invalid category: {category}")
        # Use the _EMPTY_TAG in category if possible, otherwise use the shared empty tag
        # under "empty" category
        if tag == _EMPTY_TAG and tag not in tokenizer.vocab.token_to_id.get(
            category, []
        ):
            return tokenizer("empty", tag)
        return tokenizer(category, tag)

    def get_input_ids(category: str, tags: str) -> str:
        input_ids = []
        for tag in tags:
            try:
                input_ids.append(get_input_id(category, tag))
            except StyleTagTokenizerError as e:
                raise TagError(str(e))
        return input_ids

    style_tokens = {}
    for category, tags in style_tags.items():
        style_tokens[category] = get_input_ids(category, tags)
    return style_tokens


@dataclass
class AudioTagsProto(DCBase):
    extra: list[str] = field(default_factory=list)
    genre: list[str] = field(default_factory=list)
    genre_extra: list[str] = field(default_factory=list)
    instrument: list[str] = field(default_factory=list)
    language: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)
    scene: list[str] = field(default_factory=list)
    vocal_gender: list[str] = field(default_factory=list)
    vocal_timbre: list[str] = field(default_factory=list)
    remark: str = ""
    satisfy_filter_standard: str = "yes"

    @classmethod
    def from_dict_with_norm(cls, d: dict) -> "AudioTagsProto":
        """Parse the audio_tags field in metadata with tag normalization (str -> list[str])"""

        def parse_tags(k: str, v: Union[str, list[str]]) -> Union[str, list[str]]:
            if k not in [
                "extra",
                "genre",
                "genre_extra",
                "instrument",
                "language",
                "mood",
                "scene",
                "vocal_gender",
                "vocal_timbre",
            ]:
                return v
            return _normalize_tags(v)

        # Only normalize tag slots. Fields not in the proto will be dropped.
        return cls.from_dict(
            {k: parse_tags(k, v) for k, v in d.items() if v is not None}
        )


@dataclass
class SATagsProto(DCBase):
    genre: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)
    theme: list[str] = field(default_factory=list)
    language: list[str] = field(default_factory=list)
    vocal_gender: list[str] = field(default_factory=list)
    sinking: float = 0.0

    @classmethod
    def from_dict_with_norm(cls, d: dict) -> "SATagsProto":
        return SATagsProto(
            genre=_normalize_tags(d.get("Genre20", {}).get("result")),
            mood=_normalize_tags(d.get("Mood", {}).get("result")),
            theme=_normalize_tags(d.get("Theme", {}).get("result")),
            language=_normalize_tags(d.get("Language", {}).get("result")),
            vocal_gender=_normalize_tags(
                [
                    x
                    for x in d.get("sa_gender", {}).get("result", [])
                    if x and x not in ["adult"]
                ]
            ),
            sinking=d.get("MusicLowQuality", {}).get("Sinking", 0.0),
        )


@dataclass
class MusicFMFinegrainedTagsProto(DCBase):
    genre: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)
    scene: list[str] = field(default_factory=list)
    vocal_timbre: list[str] = field(default_factory=list)
    vocal_gender: list[str] = field(default_factory=list)

    @classmethod
    def from_dict_with_norm(cls, d: dict) -> "MusicFMFinegrainedTagsProto":
        return cls(
            genre=_normalize_tags(
                [
                    x
                    for x in d.get("GENRE", [])
                    if x and x not in ["Other genre", "Empty"]
                ]
            ),
            mood=_normalize_tags(
                [x for x in d.get("MOOD", []) if x and x not in ["Other mood", "Empty"]]
            ),
            scene=_normalize_tags(
                [
                    x
                    for x in d.get("THEME", [])
                    if x and x not in ["Other scene", "Empty"]
                ]
            ),
            vocal_timbre=_normalize_tags(
                [x for x in d.get("TIMBRE", []) if x and x not in ["Empty"]]
            ),
            vocal_gender=_normalize_tags(
                [
                    x
                    for x in d.get("GENDER", [])
                    if x and x not in ["Unkonwn", "Unknown", "Adult"]
                ]
            ),
        )


@dataclass
class TagsProto(DCBase):
    genre: list[str] = field(default_factory=list)
    genre_extra: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)
    scene: list[str] = field(default_factory=list)
    vocal_gender: list[str] = field(default_factory=list)
    vocal_timbre: list[str] = field(default_factory=list)
    language: list[str] = field(default_factory=list)
    is_sinking: list[str] = field(default_factory=list)
    instrument: list[str] = field(default_factory=list)
    tempo: list[str] = field(default_factory=list)
    key: list[str] = field(default_factory=list)
    mode: list[str] = field(default_factory=list)

    @classmethod
    def from_sa_tags(
        cls, sa_tags: SATagsProto, sinking_threshold: float = _DEFAULT_SINKING_THRESHOLD
    ) -> "TagsProto":
        return cls(
            genre=sa_tags.genre,
            mood=sa_tags.mood,
            scene=sa_tags.theme,
            language=sa_tags.language,
            vocal_gender=sa_tags.vocal_gender,
            is_sinking=(
                ["Sinking"] if sa_tags.sinking > sinking_threshold else ["Non-Sinking"]
            ),
        )

    @classmethod
    def from_musicfm_finegrained_tags(
        cls, musicfm_finegrained_tags: MusicFMFinegrainedTagsProto
    ) -> "TagsProto":
        return cls(
            genre=musicfm_finegrained_tags.genre,
            mood=musicfm_finegrained_tags.mood,
            scene=musicfm_finegrained_tags.scene,
            vocal_timbre=musicfm_finegrained_tags.vocal_timbre,
            vocal_gender=musicfm_finegrained_tags.vocal_gender,
        )

    @classmethod
    def from_audio_tags(cls, audio_tags: AudioTagsProto) -> "TagsProto":
        return cls(
            genre=audio_tags.genre,
            genre_extra=audio_tags.genre_extra,
            extra=audio_tags.extra,
            mood=audio_tags.mood,
            scene=audio_tags.scene,
            vocal_gender=audio_tags.vocal_gender,
            vocal_timbre=audio_tags.vocal_timbre,
            language=audio_tags.language,
            instrument=audio_tags.instrument,
        )

    @classmethod
    def _merge_two(cls, tag_a: "TagsProto", tag_b: "TagsProto") -> "TagsProto":
        def merge_list(a: list[str], b: list[str]) -> list[str]:
            lst = a[:]
            for i in b:
                if i not in a:
                    lst.append(i)
            return lst

        tag_a_dict = tag_a.to_dict()
        tag_b_dict = tag_b.to_dict()
        tag_merged_dict = {
            k: merge_list(tag_a_dict[k], tag_b_dict[k]) for k in tag_a_dict
        }
        return cls.from_dict(tag_merged_dict)

    @classmethod
    def merge(cls, *tags_proto: "TagsProto") -> "TagsProto":
        return reduce(cls._merge_two, tags_proto)

    @classmethod
    def _fill_two(cls, tag_a: "TagsProto", tag_b: "TagsProto") -> "TagsProto":
        def fill_list(a: list[str], b: list[str]) -> list[str]:
            if (a == [] or a == [_EMPTY_TAG]) and (b != [] and b != [_EMPTY_TAG]):
                lst = b
            else:
                lst = a
            return lst

        tag_a_dict = tag_a.to_dict()
        tag_b_dict = tag_b.to_dict()
        tag_filled_dict = {
            k: fill_list(tag_a_dict[k], tag_b_dict[k]) for k in tag_a_dict
        }
        return cls.from_dict(tag_filled_dict)

    @classmethod
    def fill(cls, *tags_proto: "TagsProto") -> "TagsProto":
        return reduce(cls._fill_two, tags_proto)

    def remove_oov_tags(
        self, vocab: StyleTagVocab
    ) -> tuple["TagsProto", dict[str, list[str]]]:
        x = {}
        oov = {}
        for category, tags in self.to_dict().items():
            norm_category = "genre" if category == "genre_extra" else category
            filtered_tags = []
            for tag in tags:
                if tag in vocab.token_to_id[norm_category]:
                    filtered_tags.append(tag)
                else:
                    if (
                        tag == _EMPTY_TAG
                    ):  # empty tags are valid and will be added back by `fill_empty_inplace`, no need to log
                        continue
                    if category not in oov:
                        oov[category] = []
                    oov[category].append(tag)
            x[category] = filtered_tags
        return self.__class__.from_dict(x), oov

    def extend_extra_inplace(self) -> "TagsProto":
        genre = self.genre
        extra = copy.deepcopy(self.extra)  # create a new list for `extend`

        if (self.genre_extra == [_EMPTY_TAG]) and (genre != [_EMPTY_TAG]):
            self.genre_extra = copy.deepcopy(genre)  # point to a different list
        # when genre include DJ/MC, will add Grassroots/Tuhai
        if set(genre) & set(["DJ", "MC"]):
            extra.extend(["Grassroots", "Tuhai"])
        # when lang is non-vocal and extra do not include Canned Music, will add non-canned music (only for sft)
        if set(self.language) & set(["Instrumental/Non-Vocal"]):
            if not set(extra) & set(["Canned Music"]):
                extra.extend(["Non-Canned Music"])
        # when extra include Fashionable, will add non-nostalgic
        if set(extra) & set(["Fashionable"]):
            extra.extend(["Non-Nostalgic"])
        # when genre include sone special genre, will add non-nostalgic
        if set(genre) & set(
            [
                "Indie Pop",
                "Indie Folk",
                "Indie Rock",
                "Alternative/Indie",
                "Electropop",
                "Hip Hop",
                "Hip Hop/Rap",
                "Contemporary R&B",
                "Neo Soul",
                "Neo Funk",
            ]
        ):
            extra.extend(["Non-Nostalgic"])
        # when scene include sone special scene, will add non-nostalgic
        if set(self.scene) & set(
            [
                "Vlog/DailyLife",
                "Beauty/Fashion",
                "Transition",
                "Nightclub",
                "Marketplace",
                "Sport",
                "Game",
                "Running",
                "Danceable",
            ]
        ):
            extra.extend(["Non-Nostalgic"])
        extra = _dedup_with_order(extra)  # new list

        if len(extra) > 1 and _EMPTY_TAG in extra:
            extra.remove(_EMPTY_TAG)
        if set(extra) & set(["Grassroots"]):
            self.is_sinking = ["Sinking"]

        self.extra = extra

        return self

    def fill_empty_inplace(self) -> "TagsProto":
        """Fill empty tags with an "empty" token."""
        for k, v in self.to_dict().items():
            if not v:
                setattr(self, k, [_EMPTY_TAG])
        return self

    def standardize_inplace(self, vocab: Optional[StyleTagVocab] = None) -> "TagsProto":
        """Standardize the tags into the standard format. If `vocab` is provided, perform subword matching as well."""

        def subword_match(tag: str, vocab_words) -> str:
            # Hard-coded remap
            if tag in ["No Mood", "Others", "Empty Tempo"]:
                return "Other"
            if tag in vocab_words:
                return tag
            count = {}
            for vocab_word in vocab_words:
                subwords_tag = _split_by_separators(tag)
                subwords_vocab = _split_by_separators(vocab_word)
                n_overlaps = len(set(subwords_tag) & set(subwords_vocab))
                if n_overlaps > 0:
                    count[vocab_word] = n_overlaps
            if count:  # pick the vocab_word that has the highest count
                return max(count, key=count.get)
            return tag

        for category, tags in self.to_dict().items():
            if category == "instrument":  # instrument tags are not standardized, skip
                continue
            norm_category = "genre" if category == "genre_extra" else category
            tags = [_standardize_tag(tag) for tag in tags]
            if vocab:
                tags = [
                    subword_match(tag, vocab.token_to_id[norm_category]) for tag in tags
                ]
            setattr(self, category, tags)
        return self


def _normalize_tags(tags: Union[str, list[str]]) -> list[str]:
    def strip_all(tags: list[str]) -> list[str]:
        tags = [t.strip() for t in tags]
        return [t for t in tags if t]

    if not tags:
        return []
    if isinstance(tags, str):
        return strip_all(tags.split(","))
    return strip_all(tags)


def _standardize_tag(tag: str) -> str:
    def capitalize_word(word: str):
        # Make the first letter uppercase
        if word:  # Check if the string isn't empty
            # keep words that are all uppercase
            if word.upper() == word:
                return word
            # Make the first letter uppercase
            return word[0].upper() + word[1:]
        return word  # Return empty string as is

    def process_subword(word: str):
        # Handle slash and ampersand first
        """
        Example:
        1. "Theater/concert" -> "Theater/Concert"
        2. "r&b" -> "R&B"
        3. "Lo-fi" -> "Lo-Fi"
        4. "Sound_effects" -> "Sound_Effects"
        """
        for sep in ["/", "&", "-", "_"]:
            subword = word.split(sep)
            word = sep.join([capitalize_word(w) for w in subword])
        return word

    if tag.strip().lower() in ["other", "others"]:
        return "Other"

    # del "()" in word
    # Example: "Glam Metal\n（Hair Metal，Pop Metal）" -> "Glam Metal"
    words = tag.replace("（", "(").split("(", 1)[0]
    words = [x.strip() for x in words.split(" ")]
    word_list = []
    for word in words:
        if not word:
            continue
        word = process_subword(word)
        word_list.append(word)
    return " ".join(word_list)


def _split_by_separators(input_string: str) -> list[str]:
    result = re.split(r"[ /\-&_,]", input_string)
    result = [item for item in result if item if item]
    return result


def _parse_audio_tags_from_meta(meta: dict) -> Optional[AudioTagsProto]:
    """
    Parse the audio_tags field in metadata.
    """
    audio_tags = meta.get("audio_tags")
    if not audio_tags:
        return None
    return AudioTagsProto.from_dict_with_norm(audio_tags)


def _parse_human_label_from_meta(meta: dict) -> Optional[AudioTagsProto]:
    """
    Parse the human_label field in metadata. Possibly an old annotation format.
    """
    human_label = meta.get("human_label")
    if not human_label:
        return None

    def get_value(key: str) -> list[str]:
        v = human_label.get(key)
        if not v:  # covers both missing key or empty value
            return []
        if isinstance(v, str):
            return v.split(",")
        return v

    return AudioTagsProto.from_dict_with_norm(
        {
            "genre": get_value("label_genre") + get_value("label_subgenre"),
            "genre_extra": get_value("genre_extra"),
            "extra": get_value("extra"),
            "mood": get_value("label_mood"),
            "scene": get_value("label_theme"),
            "instrument": get_value("label_instrument"),
        }
    )


def _parse_llm_tags_from_meta(meta: dict) -> Optional[AudioTagsProto]:
    """
    Parse the llm_tags field in metadata.
    """
    llm_tags = meta.get("llm_tags")
    if not llm_tags:
        return None
    return AudioTagsProto.from_dict_with_norm(llm_tags)


def _parse_sa_tags_from_meta(meta: dict) -> Optional[SATagsProto]:
    """
    Parse the music_tagging field in metadata.
    """
    sa_tags = meta.get("music_tagging")
    sa_gender = meta.get("gender", {}).get("sa_gender")
    if not sa_tags:
        return None
    if sa_gender:
        sa_tags["sa_gender"] = sa_gender
    return SATagsProto.from_dict_with_norm(sa_tags)


def _parse_audio_tags_into_tags_proto(
    meta: dict, fn: Callable[[dict], AudioTagsProto]
) -> Optional[TagsProto]:
    audio_tags = fn(meta)
    if audio_tags:
        if audio_tags.satisfy_filter_standard != "yes":  # add a filter for quality flag
            raise TagError(
                f"Audio tags contains invalid quality flag: {audio_tags.satisfy_filter_standard}"
            )
        audio_tags = TagsProto.from_audio_tags(audio_tags)
    return audio_tags


def _parse_sa_tags_into_tags_proto(
    meta: dict, fn: Callable[[dict], SATagsProto]
) -> Optional[TagsProto]:
    sa_tags = fn(meta)
    if sa_tags:
        sa_tags = TagsProto.from_sa_tags(sa_tags)
    return sa_tags


def _parse_musicfm_finegrained_tags_from_meta(
    meta: dict,
) -> Optional[MusicFMFinegrainedTagsProto]:
    """
    Parse the musicfm_finegrained_tags field in metadata.
    """
    musicfm_finegrained_tags = meta.get("musicfm_finegrained_tagging", {}).get(
        "finegrained_tags"
    )
    if not musicfm_finegrained_tags:
        return None
    return MusicFMFinegrainedTagsProto.from_dict_with_norm(musicfm_finegrained_tags)


def _parse_musicfm_finegrained_tags_into_tags_proto(
    meta: dict, fn: Callable[[dict], MusicFMFinegrainedTagsProto]
) -> Optional[TagsProto]:
    musicfm_finegrained_tags = fn(meta)
    if musicfm_finegrained_tags:
        musicfm_finegrained_tags = TagsProto.from_musicfm_finegrained_tags(
            musicfm_finegrained_tags
        )
    return musicfm_finegrained_tags


# The labels should match the vocab
_TEMPO_LABEL_RANGES = {
    "Grave": (0, 40),
    "Largo": (40, 60),
    "Adagio": (60, 70),
    "Andante": (70, 90),
    "Moderato": (90, 110),
    "Allegro": (110, 140),
    "Vivace": (140, 160),
    "Presto": (160, 200),
}


def _tempo_to_label(tempo: Optional[int]) -> str:
    if tempo is None or tempo < 0:
        return ""
    for label, (low, high) in _TEMPO_LABEL_RANGES.items():
        if low <= tempo < high:
            return label
    return "empty tempo"


def _parse_insts_to_tags_proto(insts: Optional[list[str]]) -> Optional[TagsProto]:
    if not insts:
        return None
    return TagsProto(instrument=insts)


def _parse_tempo_to_tags_proto(tempo: Optional[float]) -> Optional[TagsProto]:
    if tempo is None or tempo < 0:
        return None
    return TagsProto(tempo=[_tempo_to_label(tempo)])


def _parse_key_mode_to_tags_proto(key_mode: Optional[str]) -> Optional[TagsProto]:
    if not key_mode:
        return None
    key, mode = key_mode.split(":")
    mode = {  # remap to match the vocab
        "Major": "Natural_Major",
        "Minor": "Natural_Minor",
    }.get(mode, mode)
    return TagsProto(key=[key], mode=[mode])


def _dedup_with_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
