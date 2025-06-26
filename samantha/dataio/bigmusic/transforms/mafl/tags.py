import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Optional, Union

from ...utils.dcbase import DCBase

__all__ = ["TagError", "transform_tags", "validate_tags"]


_DEFAULT_SINKING_THRESHOLD = 0.51


class TagError(Exception):
    pass


def transform_tags(meta: dict) -> dict:
    audio_tags = _parse_audio_tags_into_tags_proto(meta, _parse_audio_tags_from_meta)
    human_tags = _parse_audio_tags_into_tags_proto(meta, _parse_human_label_from_meta)
    llm_tags = _parse_audio_tags_into_tags_proto(meta, _parse_llm_tags_from_meta)
    sa_tags = _parse_sa_tags_into_tags_proto(meta, _parse_sa_tags_from_meta)
    raw = _parse_raw_into_tags_proto(meta, _parse_raw_from_meta)
    # Arrange tag sources from high priority to low priority
    valid_tags = list(filter(None, [human_tags, audio_tags, llm_tags, sa_tags, raw]))
    if not valid_tags:
        raise TagError("No valid tags")
    return TagsProto.merge(*valid_tags).to_dict()


def validate_tags(style_tags: dict) -> None:
    tags_proto = TagsProto.from_dict(style_tags)
    if any(
        lang
        for lang in tags_proto.language
        if lang not in ["English", "Chinese", "Cantonese", "Janpanese", "Sichuanese"]
    ):
        raise TagError(f"Audio tags contains invalid language: {tags_proto.language}")
    if set(["Chinese", "Cantonese"]).issubset(tags_proto.language):
        raise TagError(f"Audio tags contains invalid language: {tags_proto.language}")


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
    sinking: float = 0.0

    @classmethod
    def from_dict_with_norm(cls, d: dict) -> "SATagsProto":
        return SATagsProto(
            genre=_normalize_tags(d.get("Genre20", {}).get("result")),
            mood=_normalize_tags(d.get("Mood", {}).get("result")),
            theme=_normalize_tags(d.get("Theme", {}).get("result")),
            language=_normalize_tags(d.get("Language", {}).get("result")),
            sinking=d.get("MusicLowQuality", {}).get("Sinking", 0.0),
        )


@dataclass
class RawProto(DCBase):
    song_name: str = ""
    artist: str = ""
    track_genres: list[str] = field(default_factory=list)
    album_genres: list[str] = field(default_factory=list)
    album_descriptors: list[str] = field(default_factory=list)


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

    # raw
    song_name: list[str] = field(default_factory=list)
    artist: list[str] = field(default_factory=list)
    track_genres: list[str] = field(default_factory=list)
    album_genres: list[str] = field(default_factory=list)
    album_descriptors: list[str] = field(default_factory=list)

    @classmethod
    def from_sa_tags(
        cls, sa_tags: SATagsProto, sinking_threshold: float = _DEFAULT_SINKING_THRESHOLD
    ) -> "TagsProto":
        return cls(
            genre=sa_tags.genre,
            mood=sa_tags.mood,
            scene=sa_tags.theme,
            language=sa_tags.language,
            is_sinking=(
                ["Sinking"] if sa_tags.sinking > sinking_threshold else ["Non-Sinking"]
            ),
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
    def from_raw(cls, raw: RawProto) -> "TagsProto":
        return cls(
            song_name=raw.song_name,
            artist=raw.artist,
            track_genres=raw.track_genres,
            album_genres=raw.album_genres,
            album_descriptors=raw.album_descriptors,
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


def _normalize_tags(tags: Union[str, list[str]]) -> list[str]:
    def strip_all(tags: list[str]) -> list[str]:
        tags = [t.strip() for t in tags]
        return [t for t in tags if t]

    if not tags:
        return []
    if isinstance(tags, str):
        return strip_all(tags.split(","))
    return strip_all(tags)


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
    if not sa_tags:
        return None
    return SATagsProto.from_dict_with_norm(sa_tags)


def _parse_raw_from_meta(meta: dict) -> Optional[RawProto]:
    """
    Parse the raw field in metadata.
    """

    def norm_field_to_list(tag) -> list[str]:
        if isinstance(tag, str):
            return [tag]
        elif isinstance(tag, list):
            if not tag:
                return []
            if isinstance(tag[0], str):
                return tag
            return reduce(operator.add, tag)
        else:
            return []

    raw = meta.get("raw")
    if not raw:
        return None

    return RawProto(
        # NOTE: Skip song_name and artist for now
        # song_name=[raw.get("song_name", "")],
        song_name=[],
        # artist=[raw.get("Artist", "")],
        artist=[],
        track_genres=norm_field_to_list(raw.get("track_genres", [])),
        album_genres=norm_field_to_list(raw.get("album_genres", [])),
        album_descriptors=norm_field_to_list(raw.get("album_descriptors", [])),
    )


def _parse_audio_tags_into_tags_proto(
    meta: dict, fn: Callable[[dict], AudioTagsProto]
) -> Optional[TagsProto]:
    audio_tags = fn(meta)
    if audio_tags:
        if audio_tags.satisfy_filter_standard != "yes":  # add a filter for quality flag
            audio_tags.satisfy_filter_standard = "no"
        audio_tags = TagsProto.from_audio_tags(audio_tags)
    return audio_tags


def _parse_sa_tags_into_tags_proto(
    meta: dict, fn: Callable[[dict], SATagsProto]
) -> Optional[TagsProto]:
    sa_tags = fn(meta)
    if sa_tags:
        sa_tags = TagsProto.from_sa_tags(sa_tags)
    return sa_tags


def _parse_raw_into_tags_proto(
    meta: dict, fn: Callable[[dict], RawProto]
) -> Optional[TagsProto]:
    raw = fn(meta)
    if raw:
        raw = TagsProto.from_raw(raw)
    return raw
