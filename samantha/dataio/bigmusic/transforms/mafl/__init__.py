import re
import warnings

from ..song_slice import SongSliceError, transform_utts_to_song_slices_structure
from ..structure import transform_raw_segments
from ..utterance import UttError, parse_utterances
from .freeform_text import format_freeform_text, parse_freeform_text
from .tags import TagError, transform_tags


class MaflError(Exception):
    pass


def parse_asr_lyrics(meta: dict) -> str:
    warnings.warn(
        """This function is deprecated and will be removed in a future version. \
Please use parse_utterances_and_structure_to_lyrics instead.""",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        utts = parse_utterances(meta)
    except UttError as e:
        raise MaflError(str(e))
    raw_segments = meta.get("deepchorus", {}).get("raw_segments")
    if not raw_segments:
        raise MaflError("raw_segments not found")
    structure_tags = transform_raw_segments(raw_segments)
    try:
        song_slice = transform_utts_to_song_slices_structure(
            utts,
            10000000,
            structure_tags["transformed"]["tags"],
            slice_mode="full",
            language="",
        )[0]
    except SongSliceError as e:
        raise MaflError(str(e))
    lines = [phrase.format_text() for phrase in song_slice.phrases]
    return "\n".join(lines)


def parse_utterances_and_structure_to_lyrics(utterances: list, structure: dict) -> str:
    try:
        song_slice = transform_utts_to_song_slices_structure(
            utterances, 10000000, structure["tags"], slice_mode="full", language=""
        )[0]
    except SongSliceError as e:
        raise MaflError(str(e))
    lines = [phrase.format_text() for phrase in song_slice.phrases]
    return "\n".join(lines)


def parse_suno_lyrics(meta: dict) -> str:
    lyrics = meta.get("raw", {}).get("lyrics", "").strip()
    if not lyrics:
        raise MaflError("suno lyrics not found")
    return lyrics


def parse_downloaded_lyrics(meta: dict) -> str:
    lyrics = meta.get("raw", {}).get("downloaded_lyrics", "").strip()
    if not lyrics:
        raise MaflError("downloaded lyrics not found")
    return lyrics


def parse_and_format_freeform_text_legacy(meta: dict) -> str:
    try:
        tags = transform_tags(meta)
    except TagError as e:
        raise MaflError(str(e))
    freeform_text_dict = parse_freeform_text(meta, tags)
    return format_freeform_text(freeform_text_dict, keyword_dropout_rate=0)


START_SECTION = "<SECTION>"
END_SECTION = "</SECTION>"


def preprocess_section_tags(text):
    """Convert section tags into a structured format using <SECTION> and </SECTION> tokens."""
    pattern = re.compile(
        r"\[(.*?)\]"
    )  # Matches section tags like [Intro: piano + guitar]

    def replace(match):
        tag_content = match.group(1)  # Extract content inside brackets
        return (
            f"{START_SECTION} {tag_content} {END_SECTION}"  # Wrap with section markers
        )

    return pattern.sub(replace, text)
