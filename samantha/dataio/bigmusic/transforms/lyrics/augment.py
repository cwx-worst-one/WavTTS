"""
Tools for lyrics augmentation
"""

import re
from collections import defaultdict


def _lyrics_aug_mode_keep(lyrics: str) -> str:
    return lyrics


def _lyrics_aug_mode_all_lowered(lyrics: str) -> str:
    """
    Converts all text to lowercase.
    """
    return lyrics.lower()


def _lyrics_aug_mode_tag_capitalized(lyrics: str) -> str:
    """
    Capitalizes the first letter of text inside bracket tags.
    Example: [verse] -> [Verse]
    """

    def capitalize_tag(match):
        tag_content = match.group(1)
        return f"[{tag_content.capitalize()}]"

    return re.sub(r"\[([^\]]+)\]", capitalize_tag, lyrics)


def _lyrics_aug_mode_tag_capitalized_counted(lyrics: str) -> str:
    """
    Capitalizes and adds count numbers to repeated tags.
    Rules: If tag appears once, no count. If multiple times, count from 1.
    Example: [verse] ... [verse] -> [Verse 1] ... [Verse 2]

    If any tag already ends with a digit, returns original lyrics unchanged.
    """
    # First pass: find all tags and count occurrences (case-insensitive)
    tags = re.findall(r"\[([^\]]+)\]", lyrics)

    # Check if any tag already ends with a digit - if so, return original
    for tag in tags:
        if tag.strip() and tag.strip()[-1].isdigit():
            return lyrics

    tag_counts = defaultdict(int)

    for tag in tags:
        tag_counts[tag.lower()] += 1

    # WORKAROUND: don't count tags that start with "instruments:"
    tag_counts = {
        k: 1 if k.startswith("instruments") else v for k, v in tag_counts.items()
    }

    # Second pass: replace tags with appropriate formatting
    current_counts = defaultdict(int)

    def replace_tag(match):
        tag_content = match.group(1)
        tag_lower = tag_content.lower()

        if tag_counts[tag_lower] > 1:
            # Tag appears multiple times, add count
            current_counts[tag_lower] += 1
            return f"[{tag_content.capitalize()} {current_counts[tag_lower]}]"
        else:
            # Tag appears only once, no count
            return f"[{tag_content.capitalize()}]"

    return re.sub(r"\[([^\]]+)\]", replace_tag, lyrics)


def _lyrics_aug_mode_tag_lowered(lyrics: str) -> str:
    """
    Converts text inside bracket tags to lowercase.
    Example: [VERSE] or [Verse] -> [verse]
    """

    def lowercase_tag(match):
        tag_content = match.group(1)
        return f"[{tag_content.lower()}]"

    return re.sub(r"\[([^\]]+)\]", lowercase_tag, lyrics)


def _lyrics_aug_mode_tag_lowered_not_counted(lyrics: str) -> str:
    """
    Converts text inside bracket tags to lowercase.
    Example: [VERSE] or [Verse] -> [verse]
    """

    def lowercase_tag(match):
        tag_content = match.group(1)
        tag_content = "".join(c for c in tag_content if not c.isdigit())
        return f"[{tag_content.strip().lower()}]"

    return re.sub(r"\[([^\]]+)\]", lowercase_tag, lyrics)


def _lyrics_aug_mode_tag_colon_removed(lyrics: str) -> str:
    """
    Remove colons from text inside bracket tags.
    Example: [verse: singer name] -> [verse]
    """

    def remove_colon_tag(match):
        # normalize colon first
        tag_content = match.group(1)
        tag_content = tag_content.replace("：", ":")
        # take the content before the colon
        tag_content = tag_content.split(":")[0]
        return f"[{tag_content.strip()}]"

    if not (":" in lyrics or "：" in lyrics):
        return lyrics

    return re.sub(r"\[([^\]]+)\]", remove_colon_tag, lyrics)


def _lyrics_aug_mode_linebreak_collapsed(text: str) -> str:
    """
    Collapse multiple consecutive linebreaks into a single linebreak.
    """
    text = text.strip("\n")
    return re.sub(r"\n+", "\n", text)


LYRICS_AUGMENT_MODES = {
    "keep": _lyrics_aug_mode_keep,
    "tag_capitalized": _lyrics_aug_mode_tag_capitalized,
    "tag_capitalized_counted": _lyrics_aug_mode_tag_capitalized_counted,
    "tag_lowered": _lyrics_aug_mode_tag_lowered,
    "tag_lowered_not_counted": _lyrics_aug_mode_tag_lowered_not_counted,
    "tag_colon_removed": _lyrics_aug_mode_tag_colon_removed,
    "all_lowered": _lyrics_aug_mode_all_lowered,
    "linebreak_collapsed": _lyrics_aug_mode_linebreak_collapsed,
}
