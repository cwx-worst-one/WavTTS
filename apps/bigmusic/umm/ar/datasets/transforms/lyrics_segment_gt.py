import operator
from collections import Counter
from functools import reduce
from string import punctuation


def _normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    text = " ".join(text.split(" "))  # remove spaces
    return text


def count_lines(lines):
    lyrics_lines = [_normalize_text(line["text"]) for line in lines]
    return Counter(lyrics_lines)


def parse_sections(lines, lyrics_count):
    from apps.bigmusic.umm.ar.datasets.transforms.lyrics_segment import Segment

    lyrics_sections = []
    current_section = []
    prev_lyrics_repeat = None
    for idx, line in enumerate(lines):
        segment = Segment.from_dict(line)
        lyric_line = _normalize_text(segment.text)
        lyrics_repeat = lyrics_count[lyric_line] > 1
        if prev_lyrics_repeat is None:
            prev_lyrics_repeat = lyrics_repeat  # first line
        is_start, is_end = (prev_lyrics_repeat is None), (idx == len(lines) - 1)
        if prev_lyrics_repeat == lyrics_repeat:
            current_section.append(segment)
        if is_end or prev_lyrics_repeat != lyrics_repeat:
            section_name = "chorus" if prev_lyrics_repeat else "verse"
            lyrics_sections.append({"section": section_name, "lyrics": current_section})
            current_section = [segment]
        prev_lyrics_repeat = lyrics_repeat
    return lyrics_sections


def concat_segment(segments, target_duration=None):
    if target_duration is None:
        return reduce(operator.add, segments)
    base_segment = segments[0]
    for segment in segments[1:]:
        if base_segment.duration + segment.duration < target_duration:
            base_segment += segment
    return base_segment


def create_segments_from_sections(lyrics_sections, target_duration=30):
    lyrics_segments = []
    for lyrics_section in lyrics_sections:
        lyrics_segment = concat_segment(
            lyrics_section["lyrics"], target_duration=target_duration
        )
        section_name = lyrics_section["section"]
        lyrics_segment.text = f"<{section_name}> {lyrics_segment.text}"
        lyrics_segments.append(lyrics_segment)
    return lyrics_segments


def lyrics_to_section_segments(lyrics_lines, target_durations=(30,)):
    lyrics_count = count_lines(lyrics_lines)
    lyrics_sections = parse_sections(lyrics_lines, lyrics_count)
    if len(lyrics_sections) > 3:
        lyrics_segments = create_segments_from_sections(
            lyrics_sections, target_duration=max(target_durations)
        )
        return lyrics_segments
    return None
