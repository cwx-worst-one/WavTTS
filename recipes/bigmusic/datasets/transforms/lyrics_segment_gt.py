from collections import Counter
from typing import List
from string import punctuation
import operator
from functools import reduce
from copy import copy

def _normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    text = " ".join(text.split(" "))  # remove spaces
    return text

def count_lines(segments):
    lyrics_lines = [_normalize_text(s.text) for s in segments]
    return Counter(lyrics_lines)

def parse_sections(segments, lyrics_count):
    lyrics_sections = []
    current_section = []
    prev_lyrics_repeat = None
    for idx, segment in enumerate(segments):
        lyric_line = _normalize_text(segment.text)
        lyrics_repeat = lyrics_count[lyric_line] > 1
        if prev_lyrics_repeat is None: prev_lyrics_repeat = lyrics_repeat # first line
        is_end = (idx == len(segments) - 1)
        if prev_lyrics_repeat == lyrics_repeat:
            current_section.append(segment)
        if is_end or (prev_lyrics_repeat != lyrics_repeat and len(current_section) > 1):
            section_name = 'chorus' if prev_lyrics_repeat and lyrics_repeat else 'verse'
            lyrics_sections.append({
                'section': section_name,
                'lyrics': current_section
            })
            current_section = [segment]
        prev_lyrics_repeat = lyrics_repeat
    return lyrics_sections

def concat_segment(segments, target_duration=None):
    if target_duration is None:
        return reduce(operator.add, segments)
    base_segment = copy(segments[0])
    for segment in segments[1:]:
        if segment.end - base_segment.start > target_duration:
            break
        base_segment += segment
    base_segment.target_duration = target_duration
    return base_segment

def create_segments_from_sections(lyrics_sections, target_duration=30):
    lyrics_segments = []
    for idx, lyrics_section in enumerate(lyrics_sections):
        lyrics_segment = concat_segment(lyrics_section['lyrics'], target_duration=target_duration)
        section_name = lyrics_section['section']
        lyrics_segment.text = f'<{section_name}> {lyrics_segment.text}'
        lyrics_segments.append(lyrics_segment)
    return lyrics_segments

def gt_lyrics_to_section_segments(segments, target_durations=(30,)):
    lyrics_count = count_lines(segments)

    for k,count in lyrics_count.most_common(2):
        if count < 2: return None
    lyrics_sections = parse_sections(segments, lyrics_count)
    if len(lyrics_sections) < 4: return None
    lyrics_segments = create_segments_from_sections(lyrics_sections, target_duration=max(target_durations))
    return lyrics_segments