from collections import Counter
from typing import List
from string import punctuation
import operator
from functools import reduce
from copy import copy, deepcopy
from itertools import groupby

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

# Section alignment algorithm
def get_word_indices(segment):
    def get_valid_words(segment):
        valid_words = []
        current_time = segment.start * 1000
        words = deepcopy(segment.words)
        for idx, word in enumerate(words):
            word['word_index'] = idx
            if word['end_time'] > current_time:
                valid_words.append(word)
                current_time = word['end_time']
        return valid_words

    valid_words = get_valid_words(segment)
    current_index = 0
    text = segment.text.lower()
    consecutive_misaligned_counts = [0]
    for word in valid_words:
        word_text = word['text'].lower()
        index = text.find(word_text, current_index)
        if index == -1 or index - current_index > len(word_text):
            # print(f'Could not find word {word["text"]} in "{text}", {current_index}')
            # Edge case - numbers are spelled out. let's just set index +=1
            word['index'] = current_index + 1
            current_index = current_index + 1
            consecutive_misaligned_counts[-1] += 1
        else:
            word['index'] = index
            current_index = index + len(word_text)
            consecutive_misaligned_counts.append(0)
    return valid_words, max(consecutive_misaligned_counts)

def find_word_index(word_indices, target_time, start_index):
    for idx, word in enumerate(word_indices[start_index:]):
        if word['end_time']/1000 > target_time: # TODO: figure out if start time or end time aligns better
            return idx+start_index
    else:
        return None
    
def get_section_aligned_segment(segment, section_labels, max_misaligned=5):
    # Algorithm: 
    # 1. Given structure label array, find sections overlapping with segment start and end
    # 2. Create word to insertion index map.
    # 3. Get section insertion points
    # 4. Insert sections as special tockens

    # def has_overlap(x1, x2, y1, y2): return x1 <= y2 and y1 <= x2 # equal time does not count
    def has_overlap(x1, x2, y1, y2): return x1 < y2 and y1 < x2
    segment = deepcopy(segment)

    words_with_indices, misaligned_count = get_word_indices(segment)
    if max_misaligned is not None and misaligned_count > max_misaligned: 
        print('misaligned count', misaligned_count, segment.text)
        return None
    
    word_index = 0
    insertion_indices = []
    segment_text = segment.text
    for section in section_labels:
        sec_start_time, sec_end_time = section['interval']
        if not has_overlap(segment.inst_start, segment.inst_end, sec_start_time, sec_end_time): 
            continue
        word_index = find_word_index(words_with_indices, sec_start_time, start_index=word_index)
        if word_index is None: # not found. insert at the end
            insertion_index = len(segment_text)
            abs_word_index = len(segment.words)-1
        else:
            insertion_index = words_with_indices[word_index]['index']
            abs_word_index = words_with_indices[word_index]['word_index']
        # print('section start', sec_start_time, word_index, segment.words[abs_word_index], section['label'])
        insertion_indices.append({ 'text_index': insertion_index, 'word_index': abs_word_index, 'section_name': section['label'], 'sec_start_time': sec_start_time })

    for insertion_index in reversed(insertion_indices):
        text_index = insertion_index['text_index']
        section_name = insertion_index['section_name']
        segment_text = segment_text[:text_index] + f' <{section_name}> ' + segment_text[text_index:]
        section_word = {'text': f' <{section_name}> ',
        'confidence': 0.0,
        'start_time': insertion_index['sec_start_time']*1000,
        'end_time': insertion_index['sec_start_time']*1000,
        'phoneme': ""}

        segment.words.insert(insertion_index['word_index'], section_word)
    segment.text = segment_text
    return segment

# converts old music_structure to deepchorus structure
def format_music_structure(music_structures):
    if len(music_structures) == 0: return []
    def to_deepchorus_structure(s):
        return { 'interval': s['interval'], 'duration': s['duration'], 'label': s['funct_name'] }
    music_structures = list(map(to_deepchorus_structure, music_structures))
    merged_structures = []
    last_structure = music_structures[0]
    for structure in music_structures[1:]:
        if structure['label'] == last_structure['label']:
            # merge
            start, end = last_structure['interval'][0], structure['interval'][1]
            last_structure['interval'] = [start, end]
            last_structure['duration'] = end - start
        else:
            merged_structures.append(last_structure)
            last_structure = structure
    merged_structures.append(last_structure)
    return merged_structures


# V1
def is_valid_musical_structure(music_structure):
    counts = Counter([section['label'] for section in music_structure])
    return counts['chorus'] > 1

def find_largest_overlap(section_labels, segment):
    def get_overlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    
    overlap_to_section_idx = []
    for idx, section in enumerate(section_labels):
        overlap = get_overlap(section['interval'], (segment.start, segment.end))
        overlap_to_section_idx.append((overlap, idx))
    return sorted(overlap_to_section_idx, reverse=True)[0][1]

def align_structure_to_closest_segment(structures, segments):
    def find_closest_segment(target_time, segments, is_start):
        if is_start and target_time == 0: return None, 0
        diff, seg, time = 10000, None, -1
        for s in segments:
            if abs(s.start - target_time) < diff:
                diff, seg, time = abs(s.start - target_time), s, s.start
            if abs(s.end - target_time) < diff:
                diff, seg, time = abs(s.end - target_time), s, s.end
            # if is_start and abs(s.start - target_time) < diff:
            #     diff, seg, time = abs(s.start - target_time), s, s.start
            # if not is_start and abs(s.end - target_time) < diff:
            #     diff, seg, time = abs(s.end - target_time), s, s.end
        return seg, time

    structures = deepcopy(structures)
    for structure in structures:
        orig_start, orig_end = structure['interval']
        start_s, start_time = find_closest_segment(orig_start, segments, is_start=True)
        end_s, end_time = find_closest_segment(orig_end, segments, is_start=False)
        new_interval = start_time, end_time
        structure['interval'] = new_interval
    return structures

def gt_lyrics_to_song_structure(segments):
    def _normalize_text(text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", punctuation))
        text = " ".join(text.split(" "))  # remove spaces
        return text

    def is_chorus(segment):
        lyric_line = _normalize_text(segment.text)
        return lyrics_counts[lyric_line] > 1


    lyrics_lines = [_normalize_text(s.text) for s in segments]
    lyrics_counts = Counter(lyrics_lines)
    
    line_matches = []
    for idx, segment in enumerate(segments):
        # check if left and right are chorus. else it's not.
        cur = is_chorus(segment)
        left = is_chorus(segments[idx-1]) if idx > 0 else False
        right = is_chorus(segments[idx+1]) if idx < len(segments) - 1 else False
        is_c = (cur and left) or (cur and right)
        line_matches.append([is_c, segment])
        
    groups = groupby(line_matches, lambda x: x[0])
    sections = []
    for (is_c, group) in groups:
        label = 'chorus' if is_c else 'verse'
        segments = [seg for (_, seg) in group]
        sections.append(
            {
                'label': label,
                'interval': [segments[0].start, segments[-1].end],
                'duration': segments[-1].end - segments[0].start,
#                 'segments': segments
            }
        )
    return sections
import random
def song_structure_from_metadata(metadata, segments, align_section_timings=True):
    if metadata is None: return None
    if 'is_lyrics_gt' in metadata and 'deepchorus' in metadata:
        if random.random() < 0.5:
            song_structure = metadata['deepchorus']['segments']
        else:
            song_structure = gt_lyrics_to_song_structure(segments)
    elif 'is_lyrics_gt' in metadata:
        # print('Lyrics gt')
        song_structure = gt_lyrics_to_song_structure(segments)
    elif 'deepchorus' in metadata:
        # print('Deepchorus')
        song_structure = metadata['deepchorus']['segments']
    elif 'music_structure' in metadata:
        song_structure = metadata['music_structure'][0]
        song_structure = format_music_structure(song_structure)
    else:
        # raise NotImplementedError(f"No music structure found in {metadata.keys()}")
        return None
    if not is_valid_musical_structure(song_structure): return None
    if align_section_timings:
        # print('Song structure before', song_structure)
        song_structure = align_structure_to_closest_segment(song_structure, segments)
        # print('Song structure after', song_structure)
        # print('Segments', segments)
    return song_structure

def group_by_section_start(song_structure, segments, target_duration=None):
    structure_segments = []
    existing_segments = set()
    for structure in song_structure:
        st_start, st_end = structure['interval']
        target_end = st_start + target_duration if target_duration is not None else st_end
        section = []
        for s in segments:
            if s.end > target_end: break
            if s.start >= st_start:
                section.append(s)
        if not section: continue
        structure_segment = concat_segment(section, target_duration)
        if (structure_segment.start, structure_segment.end) not in existing_segments:
            structure_segments.append(structure_segment)
            existing_segments.add((structure_segment.start, structure_segment.end))
    return structure_segments