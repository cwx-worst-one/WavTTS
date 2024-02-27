from deprecated import deprecated
from collections import Counter
from recipes.bigmusic.datasets.transforms.lyrics_segment_gt import concat_segment

# Section alignment algorithm
def get_valid_words(segment):
    valid_words = []
    current_time = segment.start * 1000
    words = segment.words
    for word in words:
        if word['end_time'] > current_time:
            valid_words.append(word)
            current_time = word['end_time']
    return valid_words

def get_word_indices(segment):
    valid_words = get_valid_words(segment)
    current_index = 0
    text = segment.text.lower()
    for word in valid_words:
        index = text.find(word['text'], current_index)
        if index == -1:
            # print(f'Could not find word {word["text"]} in "{text}", {current_index}')
            # Edge case - numbers are spelled out. let's just set index +=1
            word['index'] = current_index + 1
            current_index = current_index + 1
        else:
            word['index'] = index
            current_index = index + len(word['text'])
    return valid_words

def find_section_index(section_labels, time):
    for idx, section in enumerate(section_labels):
        start, end = section['interval']
        if time >= start and time <= end:
            return idx
    else:
        return -1

def find_word_index(word_indices, target_time, start_index):
    for idx, word in enumerate(word_indices[start_index:]):
        if word['end_time']/1000 > target_time: # TODO: figure out if start time or end time aligns better
            return idx+start_index
    else:
        return None
    
def merge_intervals(music_structures):
    if len(music_structures) == 0: return []
    merged_structures = []
    last_structure = music_structures[0]
    for structure in music_structures[1:]:
        if structure['funct_name'] == last_structure['funct_name']:
            # merge
            start, end = last_structure['interval'][0], structure['interval'][1]
            last_structure['interval'] = [start, end]
            last_structure['duration'] = end - start
            last_structure['res_duration'] = structure['res_duration']
        else:
            merged_structures.append(last_structure)
            last_structure = structure
    merged_structures.append(last_structure)
    return merged_structures


@deprecated(version='0', reason="Use music_structure_to_section_segments, which splits segments by sections instead of words")
def get_section_aligned_text(segment, section_labels):
    # Algorithm: 
    # 1. Given structure label array, find sections overlapping with segment start and end
    # 2. Create word to insertion index map.
    # 3. Get section insertion points
    # 4. Insert sections as special tockens

    # 1.
    sec_start_idx = find_section_index(section_labels, segment.start)
    sec_end_idx = find_section_index(section_labels, segment.end)
    # 2.
    words_with_indices = get_word_indices(segment)

    # 3
    word_index = 0
    insertion_indices = []
    segment_text = segment.text
    for section in section_labels[sec_start_idx:sec_end_idx+1]:
        sec_start_time, sec_end_time = section['interval']
        word_index = find_word_index(words_with_indices, sec_start_time, start_index=word_index)
        if word_index is None: # not found. insert at the end
            insertion_index = len(segment_text)
        else:
            insertion_index = words_with_indices[word_index]['index']
        insertion_indices.append({ 'index': insertion_index, 'section_name': section['funct_name'] })

    # 4
    for insertion_index in reversed(insertion_indices):
        index = insertion_index['index']
        section_name = insertion_index['section_name']
        segment_text = segment_text[:index] + f' <{section_name}> ' + segment_text[index:]
    return segment_text



# V1
def is_valid_musical_structure(music_structure):
    counts = Counter([section['funct_name'] for section in music_structure])
    return counts['chorus'] > 1

def find_largest_overlap(section_labels, segment):
    def get_overlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    
    overlap_to_section_idx = []
    for idx, section in enumerate(section_labels):
        overlap = get_overlap(section['interval'], (segment.start, segment.end))
        overlap_to_section_idx.append((overlap, idx))
    return sorted(overlap_to_section_idx, reverse=True)[0][1]

def music_structure_to_section_segments(metadata, segments, target_durations):
    music_structure = merge_intervals(metadata['music_structure'][0])
    if not is_valid_musical_structure(music_structure): return None
    for idx, segment in enumerate(segments):
        if len(segment.text.strip()) == 0: continue
        section_idx = find_largest_overlap(music_structure, segment)
        section = music_structure[section_idx]
        if 'segments' in section:
            section['segments'].append(segment)
        else:
            section['segments'] = [segment]

    section_segments = []
    for section in music_structure:
        if 'segments' not in section: continue
        segments = section['segments']
        section_name = section['funct_name']
        segment_joined = concat_segment(segments, target_duration=max(target_durations))
        segment_joined.text = f'<{section_name}> {segment_joined.text}'
        section_segments.append(segment_joined)
    return section_segments