# Section alignment algorithm
def get_valid_words(segment):
    valid_words = []
    current_time = segment.start * 1000
    words = segment.words
    for word in words:
        if word["end_time"] > current_time:
            valid_words.append(word)
            current_time = word["end_time"]
    return valid_words


def get_word_indices(segment):
    valid_words = get_valid_words(segment)
    current_index = 0
    text = segment.text.lower()
    for word in valid_words:
        index = text.find(word["text"], current_index)
        if index == -1:
            # print(f'Could not find word {word["text"]} in "{text}", {current_index}')
            # Edge case - numbers are spelled out. let's just set index +=1
            word["index"] = current_index + 1
            current_index = current_index + 1
        else:
            word["index"] = index
            current_index = index + len(word["text"])
    return valid_words


def find_section_index(section_labels, time):
    for idx, section in enumerate(section_labels):
        start, end = section["interval"]
        if time >= start and time <= end:
            return idx
    else:
        return -1


def find_word_index(word_indices, target_time, start_index):
    for idx, word in enumerate(word_indices[start_index:]):
        if (
            word["end_time"] / 1000 > target_time
        ):  # TODO: figure out if start time or end time aligns better
            return idx + start_index
    else:
        return None


def merge_intervals(music_structures):
    if len(music_structures) == 0:
        return []
    merged_structures = []
    last_structure = music_structures[0]
    for structure in music_structures[1:]:
        if structure["funct_name"] == last_structure["funct_name"]:
            # merge
            start, end = last_structure["interval"][0], structure["interval"][1]
            last_structure["interval"] = [start, end]
            last_structure["duration"] = end - start
            last_structure["res_duration"] = structure["res_duration"]
        else:
            merged_structures.append(last_structure)
            last_structure = structure
    merged_structures.append(last_structure)
    return merged_structures


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
    for section in section_labels[sec_start_idx : sec_end_idx + 1]:
        sec_start_time, sec_end_time = section["interval"]
        word_index = find_word_index(
            words_with_indices, sec_start_time, start_index=word_index
        )
        if word_index is None:  # not found. insert at the end
            insertion_index = len(segment_text)
        else:
            insertion_index = words_with_indices[word_index]["index"]
        insertion_indices.append(
            {"index": insertion_index, "section_name": section["funct_name"]}
        )

    # 4
    for insertion_index in reversed(insertion_indices):
        index = insertion_index["index"]
        section_name = insertion_index["section_name"]
        segment_text = (
            segment_text[:index] + f" <{section_name}> " + segment_text[index:]
        )
    return segment_text
