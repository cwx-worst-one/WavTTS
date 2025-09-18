from copy import deepcopy
from typing import Optional


def transform_raw_segments(
    raw_segments, merge_mode: Optional[str] = "vocal", obtain_confidence: bool = True
) -> dict:
    if merge_mode:
        if merge_mode == "dc2_brutal":
            segments = process_musicfm_structure(raw_segments)
        else:
            segments = _merge_raw_song_structure_by_start_prob(raw_segments, merge_mode)
    else:
        segments = raw_segments
    if len(segments) == 0:
        return None
    return {
        "merged": segments,
        "transformed": _format_deepchorus_structure_tags(segments),
        "confidence": (
            _get_deepchorus_confidence(segments) if obtain_confidence else None
        ),
    }


def apply_post_processing_rules(merged_data):
    if not merged_data:
        return []

    processed_data = [d.copy() for d in merged_data]

    changed_in_pass = True
    while changed_in_pass:
        changed_in_pass = False

        i = 0
        while i < len(processed_data):
            section = processed_data[i]
            is_first = i == 0
            is_last = i == len(processed_data) - 1
            duration = section["interval"][1] - section["interval"][0]

            next_section = processed_data[i + 1] if not is_last else None

            if section["label"] == "intro":
                if section.get("is_protected_intro"):
                    pass
                elif not is_first:
                    if duration < 2 and next_section:
                        next_section["interval"][0] = section["interval"][0]
                        if "is_protected_intro" in section:
                            next_section["is_protected_intro"] = True
                        del processed_data[i]
                        changed_in_pass = True
                        continue
                    else:
                        if is_last or (next_section and next_section["label"] == "end"):
                            section["label"] = "outro"
                        else:
                            section["label"] = "inst"
                        changed_in_pass = True

            elif section["label"] == "outro":
                if not is_last:
                    if next_section and next_section["label"] == "end":
                        pass
                    elif duration < 2 and next_section:
                        next_section["interval"][0] = section["interval"][0]
                        del processed_data[i]
                        changed_in_pass = True
                        continue
                    else:
                        if is_first:
                            section["label"] = "intro"
                        else:
                            section["label"] = "inst"
                        changed_in_pass = True

            i += 1

    for section in processed_data:
        if "is_protected_intro" in section:
            del section["is_protected_intro"]

    return processed_data


def process_musicfm_structure(data):
    if not data:
        return []

    i = 0
    while i < len(data):
        if data[i]["label"] == "silence":
            if i + 1 < len(data):
                next_section = data[i + 1]
                if next_section["label"] == "intro":
                    next_section["is_protected_intro"] = True
                next_section["interval"][0] = data[i]["interval"][0]
                del data[i]
            else:
                data[i]["label"] = "end"
                i += 1
        else:
            i += 1

    merged_data = []
    if data:
        i = 0
        while i < len(data):
            current_section = data[i].copy()

            j = i + 1
            funct_prob_sum = current_section.get("funct_prob", 0)
            start_prob_sum = current_section.get("start_prob", 0)
            count = 1

            while j < len(data) and data[j]["label"] == current_section["label"]:
                current_section["interval"][1] = data[j]["interval"][1]
                funct_prob_sum += data[j].get("funct_prob", 0)
                start_prob_sum += data[j].get("start_prob", 0)
                if data[j].get("is_protected_intro"):
                    current_section["is_protected_intro"] = True
                count += 1
                j += 1

            if count > 1:
                current_section["funct_prob"] = funct_prob_sum / count
                current_section["start_prob"] = start_prob_sum / count

            merged_data.append(current_section)
            i = j

    final_data = apply_post_processing_rules(merged_data)

    for section in final_data:
        if "funct_prob" in section:
            section["funct_prob"] = round(section["funct_prob"], 3)
        if "start_prob" in section:
            section["start_prob"] = round(section["start_prob"], 3)

    if final_data and final_data[-1].get("label") == "end":
        if len(final_data) > 1:
            final_data[-2]["interval"][1] = final_data[-1]["interval"][1]
        final_data.pop()

    return final_data


def _get_deepchorus_confidence(segments) -> float:
    # in some cases, all segments are filtered
    try:
        boundary = [b["start_prob"] for b in segments]
        function = [f["funct_prob"] for f in segments]
        confidence = 0.7 * (sum(boundary) / len(boundary)) + 0.3 * (
            sum(function) / len(function)
        )
    except Exception:
        confidence = 0
    return confidence


def _merge_raw_song_structure_by_start_prob(raw_segments, mode="vocal"):
    """
    merge raw segments from deepchorus
    it usually contains more segments than the default deepchorus segments
    after merge, section name confidence ('funct_prob') is updated using the duration-weighted average from raw segments
    """
    raw_segments = deepcopy(raw_segments)  # do not mutate it in-place
    sections_processed = []
    sections_processed.append(raw_segments[0])

    if mode == "vocal":
        for i in range(1, len(raw_segments)):

            # if start_prob > 0.3, split here anyway
            if raw_segments[i]["start_prob"] > 0.3:
                sections_processed[-1]["interval"][1] = raw_segments[i]["interval"][0]
                sections_processed.append(raw_segments[i])

            # if section_name changes, either start_prob > 0.1 or funct_prob > 0.85, split here
            elif raw_segments[i]["label"] != raw_segments[i - 1]["label"] and (
                raw_segments[i]["start_prob"] > 0.1
                or raw_segments[i]["funct_prob"] > 0.85
            ):
                sections_processed[-1]["interval"][1] = raw_segments[i]["interval"][0]
                sections_processed.append(raw_segments[i])

    elif (
        mode == "inst"
    ):  # for instrumental, all prob values are not quite reliable. Almost keep all the raw segments (unless boundary
        # confidence is extremely low)

        for i in range(1, len(raw_segments)):
            # if start_prob > 0.01, split here anyway
            if raw_segments[i]["start_prob"] > 0.01:
                sections_processed[-1]["interval"][1] = raw_segments[i]["interval"][0]
                sections_processed.append(raw_segments[i])

    else:
        raise ValueError("mode must be vocal or inst")  # no need to catch this error

    sections_processed[-1]["interval"][1] = raw_segments[-1]["interval"][1]

    def calculate_weighted_funct_prob(intervals, new_interval):
        total_duration = 0
        weighted_funct_prob_sum = 0
        for interval in intervals:
            start, end = interval["interval"]
            if start >= new_interval[0] and end <= new_interval[1]:
                duration = end - start
                total_duration += duration
                weighted_funct_prob_sum += duration * interval["funct_prob"]
        return weighted_funct_prob_sum / total_duration if total_duration else 0

    for section in sections_processed:
        section["funct_prob"] = calculate_weighted_funct_prob(
            raw_segments, section["interval"]
        )

    return sections_processed


_SECTION_TAG_SEP = "#"


def _add_count_to_section_tag(section_tag: str, count: int) -> str:
    if _SECTION_TAG_SEP in section_tag:
        return section_tag
    return f"{section_tag}{_SECTION_TAG_SEP}{str(count)}"


def _format_deepchorus_structure_tags(segments: list[dict]) -> dict:
    """Convert the deepchorus field in metadata into the format of [{'tag': tag, 'start_time': sec, 'end_time': sec}].
    A number will be appended to the tag to differentiate adjacent tags with the same name.
    """
    structure_tags = []
    for count, segment in enumerate(segments):
        structure_tag = {
            "tag": _add_count_to_section_tag(segment["label"], count),
            "start_time": segment["interval"][0],
            "end_time": segment["interval"][1],
        }
        structure_tags.append(structure_tag)
    return {"tags": structure_tags, "confidence": _get_deepchorus_confidence(segments)}
