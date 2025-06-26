from copy import deepcopy
from typing import Optional


def transform_raw_segments(
    raw_segments, merge_mode: Optional[str] = "vocal", obtain_confidence: bool = True
) -> dict:
    if merge_mode:
        segments = _merge_raw_song_structure_by_start_prob(raw_segments, merge_mode)
    else:
        segments = raw_segments
    return {
        "merged": segments,
        "transformed": _format_deepchorus_structure_tags(segments),
        "confidence": (
            _get_deepchorus_confidence(segments) if obtain_confidence else None
        ),
    }


def _get_deepchorus_confidence(segments) -> float:
    boundary = [b["start_prob"] for b in segments]
    function = [f["funct_prob"] for f in segments]
    return 0.7 * (sum(boundary) / len(boundary)) + 0.3 * (sum(function) / len(function))


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


def _get_deepchorus_score(segments):
    boundary = [b["start_prob"] for b in segments]
    function = [f["funct_prob"] for f in segments]
    return 0.7 * (sum(boundary) / len(boundary)) + 0.3 * (sum(function) / len(function))


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
    return {"tags": structure_tags, "confidence": _get_deepchorus_score(segments)}
