import copy
import logging
import re
from typing import Optional, Union

import torch

from recipes.bigmusic.datasets.utils.zh_meta import (
    SongSlice,
    transform_utts_to_song_slices_structure,
)
from recipes.bigmusic.utils.common_utils import record_time
from recipes.datasets.mcc.sami_tokenizer import Phrase
from recipes.datasets.mcc.sami_tokenizer import section_tags as SECTION_TAGS

from .asr import ASRTransform, ForceAlignTransform
from .sami_mir import Vocal2MidiTransform, DeepchorusTransform
from .sami_phoneme import SamiTextToPhonemeTransform


class RemixMIRTransformError(Exception):
    pass


class RemixMIRTransform:
    def __init__(self, language: str = "zh-CN", sample_rate: Optional[int] = None):
        self.asr_transform = ASRTransform(language, sample_rate)
        self.forcealign_transform = ForceAlignTransform(sample_rate)
        self.vocal2midi_transform = Vocal2MidiTransform(sample_rate)
        self.deepchorus_transform = DeepchorusTransform(sample_rate)
        self.text2phone_transform = SamiTextToPhonemeTransform()
        self.language = language
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")  # independent logger for each instance

    def __call__(self,
        wav: Union[torch.Tensor, bytes],
        text: Optional[str] = None,
        sample_rate: Optional[int] = None,
        text_replace: Optional[str] = None,
        duration_scaling_factor: float = 1.0,
        offset: float = 0.0,
        mode: str = "regular",
    ) -> dict:
        """
        Args:
            wav: Audio in bytes or torch.Tensor.
            text: Lyrics of the song. Do NOT contain any section tag. If it is not provided, ASR will be used to extract lyrics.
            sample_rate: Sample rate of the audio. If it is not provided, use self.sample_rate as the sample rate.
            mode:
                - "regular": regular remix_mir
                    - text_replace: If given, the text will be replaced with the phonemes of text_replace.
                    - duration_scaling_factor: Scale the duration of the audio by this factor.
                    - offset: Offset the MIR result by this value in seconds.
                - "dup_var": remix_mir with a duplicated MIR result with a different text.
                    - text_replace: The value must be provided in this mode. The text will be used in the concatenated MIR result.
                    - duration_scaling_factor: Must be 1.0
                    - offset: The length of the audio in seconds, which will be used as the offset of the duplicated MIR result.
        Return: A dict with "utterance", "notes", and "structure".
        """
        sample_rate = sample_rate or self.sample_rate

        record = {}
        if text:
            anchors = _extract_section_anchors_from_lyrics(text)
            text = _remove_bracketed_text(text)
            with record_time(record, "forcealign"):
                # _remove_bracketed_text removes section tags from text at this step.
                # Please note that singer tags will not be removed, therefore, do NOT use any lyrics that contain singer tags.
                # This is a hack. A better solution would be also returning the lyrics without special tags from the preprocess module.
                # Remove the call to _remove_bracketed_text if we can guarantee the text does not contain any special tags.
                forcealign_resp = self.forcealign_transform(wav, text, sample_rate)
                utterances = forcealign_resp["utterances"]
        else:
            anchors = []
            with record_time(record, "asr"):
                asr_resp = self.asr_transform(wav, sample_rate)
                utterances = asr_resp["utterances"]
        with record_time(record, "vocal2midi"):
            vocal2midi = self.vocal2midi_transform(wav, sample_rate)
            notes = vocal2midi["notes"]
        with record_time(record, "deepchorus"):
            if anchors:
                segment = _get_deepchorus_segment_from_utterances_and_anchors(utterances, anchors, wav, sample_rate)
            else:
                deepchorus = self.deepchorus_transform(wav, sample_rate)
                segment = _reformat_structure_segments(deepchorus["segment"])

        self.logger.info(f"Enter mode {mode}")

        # NOTE: For all utterance replacement, the operation does not replace "words" for each utterance
        if mode == "regular":
            if text_replace:
                with record_time(record, "text2phone_replace"):
                    lines_replace = _remove_bracketed_text(text_replace).split("\n")
                    if len(lines_replace) != len(utterances):
                        raise RemixMIRTransformError(f"Number of lines in text_replace ({len(lines_replace)}) does not match the number of utterances ({len(utterances)})")
                    for text, utt in zip(lines_replace, utterances):
                        utt["text"] = text
                        utt["phoneme"] = self.text2phone_transform(utt["text"])
            else:
                with record_time(record, "text2phone"):
                    for utt in utterances:
                        utt["phoneme"] = self.text2phone_transform(utt["text"])

            utterances = _scale_and_offset_utterances(utterances, duration_scaling_factor, offset)
            notes = _scale_and_offset_notes(notes, duration_scaling_factor, offset)
            segment = _scale_and_offset_structure(segment, duration_scaling_factor, offset)

            remix_mir = {
                "utterances": utterances,
                "notes": notes,
                "structure": segment,
            }

        elif mode == "dup_var":
            if not text_replace:
                raise RemixMIRTransformError("text_replace must be provided for mode 'dup_var'")
            if duration_scaling_factor != 1.0:
                raise RemixMIRTransformError("duration_scaling_factor must be 1.0 for mode 'dup_var'")
            if offset == 0.0:
                raise RemixMIRTransformError("offset must not be 0.0 for mode 'dup_var'")

            lines_replace = _remove_bracketed_text(text_replace).split("\n")
            if len(lines_replace)!= len(utterances):
                raise RemixMIRTransformError(f"Number of lines in text_replace ({len(lines_replace)}) does not match the number of utterances ({len(utterances)})")

            with record_time(record, "text2phone"):
                for utt in utterances:
                    utt["phoneme"] = self.text2phone_transform(utt["text"])

            with record_time(record, "text2phone_replace"):
                utterances_cont = copy.deepcopy(utterances)
                for text, utt in zip(lines_replace, utterances_cont):
                    utt["text"] = text
                    utt["phoneme"] = self.text2phone_transform(utt["text"])

            utterances_cont = _scale_and_offset_utterances(utterances_cont, duration_scaling_factor, offset)
            notes_cont = _scale_and_offset_notes(notes, duration_scaling_factor, offset)
            segment_cont = _scale_and_offset_structure(segment, duration_scaling_factor, offset)

            remix_mir = {
                "utterances": utterances + utterances_cont,
                "notes": notes + notes_cont,
                "structure": segment + segment_cont,
            }
        
        else:
            raise RemixMIRTransformError(f"Invalid mode: {mode}")

        self.logger.info(f"{self.__class__.__name__}.__call__ execution time: {record}")
        return remix_mir


def transform_remix_mir_to_song_slice(
    utterances: list[dict],
    structure: list[dict],
    language: Optional[str] = None,
) -> SongSlice:
    return transform_utts_to_song_slices_structure(
        utterances=utterances,
        min_duration=0,
        max_duration=100000000,  # this value does not matter in the full mode
        structure_tags=structure,
        slice_mode="full",
        language=language,
    )[0]


def _extract_section_anchors_from_lyrics(lyrics: str) -> list[dict]:
    def is_section_vocal(section_tag: str) -> bool:
        return section_tag in ["verse", "chorus", "bridge"]

    # Make sure there's no change to the available section tags
    assert SECTION_TAGS == ['silence', 'chorus', 'verse', 'bridge', 'inst', 'outro', 'intro']

    anchors = []
    n_vocal_lines_scanned = -1
    for line in lyrics.split("\n"):
        phrase = Phrase.parse(text=line)
        if phrase.has_utterance and phrase.section_tag:
            raise RemixMIRTransformError(f"Lyric line {line} is not correctly formatted")
        if phrase.section_tag:
            if anchors and is_section_vocal(anchors[-1]["tag"]):
                anchors[-1]["end_anchor"] = {
                    "utt_idx": n_vocal_lines_scanned,
                    "time": "end_time",
                }
            if is_section_vocal(phrase.section_tag):
                anchors.append({
                    "tag": phrase.section_tag,
                    "start_anchor": {
                        "utt_idx": n_vocal_lines_scanned + 1,
                        "time": "start_time",
                    },
                })
            else:
                anchors.append({
                    "tag": phrase.section_tag,
                    "start_anchor": None if not anchors else {
                        "utt_idx": n_vocal_lines_scanned,
                        "time": "end_time",
                    },
                    "end_anchor": {
                        "utt_idx": n_vocal_lines_scanned + 1,
                        "time": "start_time",
                    }
                })
        else:
            n_vocal_lines_scanned += 1
    if anchors and not anchors[-1].get("end_anchor"):  # last one is a vocal section tag
        anchors[-1]["end_anchor"] = {
            "utt_idx": n_vocal_lines_scanned,
            "time": "end_time",
        }
    return anchors


def _get_deepchorus_segment_from_utterances_and_anchors(
    utterances: list[dict],
    anchors: list[dict],
    wav: torch.Tensor,
    sample_rate: int,
) -> list[dict]:
    # add a dummy utterance at the end
    len_in_ms = wav.shape[-1] / sample_rate * 1000
    utterances = utterances + [{"start_time": max(len_in_ms, utterances[-1]["end_time"])}]

    deepchorus = []
    for anchor in anchors:
        start_anchor = anchor["start_anchor"]
        end_anchor = anchor["end_anchor"]
        deepchorus.append({
            "tag": anchor["tag"],
            "start_time": 0 if start_anchor is None else utterances[start_anchor["utt_idx"]][start_anchor["time"]] / 1000,
            "end_time": utterances[end_anchor["utt_idx"]][end_anchor["time"]] / 1000,
        })
    return deepchorus


def _remove_bracketed_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    return "\n".join(lines)


def _reformat_structure_segments(deepchorus: list) -> list[dict]:
    return [{"tag": tag, "start_time": start, "end_time": end} for (start, end), tag in deepchorus]


def _scale_and_offset_utterances(
    utterances: list[dict],
    scale: float = 1.0,
    offset: float = 0.0,
) -> list[dict]:
    offset *= 1000  # s to ms
    utterances = copy.deepcopy(utterances)
    for utterance in utterances:
        utterance["start_time"] *= scale
        utterance["start_time"] += offset
        utterance["end_time"] *= scale
        utterance["end_time"] += offset
        for word in utterance["words"]:
            word["start_time"] *= scale
            word["start_time"] += offset
            word["end_time"] *= scale
            word["end_time"] += offset
    return utterances


def _scale_and_offset_notes(
    notes: list[dict],
    scale: float = 1.0,
    offset: float = 0.0,
) -> list[dict]:
    notes = copy.deepcopy(notes)
    for note in notes:
        note["start"] *= scale
        note["start"] += offset
        note["end"] *= scale
        note["end"] += offset
    return notes


def _scale_and_offset_structure(
    structure: list[dict],
    scale: float = 1.0,
    offset: float = 0.0,
) -> list[dict]:
    structure = copy.deepcopy(structure)
    for seg in structure:
        seg["start_time"] *= scale
        seg["start_time"] += offset
        seg["end_time"] *= scale
        seg["end_time"] += offset
    return structure