import copy
import operator
import random
from dataclasses import asdict, dataclass
from functools import reduce
from typing import Optional, Union

import numpy as np

from ..tokenizers.sami_phoneme_tokenizer.leadsheet import fetch_notes_from_phones_v3
from ..utils.dcbase import DCBase
from .phoneme_parser import (
    add_instrument_tags,
    add_section_tag,
    add_singer_tag,
    convert_phonemes,
    extract_instrument_tags,
    extract_section_tag,
    extract_singer_tag,
)

___all___ = ["SongSliceError", "split_into_song_slices", "validate_song_slice"]


class SongSliceError(Exception):
    pass


# ==================== API Functions ========================


def split_into_song_slices(
    max_duration: float,
    structure: list[dict],
    utterances: list[dict],
    style_tags: dict,
    style_input_ids: dict,
    freeform_text: str,
    vocal2midi: dict,
    tempo: float,
    slice_mode: str = "section",
    line_break_dropout_rate: float = 0.0,
) -> list[dict]:
    song_slices = transform_utts_to_song_slices_structure(
        utterances,
        max_duration,
        structure,
        slice_mode=slice_mode,
        language="",
        line_break_dropout_rate=line_break_dropout_rate,
    )
    # Assign global features
    for song_slice in song_slices:
        song_slice.style_tags = style_tags
        song_slice.style_input_ids = style_input_ids
        song_slice.freeform_text = freeform_text
    # Assign notes
    if vocal2midi:
        notes = Note.from_vocal2midi(vocal2midi)
        for song_slice in song_slices:
            song_slice.notes = slice_notes(notes, song_slice.time_span)
    # Assign tempo (BPM)
    for song_slice in song_slices:
        song_slice.tempo = tempo
    return [song_slice.to_dict() for song_slice in song_slices]


def validate_song_slice(
    song_slice: dict, duration_range: tuple[float, float], lyrics_confidence: float
) -> None:
    song_slice = SongSlice.from_dict(song_slice)
    duration = song_slice.duration
    min_dur, max_dur = duration_range
    if not (min_dur <= duration <= max_dur):
        raise SongSliceError(
            f"Duration {duration} is not in range [{min_dur}, {max_dur}]"
        )

    for phrase in song_slice.phrases:
        phrase_lc = phrase.lyrics_confidence
        if phrase_lc is not None and phrase_lc < lyrics_confidence:
            raise SongSliceError(
                f"Lyrics confidence {phrase.lyrics_confidence} is less than {lyrics_confidence}"
            )


# =========================================================


@dataclass
class Phrase(DCBase):
    phonemes: Optional[str] = None
    text: Optional[str] = None
    singer_tag: Optional[str] = None
    section_tag: Optional[str] = None
    instruments: Optional[list] = None
    time_span: Optional[Union[tuple[int, int], tuple[float, float]]] = None
    lyrics_confidence: Optional[float] = None

    @classmethod
    def parse(
        cls,
        phonemes: Optional[str] = None,
        text: Optional[str] = None,
        singer_tag: Optional[str] = None,
        section_tag: Optional[str] = None,
        instrument_tags: Optional[str] = None,
        time_span: Optional[tuple[int, int]] = None,
        lyrics_confidence: Optional[float] = None,
        normalize_tag: bool = False,
        normalize_chinese: bool = True,
    ):
        """Auto-detect singer tag and section tag from the given phonemes or text
        :param singer_tag: Override the detected singer_tag by this value
        :param section_tag: Override the detected section_tag by this value
        :param normalize_tag:
            True: Support other section format besides the strict formats (for user input)
            False: Only parse section tags enclosed by "[]"
        """

        def strip(text: Optional[str]) -> Optional[str]:
            if text is None:
                return None
            s = text.strip()
            if not s:
                return None
            return s

        if phonemes is None:
            _section_tag_p, _singer_tag_p, _instrument_tags_p, rest_phonemes = (
                None,
                None,
                None,
                None,
            )
        else:
            _section_tag_p, rest_phonemes = extract_section_tag(phonemes, normalize_tag)
            _singer_tag_p, rest_phonemes = extract_singer_tag(rest_phonemes)
            _instrument_tags_p, rest_phonemes = extract_instrument_tags(rest_phonemes)

        if text is None:
            _section_tag_t, _singer_tag_t, _instrument_tags_t, rest_text = (
                None,
                None,
                None,
                None,
            )
        else:
            _section_tag_t, rest_text = extract_section_tag(text, normalize_tag)
            _singer_tag_t, rest_text = extract_singer_tag(rest_text)
            _instrument_tags_t, rest_text = extract_instrument_tags(rest_text)

        # Use extracted tags if tags are not forced
        if section_tag is None:
            section_tag = _section_tag_p if _section_tag_t is None else _section_tag_t
        if singer_tag is None:
            singer_tag = _singer_tag_p if _singer_tag_t is None else _singer_tag_t
        if instrument_tags is None:
            instrument_tags = (
                _instrument_tags_p if _instrument_tags_t is None else _instrument_tags_t
            )

        if normalize_chinese and rest_text is not None:
            # Temporary patch for Chinese vocal v3.6: Remove the confusable character conversion
            # TODO: Add custom logic to handle confusables, do not use any third-party package
            # rest_text = norm_chinese_text(rest_text)
            rest_text = norm_chinese_text_no_confusable_conversion(rest_text)

        return cls(
            phonemes=rest_phonemes,
            text=strip(rest_text),
            singer_tag=singer_tag,
            section_tag=section_tag,
            instruments=instrument_tags,
            time_span=time_span,
            lyrics_confidence=lyrics_confidence,
        )

    @classmethod
    def concat(cls, phrase_a: "Phrase", phrase_b: "Phrase") -> "Phrase":
        def concat_opt_str(str_a: Optional[str], str_b: Optional[str]) -> Optional[str]:
            if str_a is None and str_b is None:
                return None
            str_a = "" if str_a is None else str_a
            str_b = "" if str_b is None else str_b
            return str_a + str_b

        def concat_opt_str_mix_lang(str_a: Optional[str], str_b: Optional[str]):
            if (
                not str_a or not str_b
            ) or (  # also handles the case when any of these is empty
                is_chinese_char(str_a[-1].encode("unicode_escape"))
                and is_chinese_char(str_b[0].encode("unicode_escape"))
            ):
                return concat_opt_str(str_a, str_b)
            return concat_opt_str(str_a, " " + str_b)  # add a space inbetween

        def concat_phonemes(
            phone_a: Optional[str], phone_b: Optional[str]
        ) -> Optional[str]:
            if phone_a is None or phone_b is None:
                return concat_opt_str(phone_a, phone_b)
            phone_a_split = phone_a.split("\n")
            if (
                phone_a_split[-1].split("\t")[0] == "。"
            ):  # split guarantees the list is non-empty
                phone_a = "\n".join(phone_a_split[:-1])  # cut out the line break

            phone_b_split = phone_b.split("\n")
            if phone_b_split[0].split("\t")[0] == "sil":
                phone_b = "\n".join(phone_b_split[1:])  # cut out the starting sil

            return phone_a + "\n" + phone_b

        def concat_time_span(
            time_span_a: Optional[tuple[int, int]],
            time_span_b: Optional[tuple[int, int]],
        ) -> Optional[tuple[int, int]]:
            # invalid time span is supposed to be filtered out before running this function
            if time_span_a is None or time_span_b is None:
                return None
            start = min(time_span_a[0], time_span_b[0])
            end = max(time_span_a[1], time_span_b[1])
            return start, end

        def concat_instruments(
            instruments_a: Optional[list], instruments_b: Optional[list]
        ) -> Optional[list[str]]:
            instruments_a = instruments_a if instruments_a else []
            instruments_b = instruments_b if instruments_b else []
            instruments = instruments_a + instruments_b
            if not instruments:
                instruments = None
            return instruments

        # refuse to concat if the tag does not match
        if not cls.concatable(phrase_a, phrase_b):
            raise SongSliceError(
                "Unable to concatenate two phrases with different prefix_tags"
            )

        return cls(
            text=concat_opt_str_mix_lang(phrase_a.text, phrase_b.text),
            phonemes=concat_phonemes(phrase_a.phonemes, phrase_b.phonemes),
            singer_tag=phrase_a.singer_tag,
            section_tag=phrase_b.section_tag,
            instruments=concat_instruments(phrase_a.instruments, phrase_b.instruments),
            time_span=concat_time_span(phrase_a.time_span, phrase_b.time_span),
        )

    @classmethod
    def concatable(cls, phrase_a, phrase_b) -> bool:
        def time_span_match(
            time_span_a: Optional[tuple[int, int]],
            time_span_b: Optional[tuple[int, int]],
        ) -> bool:
            # either both have time_span or both do not have
            return all(ts is None for ts in [time_span_a, time_span_b]) or all(
                ts is not None for ts in [time_span_a, time_span_b]
            )

        return (phrase_a.prefix_tags == phrase_b.prefix_tags) and time_span_match(
            phrase_a.time_span, phrase_b.time_span
        )

    @property
    def prefix_tags(self) -> list[str]:
        return list(filter(None, [self.section_tag, self.instruments, self.singer_tag]))

    @property
    def has_utterance(self) -> bool:
        return bool(self.phonemes or self.text)

    @property
    def is_empty(self) -> bool:
        return (
            not self.has_utterance
            and self.section_tag is None
            and self.singer_tag is None
        )

    @property
    def start(self) -> Optional[int]:
        return None if self.time_span is None else self.time_span[0]

    @property
    def end(self) -> Optional[int]:
        return None if self.time_span is None else self.time_span[1]

    @property
    def duration(self) -> Optional[int]:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def format_phonemes(self) -> str:
        return self._format_str(self.phonemes)

    def format_text(self) -> str:
        return self._format_str(self.text)

    def _format_str(self, _str) -> str:
        if _str is None:
            _str = ""
        _str = add_singer_tag(self.singer_tag, _str)
        _str = add_instrument_tags(self.instruments, _str)
        _str = add_section_tag(self.section_tag, _str)
        return _str


@dataclass
class Note(DCBase):
    pitch: int
    time_span: tuple[float, float]

    def to_leadsheet_note(self) -> dict:
        if not self.time_span:
            raise ValueError("time_span is not set")
        return {
            "pitch": self.pitch,
            "start": self.time_span[0],
            "end": self.time_span[1],
        }

    @classmethod
    def from_vocal2midi(cls, d: dict) -> list["Note"]:
        return [
            cls(pitch=_d["pitch"], time_span=(_d["start"], _d["end"]))
            for _d in d["notes"]
        ]

    def transpose(self, semitones: int) -> "Note":
        """For augmentation"""
        return self.__class__(**{**asdict(self), "pitch": self.pitch + semitones})


@dataclass
class SongSlice(DCBase):
    phrases: list[Phrase]
    style_tags: Optional[dict[str, list[str]]] = None
    style_input_ids: Optional[dict[str, list[int]]] = None
    freeform_text: Optional[str] = None
    notes: Optional[list[Note]] = None
    tempo: Optional[float] = None

    symbol_seq: Optional[list[dict]] = None
    phoneme_tokens: Optional[list[dict]] = None

    @property
    def start(self) -> float:
        return self.phrases[0].start

    @property
    def end(self) -> float:
        return self.phrases[-1].end

    @property
    def time_span(self) -> tuple[float, float]:
        return self.start, self.end

    @property
    def duration(self) -> float:
        if not self.phrases:
            return 0.0
        return self.end - self.start

    @property
    def has_utterance(self) -> bool:
        if not self.phrases:
            return False
        return any(phrase.has_utterance for phrase in self.phrases)

    def __add__(self, other: "SongSlice"):
        def get_freeform_text():
            self_ft = self.freeform_text if self.freeform_text else ""
            other_ft = other.freeform_text if other.freeform_text else ""
            if self_ft == other_ft:
                ft = self_ft
            else:
                ft = self_ft + ", " + other_ft
            return ft if ft else None

        notes = (
            None
            if (self.notes is None or other.notes is None)
            else self.notes + other.notes
        )
        return self.__class__(
            phrases=self.phrases + other.phrases,
            freeform_text=get_freeform_text(),
            notes=notes,
        )

    def reformat_inplace(self) -> "SongSlice":
        phrases = move_out_section_tags(self.phrases)
        for phrase in phrases:
            if phrase.section_tag is not None:
                phrase.section_tag = _remove_count_from_section_tag(phrase.section_tag)
        self.phrases = phrases
        return self

    def tokenize_inplace(self, tokenizer, **kwargs) -> "SongSlice":
        self.phoneme_tokens = tokenizer(self.symbol_seq, **kwargs)
        return self

    # =================================================================

    def sequentialize(self, mode: str = "phoneme") -> list[dict]:
        # "audio" -> specific mode
        if mode == "auto":
            mode = "phoneme_note" if self.notes else "phoneme"
        if mode == "auto_cfg":
            mode = "phoneme_note_cfg" if self.notes else "phoneme_cfg"

        if self.has_utterance:
            if mode == "phoneme":
                return self._sequentialize_vocal()
            elif mode == "phoneme_cfg":
                return self._sequentialize_vocal_cfg()
            elif mode == "phoneme_note":
                return self._sequentialize_vocal_with_notes()
            elif mode == "phoneme_note_cfg":
                return self._sequentialize_vocal_with_notes_cfg()
            raise SongSliceError(f"Unknown mode: {mode}")
        else:
            if mode in ["phoneme", "phoneme_note"]:
                return self._sequentialize_inst()
            elif mode in ["phoneme_cfg", "phoneme_note_cfg"]:
                return self._sequentialize_inst_cfg()
            raise SongSliceError(f"Unknown mode: {mode}")

    def sequentialize_inplace(self, mode: str = "phoneme") -> "SongSlice":
        self.symbol_seq = self.sequentialize(mode)
        return self

    def _sequentialize_vocal_no_duration(self) -> list[dict]:
        seq = []
        phrases = self.phrases
        for phrase in phrases:
            if phrase.section_tag:
                seq.append({"symbol": phrase.section_tag})
            if phrase.singer_tag:
                seq.append({"symbol": phrase.singer_tag})
            if phrase.phonemes:
                seq.append({"phonemes": phrase.phonemes})
        return seq

    def _sequentialize_vocal_cfg(self) -> list[dict]:
        seq = []
        phrases = self.phrases
        for phrase in phrases:
            if phrase.phonemes:
                seq.append({"phonemes": phrase.phonemes})
        return seq

    def _sequentialize_vocal(self) -> list[dict]:
        if self.duration <= 0:
            raise SongSliceError("Non-positive slice_duration")
        return [
            {"slice_duration": self.duration}
        ] + self._sequentialize_vocal_no_duration()

    def _sequentialize_inst(self) -> list[dict]:
        if self.duration <= 0:
            raise SongSliceError("Non-positive slice_duration")
        seq = []
        seq.append({"slice_duration": self.duration})
        phrases = self.phrases
        for phrase in phrases:
            if phrase.section_tag:
                seq.append({"symbol": phrase.section_tag})
                if phrase.duration <= 0:
                    raise SongSliceError("Non-positive section_duration")
                seq.append({"section_duration": phrase.duration})
            if phrase.instruments:
                for inst in phrase.instruments:
                    seq.append({"symbol": inst})
        return seq

    def _sequentialize_inst_cfg(self) -> list[dict]:
        return []

    def _sequentialize_vocal_with_notes_no_slice_duration(
        self, offset: float = 0
    ) -> list[dict]:
        seq = self._sequentialize_vocal_no_duration()
        for note in self.notes:
            seq.append(
                {
                    "pitch": note.pitch,
                    "time_span": (
                        note.time_span[0] + offset,
                        note.time_span[1] + offset,
                    ),
                }
            )
        return seq

    def _sequentialize_vocal_with_notes(self) -> list[dict]:
        def get_start_time_offset() -> float:
            if not self.phrases:
                return 0
            phrase_start = 0 if self.start is None else self.start
            if self.notes:
                note_start = self.notes[0].time_span[0]
                return min(phrase_start, note_start)
            return phrase_start

        def group_notes_by_phrases() -> list[list[Note]]:
            def get_overlap(
                time_span_a: tuple[int, int], time_span_b: tuple[float, float]
            ) -> float:
                left_overlap = max(time_span_a[0], time_span_b[0])
                right_overlap = min(time_span_a[1], time_span_b[1])
                if right_overlap <= left_overlap:
                    return 0.0
                return right_overlap - left_overlap

            if self.notes is None:
                raise SongSliceError("Notes are not available")
            if self.phrases is None:
                raise SongSliceError("Phrases are not available")
            note_groups = [[] for _ in range(len(self.phrases))]
            phrase_ts = [
                phrase.time_span if phrase.has_utterance else ()
                for phrase in self.phrases
            ]
            for note in self.notes:
                overlaps = []
                for phrase_time_span in phrase_ts:
                    if not phrase_time_span:
                        overlaps.append(0)
                    else:
                        overlaps.append(get_overlap(phrase_time_span, note.time_span))
                if sum(overlaps) == 0:
                    continue  # skip the note
                note_groups[overlaps.index(max(overlaps))].append(note)
            return note_groups

        if self.duration <= 0:
            raise SongSliceError("Non-positive slice_duration")
        seq = [{"section_duration": self.duration}]
        offset = -get_start_time_offset()
        note_groups = group_notes_by_phrases()
        for phrase, notes in zip(self.phrases, note_groups):
            sub_song_slice = copy.deepcopy(self)
            sub_song_slice.phrases = [phrase]
            sub_song_slice.notes = notes
            seq.extend(
                sub_song_slice._sequentialize_vocal_with_notes_no_slice_duration(offset)
            )
        return seq

    def _sequentialize_vocal_with_notes_cfg(self) -> list[dict]:
        def dict_drop(d):
            return "section_duration" in d or "slice_duration" in d

        seq = self._sequentialize_vocal_with_notes()
        return [d for d in seq if not dict_drop(d)]


_SECTION_TAG_SEP = "#"


def _format_utterances(utterances):
    # This function cleans up the raw utterances data from metadata.
    # Extracts necessary information: [start_time_in_sec, end_time_in_sec, lyrics text with additional tags, precomputed
    #   phonemes]
    new_utterances = []
    for u in utterances:
        phone_v86 = u.get(
            "phoneme_v86"
        )  # phoneme string tagged by tts frontend v86, use it whenever it's possible
        phone = phone_v86 if phone_v86 else u.get("phoneme")

        # NOTE: This operation tries to obtain confidence from 2 sources:
        # - For force align lyrics: utterance.confidence
        # - For ASR lyrics: utterance.attribute.confidence
        # This is not ideal, because we want to separate the parsing methods, and make each one specific.
        # But it might require more code change.
        # For now, it is only up to the validation step to make sure the format is correct.
        # Since the given utterance is either from ASR or force alignment, it not very likely to go wrong.
        confidence = u.get("confidence", u.get("attribute", {}).get("confidence"))

        utt_start = u.get("start_time")
        utt_end = u.get("end_time")
        try:
            new_utterances.append(
                {
                    "start_time": utt_start / 1000,
                    "end_time": utt_end / 1000,
                    "text": u["text"],
                    "phonemes": phone,
                    "confidence": confidence,
                }
            )
        except TypeError:
            continue
    return new_utterances


def transform_utts_to_song_slices_structure(
    utterances: list[dict],
    max_duration: int,
    structure_tags: list[dict],
    slice_mode: str,
    language: Optional[str] = None,
    line_break_dropout_rate: float = 0.0,
) -> list[SongSlice]:
    """Segment the full song into a list of SongSlice, each consists of multiple utterances.
    Args:
        slice_mode:
        - "section": Return all the possible slices within max_duration with complete sections.
        - "full": Return the entire song
    """

    def format_utterances(utterances: list[dict]) -> list[Phrase]:
        formatted_us = _format_utterances(utterances)
        return [
            Phrase.parse(
                text=us["text"],
                phonemes=convert_phonemes(us["phonemes"], language),
                time_span=(us["start_time"], us["end_time"]),
                lyrics_confidence=us["confidence"],
            )
            for us in formatted_us
        ]

    def get_overlap(
        phrase_time_span: tuple[int, int], section_time_span: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        left_overlap = max(phrase_time_span[0], section_time_span[0])
        right_overlap = min(phrase_time_span[1], section_time_span[1])
        if right_overlap <= left_overlap:
            return None
        return (left_overlap, right_overlap)

    def get_overlap_dur(
        phrase_time_span: tuple[int, int], section_time_span: tuple[float, float]
    ) -> float:
        overlap = get_overlap(phrase_time_span, section_time_span)
        if overlap is None:
            return 0
        return overlap[1] - overlap[0]

    def get_section_span(structure_tag: dict) -> tuple[float, float]:
        return structure_tag["start_time"], structure_tag["end_time"]

    def add_section_tag(phrase: Phrase) -> Phrase:
        """Return a new phrase with a section tag based on the longest overlap"""
        if not structure_tags:
            return phrase
        overlaps = [
            get_overlap_dur(phrase.time_span, get_section_span(structure_tag))  # type: ignore
            for structure_tag in structure_tags
        ]
        if not overlaps:
            return phrase
        idx = int(np.argmax(overlaps))
        return phrase._replace(section_tag=structure_tags[idx]["tag"])

    def get_inst_phrases(time_span: tuple[int, int]) -> list[Phrase]:
        phrases = []
        for structure_tag in structure_tags:
            overlap = get_overlap(time_span, get_section_span(structure_tag))
            if overlap is None:
                continue
            phrases.append(Phrase(section_tag=structure_tag["tag"], time_span=overlap))
        return phrases

    def insert_inst_phrases(
        phrases: list[Phrase], song_time_span: tuple[int, int]
    ) -> list[Phrase]:
        """Insert instrument phrases if a gap presents between two adjacent vocal phrases."""
        song_start, song_end = song_time_span
        if not phrases:
            return get_inst_phrases(song_time_span)
        new_phrases = []
        for idx, (curr_phrase, next_phrase) in enumerate(
            zip(phrases, phrases[1:] + [None])
        ):
            # the gap between song start and the first phrase's start
            if idx == 0 and curr_phrase.start - song_start > 0:  # first phrase
                _phrases = get_inst_phrases((song_start, curr_phrase.start))
                new_phrases.extend(_phrases)
            # add the current phrase
            new_phrases.append(curr_phrase)
            # the gap between the current phrase and the next phrase
            if idx < len(phrases) - 1 and next_phrase.start - curr_phrase.end > 0:
                _phrases = get_inst_phrases((curr_phrase.end, next_phrase.start))
                new_phrases.extend(_phrases)
            # the gap between the last phrase's end and the song end
            elif (
                idx == len(phrases) - 1 and song_end - curr_phrase.end > 0
            ):  # last phrase
                _phrases = get_inst_phrases((curr_phrase.end, song_end))
                new_phrases.extend(_phrases)
        return new_phrases

    def get_section_start_ind(phrases: list[Phrase]) -> list[int]:
        sec_tags = [p.section_tag for p in phrases]
        ind = [0]
        for idx, (curr_tag, next_tag) in enumerate(zip(sec_tags, sec_tags[1:])):
            if next_tag != curr_tag:
                ind.append(idx + 1)
        return ind

    def remove_long_inst_intro(
        phrases: list[Phrase], max_intro_dur: int = 20
    ) -> list[Phrase]:
        """Remove instrumental intro phrases whose lengths are longer max_intro_dur.
        `max_intro_dur` is hard-coded for now. Ideally it should change according to the full song length.
        """
        return [
            phrase
            for phrase in phrases
            if not (
                phrase.section_tag is not None
                and _remove_count_from_section_tag(phrase.section_tag) == "intro"
                and not phrase.has_utterance
                and phrase.duration > max_intro_dur
            )
        ]

    def get_song_slices_complete_section(phrases: list[Phrase]) -> list[SongSlice]:
        """Group phrases into song slices, but do not split any section."""
        if not phrases:
            return []
        section_start_ind = get_section_start_ind(phrases) + [len(phrases)]
        song_slices_by_sections = [
            SongSlice(phrases=phrases[start:end])
            for start, end in zip(section_start_ind, section_start_ind[1:])
        ]
        end_ts = np.array([song_slice.end for song_slice in song_slices_by_sections])
        song_slices = []
        for idx, song_slice in enumerate(song_slices_by_sections):
            rel_end_ts = (
                end_ts[idx:] - song_slice.start
            )  # section end times relative to the current song_slice's start
            # idx_inc indicates the max number of song slices that can be combined
            idx_inc = max(
                1, int(np.searchsorted(rel_end_ts, max_duration, side="right"))
            )
            # NOTE: The next step pushes every possible lengths into the list starting from idx.
            # If we simply push [idx: idx+idx_inc], we are enforcing the slice to reach max_duration
            # as much as possible, which is not necessarily what we want if we need to handle
            # short lyrics and generate short songs.
            for _inc in range(1, idx_inc + 1):
                song_slices.append(
                    reduce(operator.add, song_slices_by_sections[idx : idx + _inc])
                )
        return song_slices

    def get_song_slices_full(phrases: list[Phrase]) -> list[SongSlice]:
        if not phrases:
            return []
        return [SongSlice(phrases=phrases)]

    slice_fn = {
        "section": get_song_slices_complete_section,
        "full": get_song_slices_full,
    }[slice_mode]
    song_time_span = (structure_tags[0]["start_time"], structure_tags[-1]["end_time"])
    phrases = format_utterances(utterances)
    phrases = list(map(add_section_tag, phrases))
    phrases = insert_inst_phrases(phrases, song_time_span)
    if slice_mode != "full":  # long intro can be included under "full" mode
        phrases = remove_long_inst_intro(phrases)
    song_slices = slice_fn(phrases)
    # Remove short instrumental sections' section tags. After the removal, these phrases will become placeholders
    # for SongSlice to correctly calculate the start and end times, but will be completely ignored during tokenization.
    song_slices = list(map(_reset_section_tags_for_short_inst_phrases, song_slices))
    for song_slice in song_slices:
        song_slice.phrases = drop_out_line_breaks(
            song_slice.phrases, line_break_dropout_rate
        )
    return [
        song_slice.reformat_inplace()
        for song_slice in song_slices
        if song_slice.phrases
    ]


def is_chinese_char(ch):
    return ch >= b"\\u4e00" and ch <= b"\\u9fff"


def norm_chinese_char_no_confusable_conversion(ch):
    """A temporary normalization function without confusable conversion"""
    return ch if is_chinese_char(ch.encode("unicode_escape")) else None


def norm_chinese_text_no_confusable_conversion(s):
    """A temporary normalization function without confusable conversion"""
    res = ""
    for ch in s:
        normed = norm_chinese_char_no_confusable_conversion(ch)
        if not normed or normed == ch:
            res += ch
            continue
        res += normed
    return res


def _reset_section_tags_for_short_inst_phrases(song_slice: SongSlice) -> SongSlice:
    """Remove short instrumental phrases' section tags to avoid adding unnessary section tags."""

    def does_phrase_need_reset(phrase: Phrase) -> bool:
        return (
            not phrase.has_utterance
            and phrase.section_tag is not None
            and phrase.duration is not None
            and phrase.duration < 2  # hard-coded duration threshold
        )

    phrases = [
        phrase._replace(section_tag=None) if does_phrase_need_reset(phrase) else phrase
        for phrase in song_slice.phrases
    ]
    phrases = [phrase for phrase in phrases if not phrase.is_empty]
    return SongSlice(phrases=phrases)


def _remove_count_from_section_tag(section_tag: str) -> str:
    return section_tag.split(_SECTION_TAG_SEP)[0]


def slice_notes(notes: list[Note], time_span: tuple[float, float]) -> list[Note]:
    start, end = time_span
    note_sequence = [note.to_leadsheet_note() for note in notes]
    sliced_notes, _ = fetch_notes_from_phones_v3(
        [{"start": start, "end": end, "phone": ""}], note_sequence
    )
    return [
        Note(pitch=note["pitch"], time_span=(note["start"], note["end"]))
        for note in sliced_notes
    ]


def move_out_section_tags(phrases: list[Phrase]) -> list[Phrase]:
    """Move section tags out of phrases as single phrases
    This function reformats a list of phrases to a format where section tags
    only appear once at the top of each section.

    2024/06/25 update:
    The function also calculates section-wise time spans, assign them to section-tag phrases,
    which makes it easier to calculate section durations after reformatting.
    The phrases may have incomplete section or section-wise time_span coverage.
    """
    section_duration_dict = {}
    for prev_phrase, phrase in zip([None] + phrases, phrases):
        if not phrase.section_tag:
            continue
        if prev_phrase is None or phrase.section_tag != prev_phrase.section_tag:
            # Insert a new record whenever a new section is found
            section_duration_dict[phrase.section_tag] = phrase.time_span
        else:
            # For a phrase in the same section, extend the end time of the time_span
            if section_duration_dict[phrase.section_tag]:
                section_duration_dict[phrase.section_tag] = (
                    section_duration_dict[phrase.section_tag][0],
                    phrase.end,
                )

    out_phrases = []
    last_section_tag_phrase_idx = -1
    for prev_phrase, phrase in zip([None] + phrases, phrases):
        if phrase.section_tag and (
            prev_phrase is None or phrase.section_tag != prev_phrase.section_tag
        ):
            # Insert section-tag phrase with time_span calculated by the previous step
            out_phrases.append(
                Phrase(
                    section_tag=phrase.section_tag,
                    time_span=section_duration_dict.get(phrase.section_tag),
                    instruments=phrase.instruments,
                )
            )
            last_section_tag_phrase_idx = len(out_phrases) - 1
        if phrase.has_utterance:
            # Remove section tag
            out_phrases.append(phrase._replace(section_tag=None))
            # Phrase is not covered by deepchorus (it should not happen most of the time), extend the section end time
            if phrase.section_tag is None and last_section_tag_phrase_idx >= 0:
                last_section_tag_phrase = out_phrases[last_section_tag_phrase_idx]
                section_time_span = last_section_tag_phrase.time_span
                if section_time_span is not None:
                    out_phrases[last_section_tag_phrase_idx] = (
                        last_section_tag_phrase._replace(
                            time_span=(section_time_span[0], phrase.end)
                        )
                    )
    return out_phrases


def drop_out_line_breaks(phrases: list[Phrase], rate: float) -> list[Phrase]:
    """Merge adjacent concatable phrases. The phrase list should NOT be reformatted by move_out_section_tags."""
    if not (0 <= rate <= 1):
        raise ValueError(f"Invalid dropout rate: {rate}")
    if len(phrases) <= 1 or rate == 0:
        return phrases
    mergeable_ind = [
        idx
        for idx, (curr_phrase, next_phrase) in enumerate(zip(phrases, phrases[1:]))
        if Phrase.concatable(curr_phrase, next_phrase)
    ]
    if not mergeable_ind:
        return phrases
    phrases = phrases[:]  # shallow copy a new slice, Phrase is immutable so it's fine
    for idx in reversed(mergeable_ind):  # reverse it because the list shrinks
        if random.random() < rate:
            phrases[idx : idx + 2] = [Phrase.concat(phrases[idx], phrases[idx + 1])]
    return phrases
