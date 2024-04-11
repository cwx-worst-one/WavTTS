from typing import Any, Dict, List
from collections import defaultdict, namedtuple
from functools import lru_cache

import numpy as np
from bidict import bidict
from pydantic import BaseModel
import pretty_midi

from recipes.datasets.mcc.sami_tokenizer import (
    offset,
    phone_to_int,
    all_phones,
)
from recipes.bigmusic.datasets.utils.symbolic_music import chord_name_to_notes
from recipes.bigmusic.datasets.symbolic_music.consts import (
    CHORD_LABELS,
    STEM_LABELS,
    SECTION_LABELS,
    BPM_VALUES,
    GENRE_TAGS,
    # BPM_DIVISIONS,
)


class HashableConfig(BaseModel):
    def __hash__(self):
        return hash(frozenset(self.__dict__.items()))
    
    def __eq__(self, other: Any):
        if isinstance(other, HashableConfig):
            return self.__dict__ == other.__dict__
        return False


class IndexerConfig(HashableConfig):
    include_pad: bool = True
    ## Include phoneme in vocab or not
    include_phoneme: bool = True
    ## Include utterance tokens as prefix to leadsheet tokens
    ## This is only valid when include_phoneme is True
    include_utterance_phoneme_tokens: bool = True
    include_stem_indicator: bool = True

    ## We recommend setting this to True
    include_bar_event: bool = True

    ## Disable this when no drum track is included
    include_drum_events: bool = False

    ## Include symbolic prompt bars.
    ## Will introduce extra eop (end of prompt) token to the indexer
    include_prompt: bool = False

    ## (TODO) genre tags
    include_tagging: bool = False

    ## section tags
    include_section_indicator_each_bar: bool = False

    ## bpm levels
    include_bpm_levels: bool = False


@lru_cache(maxsize=None)
def get_indexer(config: IndexerConfig) -> bidict[str, int]:
    ## Add symbolic music tokens after phoneme tokens
    indexer = {}
    idx = 0

    ## Padding
    # if config.include_pad:
    indexer["pad"] = idx
    idx += 1

    ## End of song
    indexer["eos"] = idx
    idx += 1

    """DO NOT ADD MORE ENCODING HERE!!!
    Since phone assume there are only 2 tokens in front of phone tokens
    (offset = 2)
    """
    if config.include_phoneme:
        for i in range(len(phone_to_int)):
            indexer[f"phone_{i + offset}"] = idx
            idx += 1

    # Note-on events
    for i in range(128):
        indexer[f"note_on_{i}"] = idx
        idx += 1
    
    # Note-duration events
    for i in range(1, 33):
        indexer[f"note_duration_{i}"] = idx
        idx += 1

    # Drum events
    # General MIDI (GM) Level 1 Percussion Key Map range is 35~81
    if config.include_drum_events:
        for i in range(35, 82):
            indexer[f"drum_{i}"] = idx
            idx += 1

    # Position events
    for i in range(16):
        indexer[f"position_{i}"] = idx
        idx += 1

    # Bar event
    if config.include_bar_event:
        indexer["bar"] = idx
        idx += 1

    # Chord event
    for i, _ in enumerate(CHORD_LABELS):
        indexer[f"chord_{i}"] = idx
        idx += 1

    # Section indicator event
    if config.include_section_indicator_each_bar:
        for i, _ in enumerate(SECTION_LABELS):
            indexer[f"sec_{i}"] = idx
            idx += 1

    # Stem indicator event
    if config.include_stem_indicator:
        for i, _ in enumerate(STEM_LABELS):
            indexer[f"stem_{i}"] = idx
            idx += 1

    # BPM level indicators
    if config.include_bpm_levels:
        for i in range(len(BPM_VALUES)):
            indexer[f"bpm_level_{i}"] = idx
            idx += 1

    ## End of prompt
    if config.include_prompt:
        indexer["eop"] = idx
        idx += 1

    ## End of lyrics
    if config.include_utterance_phoneme_tokens:
        indexer["eol"] = idx
        idx += 1

    if config.include_tagging:
        for i, _ in enumerate(GENRE_TAGS):
            indexer[f"genre_{i}"] = idx
            idx += 1

    return bidict(indexer)


ParsedEvent = namedtuple("ParsedEvent", ["event_type", "event_value"])


class BaseCodec:
    def __init__(
        self,
        config: IndexerConfig,
    ):
        self.config = config
        self.indexer = get_indexer(self.config)

    def chop_or_pad(self, arr, target_len, constant_values=None):
        if constant_values is None:
            constant_values = self.indexer["pad"]
        if len(arr) < target_len:
            return np.pad(
                arr,
                (0, target_len - len(arr)),
                constant_values=constant_values,
            )
        return arr[:target_len]

    def encode(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError()

    def parse_event(self, index: int):
        event_str = self.indexer.inverse[index]
        if event_str.startswith("bar"):
            return ParsedEvent("bar", -1)
        if event_str.startswith("eos"):
            return ParsedEvent("eos", -1)
        if event_str.startswith("stem"):
            return ParsedEvent("stem", STEM_LABELS[int(event_str[len("stem_") :])])
        if event_str.startswith("note_on"):
            return ParsedEvent("note_on", int(event_str[len("note_on_") :]))
        if event_str.startswith("drum"):
            return ParsedEvent("drum", int(event_str[len("drum_") :]))
        if event_str.startswith("note_duration"):
            return ParsedEvent("note_duration", int(event_str[len("note_duration_") :]))
        if event_str.startswith("position"):
            return ParsedEvent("position", int(event_str[len("position_") :]))
        if event_str.startswith("chord"):
            return ParsedEvent("chord", CHORD_LABELS[int(event_str[len("chord_") :])])
        if event_str.startswith("phone"):
            return ParsedEvent("phone", int(event_str[len("phone_") :]))
        if event_str.startswith("sec"):
            return ParsedEvent("sec", SECTION_LABELS[int(event_str[len("sec_") :])])
        if event_str.startswith("bpm_level"):
            return ParsedEvent("bpm_level", BPM_VALUES[int(event_str[len("bpm_level_") :])])
        if event_str.startswith("eom"):
            return ParsedEvent("eom", -1)
        # if event_str.startswith("genre"):
        #     return ParsedEvent("genre", GENRE_TAGS[int(event_str[len("genre_") :])])

    def decode(self, arr: np.ndarray):
        divisions = 16
        chord_fixed_duration = 0.5
        drum_fixed_duration = 0.5
        phone_default_pitch = 64
        bpm = 120

        measure_counter = -1
        text_events: List[pretty_midi.Text] = []
        inst_dict = defaultdict(
            lambda : pretty_midi.Instrument(program=0, is_drum=False)
        )

        def report_wrong_format(arr, i):
            a = arr[max(0, i - 3) : min(len(arr), i + 3)]
            print("wrong format:")
            print(", ".join([self.indexer.inverse[c] for c in a]))

        phones = []
        phone_start_time = None
        start_time = None
        phone_events = []
        curr_stem = None
        for i, a in enumerate(arr):
            e = self.parse_event(a)
            if e.event_type == "bar":
                measure_counter += 1
                ## Reset stem each bar
                curr_stem = None
            elif e.event_type == "position":
                ## Position value is divided beat
                ## We assume 4/4 here
                pos_value = e.event_value
                start_beat = (measure_counter + pos_value / divisions) * 4
                start_time = start_beat / bpm * 60

                if len(phones) > 0:
                    ## Lazy append to phone track
                    assert (phone_start_time is not None and dur_time is not None)
                    phone_text = ",".join(phones)
                    phone_events.append(pretty_midi.Text(phone_text, phone_start_time))

                    ## Also add "phone note"
                    inst_dict["lyrics"].notes.append(
                        pretty_midi.Note(
                            velocity=100,
                            pitch=phone_default_pitch,
                            start=phone_start_time,
                            end=phone_start_time + dur_time,
                        )
                    )

                    phones = []
                    phone_start_time = None

                pitch = None
                dur_time = None
            elif e.event_type == "stem":
                curr_stem = e.event_value
            elif e.event_type == "chord":
                ## TODO: Add chord to another track. Currently only consume position event
                assert start_time is not None
                text_events.append(pretty_midi.Text(f"chord_{e.event_value}", start_time))
                chord_notes = chord_name_to_notes(e.event_value)
                for cn in chord_notes:
                    inst_dict["chord"].notes.append(pretty_midi.Note(
                        velocity=100,
                        pitch=cn,
                        start=start_time,
                        end=start_time + chord_fixed_duration,
                    ))
                start_time = None
            elif e.event_type == "note_on":
                pitch = e.event_value
                # assert (
                #     curr_stem is not None
                #     and dur_time is not None
                #     and start_time is not None
                # )
                if (curr_stem is None
                    or dur_time is None
                    or start_time is None):
                    report_wrong_format(arr, i)

                    continue
                inst_dict[curr_stem].notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=start_time + dur_time,
                    )
                )
            elif e.event_type == "drum":
                # assert (curr_stem is not None and start_time is not None)
                if (curr_stem is None or start_time is None):
                    continue
                pitch = e.event_value
                inst_dict[curr_stem].notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=start_time + drum_fixed_duration,
                    )
                )
            elif e.event_type == "note_duration":
                dur_value = e.event_value
                dur_time = dur_value / divisions * 4 / bpm * 60
            elif e.event_type == "phone":
                if len(phones) == 0:
                    assert start_time is not None, "Need to have position event before phone"
                    phone_start_time = start_time
                phones.append(all_phones[e.event_value - offset])
        
        if len(phones) > 0:
            ## Lazy append to phone track
            assert (phone_start_time is not None and dur_time is not None)
            phone_text = ",".join(phones)
            phone_events.append(pretty_midi.Text(phone_text, phone_start_time))

            ## Also add "phone note"
            inst_dict["lyrics"].notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=phone_default_pitch,
                    start=phone_start_time,
                    end=phone_start_time + dur_time,
                )
            )

        inst_program_dict = {
            "lyrics": 2,
            "vocal": 0,
            "chord": 24,
            "guitar": 24,
            "bass": 38,
        }
        inst_order_dict = {
            "lyrics": 0,
            "vocal": 1, 
            "piano": 2,
            "guitar": 3,
            "bass": 4,
            "drums": 5,
            "chord": 6,
        }
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=120)
        for k, inst in inst_dict.items():
            inst.name = k
            if inst.name in inst_program_dict.keys():
                inst.program = inst_program_dict[k]
                ## HACK: show genre together with vocal stem name for convenience
                # if inst.name == "vocal" and curr_genre is not None:
                #     inst.name = f"vocal_{curr_genre}"
            elif inst.name == "drums":
                inst.is_drum = True
            midi_obj.instruments.append(inst)
        midi_obj.text_events.extend(text_events)
        midi_obj.text_events.extend(phone_events)
        ## Reorder instruments
        midi_obj.instruments.sort(
            key=lambda x: inst_order_dict[x.name]
        )
        return midi_obj
