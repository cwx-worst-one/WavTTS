from typing import Any, Dict, List, Tuple, Optional, Iterable
import math
from bisect import bisect
from collections import defaultdict, namedtuple

import copy
import numpy as np
import pandas as pd
import pretty_midi

from recipes.datasets.mcc.sami_tokenizer import (
    convert_labels_to_text_id,
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
    BPM_DIVISIONS,
)
from recipes.bigmusic.datasets.symbolic_music.base_codec import (
    BaseCodec,
)
from recipes.bigmusic.datasets.symbolic_music.events_list_builder import (
    Priority,
    QueuedEvents,
)


ParsedEvent = namedtuple("ParsedEvent", ["event_type", "event_value"])


class SymbolicMusicCodecBase(BaseCodec):
    bar_part_priority = -5
    # genre_part_priority = -4
    section_part_priority = -3
    tempo_part_priority = -2
    chord_part_priority = -1

    lyrics_part_priority = 0
    vocal_part_priority = 1

    chord_index_dict = {c: i for i, c in enumerate(CHORD_LABELS)}
    to_chord_event = lambda c: f"chord_{SymbolicMusicCodecBase.chord_index_dict[c]}"
    to_note_on_event = lambda x: f"note_on_{x}"
    to_note_duration_event = lambda x: f"note_duration_{x}"
    to_drum_event = lambda x: f"drum_{x}"
    to_bpm_level_event = lambda x: f"bpm_level_{bisect(BPM_DIVISIONS, x)}"

    to_phone_event = lambda x: f"phone_{x}"

    stem_index_dict = {s: i for i, s in enumerate(STEM_LABELS)}
    to_stem_indicator_event = lambda s: f"stem_{SymbolicMusicCodecBase.stem_index_dict[s]}"

    section_index_dict = {s: i for i, s in enumerate(SECTION_LABELS)}
    to_section_indicator_event = lambda x: f"sec_{SymbolicMusicCodecBase.section_index_dict[x]}"

    to_position_event = lambda x: f"position_{x}"

    get_bar_event = lambda: "bar"
    get_eos_event = lambda: "eos"
    get_eop_event = lambda: "eop"

    @classmethod
    def _get_measure_position(
        cls,
        measure_times: List[float],
        time: float,
        divisions: int,
    ) -> Tuple[int, int]:
        measure_idx = np.searchsorted(measure_times, time, "right")
        if measure_idx == 0:
            ## Pickup bar, index 0
            measure_length = measure_times[1] - measure_times[0]
            position = math.floor(
                divisions * (time - measure_times[0] + measure_length) / measure_length
            )
            assert position >= 0, "Position before earlist measure time. Maybe missing bar lines at the beginning."
        elif measure_idx == len(measure_times):
            ## After the last bar, use the time for the last bar
            measure_length = measure_times[measure_idx - 1] - measure_times[measure_idx - 2]
            position = int(
                math.floor(
                    divisions * (time - measure_times[measure_idx - 1]) / measure_length
                )
            )
            assert position < divisions, "Position exceed divisions. Maybe missing bar lines at the end."
        else:
            measure_length = measure_times[measure_idx] - measure_times[measure_idx - 1]
            position = int(
                math.floor(
                    divisions * (time - measure_times[measure_idx - 1]) / measure_length
                )
            )

        return measure_idx, position
    
    def get_chord_queued_events(
        self,
        row: pd.Series,
        prev_chord: Optional[str],
        prev_measure_index: Optional[int],
        downbeat_times: Iterable[float],
        divisions: int,
        min_time: float,
        max_time: float,
    ) -> Optional[QueuedEvents]:
        # if row.chord == "N" or row.chord == prev_chord:
        #     return None
        if row.time < min_time or row.time >= max_time:
            return None
        measure_index, position = self._get_measure_position(
            downbeat_times, row.time, divisions,
        )
        if row.chord == prev_chord and measure_index == prev_measure_index:
            return None
        return QueuedEvents(
            priority=Priority(
                measure_index=measure_index,
                part_priority=SymbolicMusicCodecBase.chord_part_priority,
                position=position,
            ),
            events=[SymbolicMusicCodecBase.to_chord_event(row.chord)],
        )

    def get_note_queued_events(
        self,
        note: pd.Series,
        downbeat_times: Iterable[float],
        part_priority: int,
        divisions: int,
        min_time: float,
        max_time: float,
    ) -> Optional[QueuedEvents]:
        if note.start < min_time or note.end >= max_time:
            return None
        measure_index, position = self._get_measure_position(
            downbeat_times, note.start, divisions,
        )
        measure_index_end, position_end = self._get_measure_position(
            downbeat_times, note.end, divisions,
        )
        quantized_duration = (
            (measure_index_end - measure_index) * divisions
            + (position_end - position)
        )
        if quantized_duration == 0:
            return None

        events = [
            SymbolicMusicCodecBase.to_note_duration_event(min(quantized_duration, divisions * 2)),
            SymbolicMusicCodecBase.to_note_on_event(int(note.pitch)),
        ]

        return QueuedEvents(
            priority=Priority(
                measure_index=measure_index,
                part_priority=part_priority,
                position=position
            ),
            events=events,
        )

    def get_word_queued_events(
        self,
        word: pd.Series,
        downbeat_times: Iterable[float],
        part_priority: int,
        divisions: int,
        min_time: float,
        max_time: float,
    ):
        start, end = word.start_time / 1000, word.end_time / 1000
        if start < min_time or end >= max_time:
            return None
        measure_index, position = self._get_measure_position(
            downbeat_times, start, divisions,
        )
        measure_index_end, position_end = self._get_measure_position(
            downbeat_times, end, divisions,
        )
        quantized_duration = (
            (measure_index_end - measure_index) * divisions
            + (position_end - position)
        )
        if quantized_duration == 0:
            return None

        events = [
            SymbolicMusicCodecBase.to_note_duration_event(min(quantized_duration, divisions * 2)),
            *[SymbolicMusicCodecBase.to_phone_event(p) for p in word.phoneme_ids],
        ]

        return QueuedEvents(
            priority=Priority(
                measure_index=measure_index,
                part_priority=part_priority,
                position=position
            ),
            events=events,
        )

    def get_drum_queued_events(
        self,
        note: pd.Series,
        downbeat_times: Iterable[float],
        part_priority: int,
        divisions: int,
        min_time: float,
        max_time: float,
    ) -> Optional[QueuedEvents]:
        if note.start < min_time or note.start >= max_time:
            return None
        if note.pitch < 35 or note.pitch > 81:
            return None
        measure_index, position = self._get_measure_position(
            downbeat_times, note.start, divisions,
        )
        return QueuedEvents(
            priority=Priority(
                measure_index=measure_index,
                part_priority=part_priority,
                position=position
            ),
            events=[
                SymbolicMusicCodecBase.to_drum_event(note.pitch),
            ]
        )
    
    def select_sentence_endpoint(self, dfs_dict: Dict[str, Any], min_sec=0, max_sec=1000) -> np.ndarray:
        # df_lyrics contains multiple sentences
        # we random select one sentence as our last sentence according to its end_time
        df_lyrics = copy.deepcopy(dfs_dict["df_lyrics"])
        df_lyrics = df_lyrics[df_lyrics["end_time"]>=min_sec * 1000]
        df_lyrics = df_lyrics[df_lyrics["end_time"]<=max_sec * 1000]
        if len(df_lyrics) == 0:
            return None
        df_lyrics = df_lyrics.sample(1)
        df_lyrics = df_lyrics.reset_index(drop=True, inplace=False)
        return df_lyrics

    def encode_lyrics(self, dfs_dict: Dict[str, Any], start_sec=0, end_sec=1000) -> np.ndarray:
        df_lyrics = copy.deepcopy(dfs_dict["df_lyrics"])
        start_ms = math.floor(start_sec * 1000)
        end_ms = math.ceil(end_sec * 1000)
        df_lyrics = df_lyrics[df_lyrics["start_time"]>=start_ms]
        df_lyrics = df_lyrics[df_lyrics["end_time"]<=end_ms]
        if len(df_lyrics) == 0:
            return np.array([]), ""
        text = list(df_lyrics.text)
        text = '\n'.join(text)
        result = df_lyrics["phoneme"].apply(lambda x: convert_labels_to_text_id(x.split("\n")))
        if len(result) == 0:
            return np.array([]), ""
        result = result.reset_index(drop=True, inplace=False)
        lyrics_tokens = np.concatenate(result.apply(lambda x: x[0][0]).values)
        return np.append(lyrics_tokens, self.indexer["eol"]), text

    def encode_leadsheet(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        df_lyrics = dfs_dict["df_lyrics"]
        df_vocal2midi = dfs_dict["df_vocal2midi"]
        df_beat = dfs_dict["df_beat"]

        df_lyrics_word = pd.DataFrame(np.concatenate(df_lyrics.words).tolist())
        df_lyrics_word = df_lyrics_word[df_lyrics_word.apply(lambda x: x.normalized_text is not None, axis=1)]
        result = df_lyrics_word.phoneme.apply(lambda x: convert_labels_to_text_id(x.split("\n")))

        ## There might be weird errors for phoneme and the assertion below cannot always success.
        ## So we just skip the check here and use word_phoneme_ids[1:-3] no matter what
        # def assert_word_phoneme_format(x):
        #     assert (
        #         x[0] == 2 and x[-3] == 264 and x[-2] == 85 and x[-1] == 264
        #     )
        # result.apply(lambda x: assert_word_phoneme_format(x[0][0]))

        df_lyrics_word["phoneme_ids"] = result.apply(lambda x: x[0][0][1:-3])

        downbeat_times = df_beat[df_beat.beat == 1].time.values
        if len(downbeat_times) < 2:
            ## piece too short, skip
            return np.empty(0, dtype=np.uint16)    ## Extend one downbeat to cover notes and chords at the end
        assert "chord" in df_beat.columns, "Missing chord colum in df_beat"

        min_time = 2 * downbeat_times[0] - downbeat_times[1]
        max_time = 2 * downbeat_times[-1] - downbeat_times[-2]

        ## 2.2 Prepare events list
        ## 目前把每小节切分为16份（如果是3拍，切12份可能更合理，不过这需要很多额外逻辑）
        # divisions = 16 if time_sig_num == 4 else 12
        divisions = 16

        events_list: List[QueuedEvents] = []

        ## 2.2.1. Chord events
        prev_chord = None
        prev_measure_index = None
        for i, row in df_beat.iterrows():
            ## Ignore the last chord to avoid measure index going beyond len(measure_times)
            if i == len(df_beat) - 1:
                break
            qe = self.get_chord_queued_events(
                row,
                prev_chord,
                prev_measure_index,
                downbeat_times,
                divisions,
                min_time,
                max_time,
            )
            if qe is None:
                continue
            events_list.append(qe)
            prev_chord = row.chord
            prev_measure_index = qe.priority.measure_index

        ## 2.2.2 Words events
        for i, row in df_lyrics_word.iterrows():
            qe = self.get_word_queued_events(
                row,
                downbeat_times,
                SymbolicMusicCodecBase.lyrics_part_priority,
                divisions,
                min_time,
                max_time,
            )
            if qe is None:
                continue
            measure_index = qe.priority.measure_index
            if self.config.include_stem_indicator and prev_measure_index != measure_index:
                priority = Priority(
                    measure_index=measure_index,
                    part_priority=SymbolicMusicCodecBase.lyrics_part_priority,
                    position=-1,
                )
                events_list.append(QueuedEvents(
                    priority=priority,
                    events=[SymbolicMusicCodecBase.to_stem_indicator_event("lyrics")]
                ))
                prev_measure_index = measure_index
            events_list.append(qe)

        ## 3. Note events
        for _, row in df_vocal2midi.iterrows():
            qe = self.get_note_queued_events(
                row,
                downbeat_times,
                SymbolicMusicCodecBase.vocal_part_priority,
                divisions,
                min_time,
                max_time,
            )
            if qe is None:
                continue
            measure_index = qe.priority.measure_index
            if self.config.include_stem_indicator and prev_measure_index != measure_index:
                priority = Priority(
                    measure_index=measure_index,
                    part_priority=SymbolicMusicCodecBase.vocal_part_priority,
                    position=-1,
                )
                events_list.append(QueuedEvents(
                    priority=priority,
                    events=[SymbolicMusicCodecBase.to_stem_indicator_event("vocal")]
                ))
                prev_measure_index = measure_index
            events_list.append(qe)

        if self.config.include_bar_event:
            for i, _ in enumerate(downbeat_times):
                events_list.append(QueuedEvents(
                    priority=Priority(
                        measure_index=i,
                        part_priority=SymbolicMusicCodecBase.bar_part_priority,
                        position=0,
                    ),
                    events=[SymbolicMusicCodecBase.get_bar_event()]
                ))

        events_list.sort(key=lambda x: (
            x.priority.measure_index,
            x.priority.part_priority,
            x.priority.position,
        ))

        event_str_seq = []
        for events in events_list:
            first_event = events.events[0]
            if first_event.split("_")[0] in ["chord", "note", "phone"]:
                event_str_seq.append(SymbolicMusicCodecBase.to_position_event(events.priority.position))
            event_str_seq.extend(events.events)
        event_str_seq.append(SymbolicMusicCodecBase.get_eos_event())

        leadsheet_tokens = [self.indexer[es] for es in event_str_seq]
        return np.array(leadsheet_tokens)

    def encode(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        leadsheet_tokens = self.encode_leadsheet(dfs_dict)
        if self.config.include_utterance_phoneme_tokens:
            lyrics_tokens = self.encode_lyrics(dfs_dict)
            return np.concatenate([lyrics_tokens, leadsheet_tokens])
        return leadsheet_tokens

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
        phone_events = []
        for i, a in enumerate(arr):
            e = self.parse_event(a)
            if e.event_type == "bar":
                measure_counter += 1
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
                    # from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
                inst_dict[curr_stem].notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=start_time + dur_time,
                    )
                )
            elif e.event_type == "drum":
                assert (curr_stem is not None and start_time is not None)
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
            "chord": 2,
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


def get_datamodule_from_cls(cls_path, **kwargs):
    import importlib
    *module_paths, cls_name = cls_path.split('.')
    module = importlib.import_module('.'.join(module_paths))
    cls = getattr(module, cls_name)
    return cls.get_datamodule(**kwargs)
