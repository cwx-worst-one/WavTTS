from typing import Dict, Any, List, Iterable
from bisect import bisect

import numpy as np
import pandas as pd
from pydantic import BaseModel

from recipes.bigmusic.utils.common_utils import (
    colorful_sequence,
)
from recipes.datasets.mcc.sami_tokenizer import (
    convert_labels_to_text_id,
)
from recipes.bigmusic.datasets.symbolic_music.consts import (
    CHORD_LABELS,
    STEM_LABELS,
    SECTION_LABELS,
    BPM_VALUES,
    BPM_DIVISIONS,
)


class Priority(BaseModel):
    """Every event has a global priority, with
    measure_index > part_priority > position
    so that we can add event in any order and generate
    the final remi numpy after re-ordering using the
    global priority
    """
    measure_index: int
    part_priority: int
    position: int


class QueuedEvents(BaseModel):
    priority: Priority
    events: List[str]


def divide_sequence(sequence: Iterable, division: int):
    # Ensure that the division count is valid
    if division < 1:
        raise ValueError("Division must be at least 1.")

    # The final result starts with the first element
    result = [sequence[0]]

    # Calculate the intervals and divide them
    for start, end in zip(sequence[:-1], sequence[1:]):
        # Calculate the step size for each division
        step = (end - start) / division
        # Create the divided time points within the interval, excluding the endpoint
        divided_times = [start + step * i for i in range(1, division)]
        # Extend the result list
        result.extend(divided_times)
        # Append the interval endpoint
        result.append(end)

    return result


def search_closest_index(input_array, value):
    # Find the index where 'value' should be inserted to maintain order
    idx = np.searchsorted(input_array, value, side="left")
    if (
        idx > 0 and
        (idx == len(input_array) or
         np.abs(value - input_array[idx - 1]) <= np.abs(value - input_array[idx])
        )):
        return idx - 1
    else:
        return idx


class EventsListBuilder:
    bar_part_priority = -5
    # genre_part_priority = -4
    section_part_priority = -3
    tempo_part_priority = -2
    chord_part_priority = -1

    lyrics_part_priority = 0
    inst_priority_dict = {
        "vocal": 1,
        "piano": 2,
        "guitar": 3,
        "bass": 4,
        "drums": 5,
    }

    chord_index_dict = {c: i for i, c in enumerate(CHORD_LABELS)}
    to_chord_event = lambda c: f"chord_{EventsListBuilder.chord_index_dict[c]}"
    to_note_on_event = lambda x: f"note_on_{x}"
    to_note_duration_event = lambda x: f"note_duration_{x}"
    to_drum_event = lambda x: f"drum_{x}"
    to_bpm_level_event = lambda x: f"bpm_level_{bisect(BPM_DIVISIONS, x)}"

    to_phone_event = lambda x: f"phone_{x}"

    stem_index_dict = {s: i for i, s in enumerate(STEM_LABELS)}
    to_stem_indicator_event = lambda s: f"stem_{EventsListBuilder.stem_index_dict[s]}"

    section_index_dict = {s: i for i, s in enumerate(SECTION_LABELS)}
    to_section_indicator_event = lambda x: f"sec_{EventsListBuilder.section_index_dict[x]}"

    to_position_event = lambda x: f"position_{x}"

    get_bar_event = lambda: "bar"
    get_eos_event = lambda: "eos"
    get_eop_event = lambda: "eop"

    def __init__(
        self,
        dfs_dict: Dict[str, Any],
        division: int = 16,
    ):
        self.dfs_dict = dfs_dict
        self.division = division

        self.error_dict = {}
        self.events_list: List[QueuedEvents] = []

        self.initialize_downbeats()

    def initialize_downbeats(self):
        df_beat = self.dfs_dict["df_beat"]
        downbeat_times = df_beat[df_beat.beat == 1].time.values
        if len(downbeat_times) < 2:
            self.error_dict["initialize_downbeats"] = "Not enough downbeats"
        ## Extend the downbeat at beginning and end
        begin_time = downbeat_times[0] * 2 - downbeat_times[1]
        end_time = 2 * downbeat_times[-1] - downbeat_times[-2]
        downbeat_times = np.array([begin_time, *downbeat_times, end_time])
        division_times = np.array(divide_sequence(downbeat_times, self.division))
        measures = np.repeat(np.arange(len(downbeat_times), dtype=np.int16), self.division)[:len(division_times)]
        positions = np.tile(np.arange(self.division, dtype=np.int16), len(downbeat_times))[:len(division_times)]
        self.df_div = pd.DataFrame({"time": division_times, "measure": measures, "position": positions})
        self.df_div_db = self.df_div[self.df_div.position == 0]

    def get_closest_div_row(self, time):
        assert time >= self.df_div.iloc[0].time, "time earlier than the first downbeat"
        assert time <= self.df_div.iloc[-1].time, "time later than the last downbeat"
        ind = search_closest_index(self.df_div.time, time)
        return self.df_div.iloc[ind]

    def add_chord_events(self):
        self._check_error()

        df_beat = self.dfs_dict["df_beat"]
        assert "chord" in df_beat.columns, "Missing chord colum in df_beat"

        prev_chord = None
        prev_measure_index = None
        for i, row in df_beat.iterrows():
            try:
                row_div = self.get_closest_div_row(row.time)
            except AssertionError as e:
                continue
            if row_div.measure == prev_measure_index and row.chord == prev_chord:
                continue
            self.events_list.append(QueuedEvents(
                priority=Priority(
                    measure_index=row_div.measure,
                    part_priority=EventsListBuilder.chord_part_priority,
                    position=row_div.position,
                ),
                events=[EventsListBuilder.to_chord_event(row.chord)],
            ))

            prev_chord = row.chord
            prev_measure_index = row_div.measure
        return self
    
    def add_phoneme_events(self, include_stem_indicator: bool = True):
        self._check_error()
        df_lyrics = self.dfs_dict["df_lyrics"]
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

        prev_measure_index = None
        for i, row in df_lyrics_word.iterrows():
            start, end = row.start_time / 1000, row.end_time / 1000
            try:
                row_div_start = self.get_closest_div_row(start)
                row_div_end = self.get_closest_div_row(end)
            except AssertionError as e:
                # print(repr(e))
                continue
            ## We make sure every word is included
            quantized_duration = max(1, int(
                (row_div_end.measure - row_div_start.measure) * self.division
                + (row_div_end.position - row_div_start.position)
            ))
            # if quantized_duration == 0:
            #     # print(f"too short, remove: {row}")
            #     continue
            self.events_list.append(QueuedEvents(
                priority=Priority(
                    measure_index=row_div_start.measure,
                    part_priority=EventsListBuilder.lyrics_part_priority,
                    position=row_div_start.position,
                ),
                events=[
                    EventsListBuilder.to_note_duration_event(min(quantized_duration, self.division * 2)),
                    *[EventsListBuilder.to_phone_event(p) for p in row.phoneme_ids],
                ],
            ))
            if include_stem_indicator and prev_measure_index != row_div_start.measure:
                self.events_list.append(QueuedEvents(
                    priority=Priority(
                        measure_index=row_div_start.measure,
                        part_priority=EventsListBuilder.lyrics_part_priority,
                        position=-1,
                    ),
                    events=[EventsListBuilder.to_stem_indicator_event("lyrics")]
                ))
                prev_measure_index = row_div_start.measure

        return self

    def add_note_events_from_vocal2midi(
        self,
        include_stem_indicator: bool = True
    ):
        self._check_error()
        df_vocal2midi = self.dfs_dict["df_vocal2midi"]
        prev_measure_index = None
        for _, row in df_vocal2midi.iterrows():
            try:
                row_div_start = self.get_closest_div_row(row.start)
                row_div_end = self.get_closest_div_row(row.end)
            except AssertionError as e:
                continue

            quantized_duration = int(
                (row_div_end.measure - row_div_start.measure) * self.division
                + (row_div_end.position - row_div_start.position)
            )
            if quantized_duration == 0:
                # print(f"too short, remove note: {row}")
                continue

            self.events_list.append(QueuedEvents(
                priority=Priority(
                    measure_index=row_div_start.measure,
                    part_priority=EventsListBuilder.inst_priority_dict["vocal"],
                    position=row_div_start.position,
                ),
                events=[
                    EventsListBuilder.to_note_duration_event(min(quantized_duration, self.division * 2)),
                    EventsListBuilder.to_note_on_event(int(row.pitch)),
                ],
            ))

            if include_stem_indicator and prev_measure_index != row_div_start.measure:
                priority = Priority(
                    measure_index=row_div_start.measure,
                    part_priority=EventsListBuilder.inst_priority_dict["vocal"],
                    position=-1,
                )
                self.events_list.append(QueuedEvents(
                    priority=priority,
                    events=[EventsListBuilder.to_stem_indicator_event("vocal")]
                ))
                prev_measure_index = row_div_start.measure

        return self

    def add_note_events_from_trans5stem(
        self,
        include_drum_events: bool = True,
        include_stem_indicator: bool = True,
        stem_list: List[str] = [],
    ):
        self._check_error()
        df_note = self.dfs_dict["df_note"]
        df_note = df_note[df_note.stem.isin(stem_list)]
        df_note["stem"] = pd.Categorical(
            df_note['stem'], categories=stem_list, ordered=True,
        )
        df_note = df_note.sort_values("stem")
        for inst_name, df_inst in df_note.groupby("stem"):
            inst_priority = EventsListBuilder.inst_priority_dict[inst_name]
            prev_measure_index = -1
            df_inst = df_inst.sort_values(by="start").reset_index(drop=True)
            for i, row in df_inst.iterrows():
                try:
                    row_div_start = self.get_closest_div_row(row.start)
                except AssertionError as e:
                    continue
                if include_drum_events and inst_name == "drums":
                    if row.pitch < 35 or row.pitch > 81:
                        # print(f"drum event out of range, remove: {row}")
                        continue
                    self.events_list.append(QueuedEvents(
                        priority=Priority(
                            measure_index=row_div_start.measure,
                            part_priority=inst_priority,
                            position=row_div_start.position
                        ),
                        events=[
                            EventsListBuilder.to_drum_event(row.pitch),
                        ]
                    ))
                else:
                    try:
                        row_div_end = self.get_closest_div_row(row.end)
                    except AssertionError as e:
                        continue

                    quantized_duration = int(
                        (row_div_end.measure - row_div_start.measure) * self.division
                        + (row_div_end.position - row_div_start.position)
                    )
                    if quantized_duration == 0:
                        # print(f"too short, remove note: {row}")
                        continue
                    
                    self.events_list.append(QueuedEvents(
                        priority=Priority(
                            measure_index=row_div_start.measure,
                            part_priority=inst_priority,
                            position=row_div_start.position,
                        ),
                        events=[
                            EventsListBuilder.to_note_duration_event(min(quantized_duration, self.division * 2)),
                            EventsListBuilder.to_note_on_event(int(row.pitch)),
                        ],
                    ))
                
                if include_stem_indicator and prev_measure_index != row_div_start.measure:
                    priority = Priority(
                        measure_index=row_div_start.measure,
                        part_priority=inst_priority,
                        position=-1,
                    )
                    self.events_list.append(QueuedEvents(
                        priority=priority,
                        events=[EventsListBuilder.to_stem_indicator_event(inst_name)]
                    ))
                    prev_measure_index = row_div_start.measure
        return self

    def add_bar_events(self):
        self._check_error()
        for i, row in self.df_div_db[:-1].iterrows():
            self.events_list.append(QueuedEvents(
                priority=Priority(
                    measure_index=row.measure,
                    part_priority=EventsListBuilder.bar_part_priority,
                    position=0,
                ),
                events=[EventsListBuilder.get_bar_event()]
            ))
        return self
    
    def add_section_events_each_bar(self):
        self._check_error()
        df_section = self.dfs_dict["df_section"]
        cs = [
            *df_section.start_time,
            max(df_section.iloc[-1].end_time, self.df_div_db.iloc[-1].time)
        ]
        ## Exclude pickup bar since the section start time is quantized to
        ## the end of the pickup bar
        indexes = colorful_sequence(cs, self.df_div_db.time.tolist()[1:])
        ## Add back the starting pickup bar
        indexes = [indexes[0], *indexes]
        for i, ind in enumerate(indexes):
            self.events_list.append(QueuedEvents(
                priority=Priority(
                    measure_index=i,
                    part_priority=EventsListBuilder.section_part_priority,
                    position=0,
                ),
                events=[EventsListBuilder.to_section_indicator_event(
                    df_section.iloc[ind].function_name
                )]
            ))
        return self
    
    def add_bpm_level_events(self):
        self._check_error()
        for i in range(len(self.df_div_db) - 1):
            bpm = (self.division / (self.df_div_db.iloc[i + 1].time - self.df_div_db.iloc[i].time) * 15)
            self.events_list.append(QueuedEvents(
                priority=Priority(
                    measure_index=i,
                    part_priority=EventsListBuilder.tempo_part_priority,
                    position=0,
                ),
                events=[EventsListBuilder.to_bpm_level_event(bpm)],
            ))
        return self

    def _check_error(self):
        error_str = "; ".join(
            [f"{k}: {v}" for k, v in self.error_dict.items() if len(v) > 0]
        )
        assert len(error_str) == 0, error_str

    def sort_events_list(self):
        self.events_list.sort(key=lambda x: (
            x.priority.measure_index,
            x.priority.part_priority,
            x.priority.position,
        ))
        return self

    def create_event_str_seq(self):
        event_str_seq = []
        for events in self.events_list:
            first_event = events.events[0]
            if first_event.split("_")[0] in ["chord", "note", "phone", "drum"]:
                event_str_seq.append(EventsListBuilder.to_position_event(events.priority.position))
            event_str_seq.extend(events.events)
        event_str_seq.append(EventsListBuilder.get_eos_event())
        return event_str_seq
