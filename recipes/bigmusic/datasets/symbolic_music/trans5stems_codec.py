from __future__ import annotations
import os
from typing import Dict, Any, List, Optional

import numpy as np

from recipes.bigmusic.utils.common_utils import colorful_sequence
from recipes.bigmusic.datasets.symbolic_music.base import (
    SymbolicMusicCodecBase,
    QueuedEvents,
    Priority,
)
from recipes.bigmusic.datasets.symbolic_music.decorators import (
    split_section,
    skip_by,
    skip_keys,
    until,
)
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder


class Trans5StemsCodec(SymbolicMusicCodecBase):
    inst_priority_dict = {
        "vocal": 1,
        "piano": 2,
        "guitar": 3,
        "bass": 4,
        "drums": 5,
    }

    def encode_leadsheet(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        df_note = dfs_dict["df_note"]
        df_beat = dfs_dict["df_beat"]

        downbeat_times = df_beat[df_beat.beat == 1].time.values
        if len(downbeat_times) < 2:
            ## piece too short, skip
            return np.empty(0, dtype=np.uint16)    ## Extend one downbeat to cover notes and chords at the end
        assert "chord" in df_beat.columns, "Missing chord colum in df_beat"

        min_time = 2 * downbeat_times[0] - downbeat_times[1]
        max_time = 2 * downbeat_times[-1] - downbeat_times[-2]

        ## 目前把每小节切分为16份（如果是3拍，切12份可能更合理，不过这需要很多额外逻辑）
        # divisions = 16 if time_sig_num == 4 else 12
        divisions = 16

        events_list: List[QueuedEvents] = []

        ## 1. Chord events
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

        ## 2. Note events
        for inst_name, df_inst in df_note.groupby("stem"):
            inst_priority = Trans5StemsCodec.inst_priority_dict[inst_name]
            prev_measure_index = -1
            df_inst = df_inst.sort_values(by="start").reset_index(drop=True)
            for i, note in df_inst.iterrows():
                qe = (
                    self.get_drum_queued_events(
                        note,
                        downbeat_times,
                        inst_priority,
                        divisions,
                        min_time,
                        max_time,
                    ) if self.config.include_drum_events and inst_name == "drums"
                    else self.get_note_queued_events(
                        note,
                        downbeat_times,
                        inst_priority,
                        divisions,
                        min_time,
                        max_time,
                    )
                )
                if qe is None:
                    continue
                measure_index = qe.priority.measure_index
                ## When coming to a new measure, add a stem indicator event first
                ## position value for stem indicator is -1
                if self.config.include_stem_indicator and prev_measure_index != measure_index:
                    priority = Priority(
                        measure_index=measure_index,
                        part_priority=inst_priority,
                        position=-1,
                    )
                    events_list.append(QueuedEvents(
                        priority=priority,
                        events=[SymbolicMusicCodecBase.to_stem_indicator_event(inst_name)]
                    ))
                    prev_measure_index = measure_index
                events_list.append(qe)

        ## Fix last bar (we want to make sure notes at the end does not go beyond the last bar)
        ## If so we extend 1 bar. If everything is already in creating dfs_dict, we should
        ## need at most 1 bar extension.
        max_mi = max(map(lambda x: x.priority.measure_index, events_list))
        if max_mi >= len(downbeat_times):
            ## Extend downbeat_times
            assert max_mi == len(downbeat_times), "Note exceed more than one bar. Need to debug"
            extended_last_db_time = 2 * downbeat_times[-1] - downbeat_times[-2]
            downbeat_times = np.append(downbeat_times, extended_last_db_time)

        ## 3. Bar events
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

        ## 4. Section indicators
        if self.config.include_section_indicator_each_bar:
            df_section = dfs_dict["df_section"]
            us = [*downbeat_times, 2 * downbeat_times[-1] - downbeat_times[-2]]
            cs = [*df_section.start_time, max(df_section.iloc[-1].end_time, us[-1])]
            indexes = colorful_sequence(cs, us)
            assert -1 not in indexes, "Impossible!!! Need to debug"
            for i, ind in enumerate(indexes):
                events_list.append(QueuedEvents(
                    priority=Priority(
                        measure_index=i,
                        part_priority=SymbolicMusicCodecBase.section_part_priority,
                        position=0,
                    ),
                    events=[SymbolicMusicCodecBase.to_section_indicator_event(
                        df_section.iloc[ind].function_name
                    )]
                ))

        ## 5. BPM
        if self.config.include_bpm_levels:
            for i in range(len(downbeat_times) - 1):
                bpm = divisions / (downbeat_times[i + 1] - downbeat_times[i]) * 15
                events_list.append(QueuedEvents(
                    priority=Priority(
                        measure_index=i,
                        part_priority=SymbolicMusicCodecBase.tempo_part_priority,
                        position=0,
                    ),
                    events=[SymbolicMusicCodecBase.to_bpm_level_event(bpm)],
                ))
            events_list.append(QueuedEvents(
                priority=Priority(
                    measure_index=len(downbeat_times) - 1,
                    part_priority=SymbolicMusicCodecBase.tempo_part_priority,
                    position=0,
                ),
                events=[SymbolicMusicCodecBase.to_bpm_level_event(bpm)],
            ))

        events_list.sort(key=lambda x: (
            x.priority.measure_index,
            x.priority.part_priority,
            x.priority.position,
        ))

        return events_list
        from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip

        event_str_seq = []

        if self.config.include_prompt:
            prompt_events_list: List[QueuedEvents] = []
            for e in events_list:
                first_event = e.events[0]
                if first_event.split("_")[0] in [
                    "bar", "sec", "bpm", "chord", "stem", "genre",
                ]:
                # if first_event.split("_")[0] in ["bar", "sec", "bpm", "chord"]:
                    prompt_events_list.append(e)

            ## Also add position events to chord (there is actually no note or drum events
            ## but we keep them here for consistency)
            for events in prompt_events_list:
                first_event = events.events[0]
                if first_event.split("_")[0] in ["chord", "note", "drum"]:
                    event_str_seq.append(
                        SymbolicMusicCodecBase.to_position_event(events.priority.position)
                    )
                event_str_seq.extend(events.events)
            event_str_seq.append(SymbolicMusicCodecBase.get_eop_event())

        for events in events_list:
            first_event = events.events[0]
            if first_event.split("_")[0] in ["chord", "note", "drum"]:
                event_str_seq.append(
                    SymbolicMusicCodecBase.to_position_event(events.priority.position)
                )
            event_str_seq.extend(events.events)
        event_str_seq.append(SymbolicMusicCodecBase.get_eos_event())

        leadsheet_tokens = [self.indexer[es] for es in event_str_seq]
        return np.array(leadsheet_tokens, dtype=np.uint16)

    def encode(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        return self.encode_leadsheet(dfs_dict)

    @staticmethod
    def get_dfs_dict_iter(
        data_ids: List[int],
        num_samples: int = -1,
        skip_ratio: float = 0.0,
        note_subsets: List[str] = ["vocal"],
        skip_keys_set: set = set(),
        seed: Optional[float] = None,
        selected_sec: Optional[List[str]] = None,
        reset_start_time: bool = True,
    ):
        from samantha.dataio.parquet import ParquetDataset
        from samantha.dataio.dataset import MultiIterableDataset

        def base_iter():
            dataset = MultiIterableDataset(datasets=[
                ParquetDataset(data_id=i) for i in data_ids
            ])
            data_iter = dataset.__iter__()
            for i, sample in enumerate(data_iter):
                try:
                    builder = (BMDfsDictBuilder(sample)
                        .pre_load_meta()
                        .add_key()
                        .add_df_note(subsets=note_subsets)
                        .add_df_beat()
                        .add_df_section()
                        .add_df_chord()
                        .quantize_chord_to_beat()
                        .quantize_section_to_downbeat()
                    )
                    # if include_original_audio:
                    #     builder.add_audio()
                    # if include_tagging:
                    #     builder.add_tagging_json()
                    dfs_dict = builder.create_output()

                except Exception as e:
                    print(e)
                    continue
                dfs_dict["index_dict"] = {
                    "o": i,
                }
                yield dfs_dict

        @until(num_samples)
        @skip_by(skip_ratio=skip_ratio, seed=seed)
        @skip_keys(keys=skip_keys_set)
        def my_iter():
            if selected_sec is not None:
                modified_iter = split_section(
                    selected_sec,
                    reset_start_time=reset_start_time,
                )(base_iter)
            else:
                modified_iter = base_iter
            for dfs_dict in modified_iter():
                yield dfs_dict

        return my_iter()
