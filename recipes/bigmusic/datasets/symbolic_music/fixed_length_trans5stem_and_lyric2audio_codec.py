from __future__ import annotations
import math
import copy
import torch

import webdataset as wds
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Any, List

from samantha.utils.webdataset import return_self
from samantha.dataio.parquet import ParquetDataset

from recipes.bigmusic.datasets.svs import pad_crop
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from recipes.bigmusic.datasets.symbolic_music.base_codec import IndexerConfig
from recipes.bigmusic.datasets.symbolic_music.trans5stems_codec import Trans5StemsCodec
from recipes.bigmusic.utils.common_utils import colorful_sequence
from recipes.bigmusic.datasets.symbolic_music.base import (
    SymbolicMusicCodecBase,
    QueuedEvents,
    Priority,
)
from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id
from recipes.bigmusic.datasets.svs import rewrite_metadata

class FixedLengthTrans5StemsAndLyric2AudioCodec(Trans5StemsCodec):
    duration = 0
    lentoken = 0
    number = 0

    class Config(IndexerConfig):
        lyrics_seq_len: int
        leadsheet_seq_len: int
        sample_rate: int
        conditions: str
        semantic_frame_rate: int
        audio_key: str
        audio_max_duration: int = 60
        include_utterance_phoneme_tokens: bool = False
        include_drum_events: bool = True
        include_prompt: bool = False
        include_section_indicator_each_bar: bool = True
        include_bpm_levels: bool = True
        stems: str = "vocal,piano,guitar,bass,drums"


    def encode_leadsheet(self, dfs_dict_origin: Dict[str, Any], start_sec=0, end_sec=10000) -> np.ndarray:
        dfs_dict = copy.deepcopy(dfs_dict_origin)
        start_ms = math.floor(start_sec * 1000)
        end_ms = math.ceil(end_sec * 1000)
        df_note = dfs_dict["df_note"]
        df_beat = dfs_dict["df_beat"]
        df_lyrics = dfs_dict["df_lyrics"]

        df_note = df_note[df_note["start"]>=start_sec]
        df_note = df_note[df_note["end"]<=end_sec]
        df_note = df_note.reset_index(drop=True, inplace=False)

        df_beat = df_beat[df_beat["time"]>=start_sec]
        df_beat = df_beat[df_beat["time"]<=end_sec]
        df_beat = df_beat.reset_index(drop=True, inplace=False)

        df_lyrics = df_lyrics[df_lyrics["start_time"]>=start_ms]
        df_lyrics = df_lyrics[df_lyrics["end_time"]<=end_ms]
        df_lyrics = df_lyrics.reset_index(drop=True, inplace=False)
        if len(df_lyrics) == 0 or len(df_note) == 0 or len(df_beat) == 0:
            return None

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
        return np.array(leadsheet_tokens)



    def encode_leadsheet_wo_lyric(self, dfs_dict_origin: Dict[str, Any], start_sec=0, end_sec=10000) -> np.ndarray:
        dfs_dict = copy.deepcopy(dfs_dict_origin)
        start_ms = math.floor(start_sec * 1000)
        end_ms = math.ceil(end_sec * 1000)
        df_note = dfs_dict["df_note"]
        df_beat = dfs_dict["df_beat"]

        df_note = df_note[df_note["start"]>=start_sec]
        df_note = df_note[df_note["end"]<=end_sec]
        df_note = df_note.reset_index(drop=True, inplace=False)

        df_beat = df_beat[df_beat["time"]>=start_sec]
        df_beat = df_beat[df_beat["time"]<=end_sec]
        df_beat = df_beat.reset_index(drop=True, inplace=False)

        if len(df_note) == 0 or len(df_beat) == 0:
            return None

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
        return np.array(leadsheet_tokens)



    @classmethod
    def get_datamodule(
        cls,
        config: FixedLengthTrans5StemsAndLyric2AudioCodec.Config,
        train_id: int,
        val_id: int,
        batch_size: int,
        num_workers: int,
        shuffle_buffer: int = 10,
        fast_eval: bool = False,
    ) -> pl.LightningDataModule:
        codec = cls(config=config)

        def preproc_each_sample(sample):

            cls.number += 1
            if fast_eval and cls.number > 10:
                return None

            dfs_dict = BMDfsDictBuilder(sample)\
                .pre_load_meta()\
                .add_df_note(subsets=config.stems.split(","))\
                .add_df_lyrics()\
                .add_df_beat()\
                .add_df_section()\
                .add_df_chord()\
                .quantize_chord_to_beat()\
                .quantize_section_to_downbeat()\
                .add_audio(audio_key=config.audio_key)\
                .create_output()
            tokens = codec.encode_leadsheet(dfs_dict)

            audio, sample_rate = dfs_dict["audio"]
            audio = audio.squeeze(0)
            # original audio length
            audio_len =  len(audio)
            original_target_tokens_length = math.ceil(
                audio_len / sample_rate * config.semantic_frame_rate
            )
            # cls.duration += audio_len / sample_rate
            # cls.lentoken += len(tokens)
            # print("token rate", int(cls.lentoken/cls.duration), cls.duration) #for debug

            metadata = sample["meta"]
            style_text = rewrite_metadata(metadata)

            # Crop audio length
            clip_audio_len = min(
                math.ceil(config.audio_max_duration * sample_rate), audio_len
            )
            target_tokens_length = math.floor(
                clip_audio_len / sample_rate * config.semantic_frame_rate
            )

            # Pad or crop audio array
            pad_audio_len = math.ceil(config.audio_max_duration * sample_rate)
            audio = codec.chop_or_pad(arr=audio,
                target_len=pad_audio_len,
                constant_values=0.0,
            )
            ## Convert tokens to required model input format
            leadsheet_tokens = codec.chop_or_pad(arr=tokens[:-1],
                target_len=config.leadsheet_seq_len - 1,
            )
            leadsheet_tokens = torch.LongTensor(np.append(leadsheet_tokens, codec.indexer["eos"]))

            ## BaseModule requires the following nested data as input
            data = {
                "remi_leadsheet_tokens": leadsheet_tokens,
                "original_remi_leadsheet_tokens_length": len(tokens),
                "target_audio": audio,
                "style_text": style_text,
                "target_tokens_length": target_tokens_length,
                "original_target_tokens_length": original_target_tokens_length,
                "conditions": config.conditions,
            }
            return data

        train_dataset = ParquetDataset(data_id=train_id,
                nodesplitter=return_self).shuffle(shuffle_buffer).map(
            preproc_each_sample,
            handler=wds.warn_and_continue,
        )
        predict_dataset = ParquetDataset(data_id=val_id,
                nodesplitter=return_self).map(
            preproc_each_sample,
            handler=wds.warn_and_continue,
        )
        val_dataset = ParquetDataset(data_id=val_id,
                nodesplitter=return_self).map(
            preproc_each_sample,
            handler=wds.warn_and_continue,
        )
        return pl.LightningDataModule.from_datasets(
            train_dataset=train_dataset,
            predict_dataset=predict_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

