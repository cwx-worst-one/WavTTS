from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pytorch_lightning as pl

from samantha.dataio.parquet import ParquetDataset

from recipes.bigmusic.datasets.symbolic_music.base import (
    SymbolicMusicCodecBase,
    IndexerConfig,
    QueuedEvents,
    Priority,
)
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder


class MelodyOnlyCodec(SymbolicMusicCodecBase):
    class Config(IndexerConfig):
        inclue_phoneme: bool = False
        leadsheet_seq_len: int

    def encode_leadsheet(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        df_vocal2midi = dfs_dict["df_vocal2midi"]
        df_beat = dfs_dict["df_beat"]

        downbeat_times = df_beat[df_beat.beat == 1].time.values
        if len(downbeat_times) < 2:
            ## piece too short, skip
            return np.empty(0, dtype=np.uint16)    ## Extend one downbeat to cover notes and chords at the end
        assert "chord" in df_beat.columns, "Missing chord colum in df_beat"

        min_time = 2 * downbeat_times[0] - downbeat_times[1]
        max_time = 2 * downbeat_times[-1] - downbeat_times[-2]

        ## Prepare events list
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
        leadsheet_tokens = self.chop_or_pad(
            self.encode_leadsheet(dfs_dict)[:-1],
            self.config.leadsheet_seq_len - 1,
        )
        leadsheet_tokens = np.append(leadsheet_tokens, self.indexer["eos"])
        return leadsheet_tokens
    
    @classmethod
    def get_datamodule(
        cls,
        config: MelodyOnlyCodec.Config,
        train_id: int,
        val_id: int,
        batch_size: int,
        num_workers: int,
        shuffle_buffer: int = 10,
    ) -> pl.LightningDataModule:
        codec = MelodyOnlyCodec(config=config)

        def preproc_each_sample(sample):
            dfs_dict = BMDfsDictBuilder(sample)\
                .pre_load_meta()\
                .add_df_lyrics()\
                .add_df_vocal2midi()\
                .add_df_beat()\
                .add_df_chord()\
                .add_key()\
                .quantize_chord_to_beat()\
                .create_output()
            tokens = codec.encode(dfs_dict)

            ## Convert tokens to required model input format
            leadsheet_tokens = tokens[-config.leadsheet_seq_len:]
            pad_index = codec.indexer["pad"]
            pad_occ = np.where(leadsheet_tokens == pad_index)[0]
            target_length = pad_occ[0] if len(pad_occ) > 0 else config.leadsheet_seq_len - 1

            ## BaseModule requires the following nested data as input
            return {
                "model_inputs": {
                    "input_ids": tokens[:-1],
                },
                "target_ids": tokens[1:],
                "target_lengths": target_length,
            }

        train_dataset = ParquetDataset(data_id=train_id).shuffle(shuffle_buffer).map(
            preproc_each_sample,
        )
        val_dataset = ParquetDataset(data_id=val_id).map(
            preproc_each_sample,
        )
        return pl.LightningDataModule.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
