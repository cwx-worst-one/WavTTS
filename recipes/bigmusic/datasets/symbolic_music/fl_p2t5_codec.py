from __future__ import annotations
import os
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
import webdataset as wds

from samantha.dataio.parquet import ParquetDataset
from recipes.musiclm.datamodules.webdataset import return_self
from recipes.bigmusic.datasets.symbolic_music.base_codec import (
    get_indexer,
    IndexerConfig,
    BaseCodec,
)
from recipes.datasets.mcc.sami_tokenizer import (
    convert_labels_to_text_id,
)
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import (
    BMDfsDictBuilder,
)
from recipes.bigmusic.datasets.symbolic_music.events_list_builder import (
    EventsListBuilder,
)


class FLP2T5Codec(BaseCodec):
    class Config(IndexerConfig):
        lyrics_seq_len: int
        leadsheet_seq_len: int

        ## Override default configs
        include_utterance_phoneme_tokens: bool = True
        include_phoneme: bool = True
        include_section_indicator_each_bar: bool = True
        include_drum_events: bool = True
        stems: str = "vocal,piano,guitar,bass,drums"

    def __init__(
        self,
        config: FLP2T5Codec.Config,
    ):
        self.config = config
        self.indexer = get_indexer(self.config)

        self._debug_info = {}
    
    def encode_lyrics(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        df_lyrics = dfs_dict["df_lyrics"]
        result = df_lyrics["phoneme"].apply(lambda x: convert_labels_to_text_id(x.split("\n")))
        lyrics_tokens = np.concatenate(result.apply(lambda x: x[0][0]))
        lyrics_tokens = np.append(lyrics_tokens, self.indexer["eol"])
        self._debug_info["original_lyrics_seq_len"] = len(lyrics_tokens)
        return lyrics_tokens

    def encode_leadsheet(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        builder = (
            EventsListBuilder(dfs_dict)
            .add_chord_events()
            .add_note_events_from_trans5stem(
                include_drum_events=self.config.include_drum_events,
                include_stem_indicator=self.config.include_stem_indicator,
                stem_list=self.config.stems.split(","),
            )
            .add_bar_events()
        )
        if self.config.include_phoneme:
            builder.add_phoneme_events(self.config.include_stem_indicator)
        if self.config.include_bpm_levels:
            builder.add_bpm_level_events()
        if self.config.include_section_indicator_each_bar:
            builder.add_section_events_each_bar()
        builder.sort_events_list()

        event_str_seq = builder.create_event_str_seq()
        leadsheet_tokens = [self.indexer[es] for es in event_str_seq]
        self._debug_info["original_leadsheet_seq_len"] = len(leadsheet_tokens)
        return np.array(leadsheet_tokens)

    def encode(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        try:
            leadsheet_tokens = self.chop_or_pad(
                self.encode_leadsheet(dfs_dict),
                self.config.leadsheet_seq_len,
            )
            if self.config.include_utterance_phoneme_tokens:
                lyrics_tokens = self.chop_or_pad(
                    self.encode_lyrics(dfs_dict),
                    self.config.lyrics_seq_len,
                )
                return np.concatenate([lyrics_tokens, leadsheet_tokens])
            return leadsheet_tokens
        except Exception as e:
            print(repr(e))
            from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip

    @classmethod
    def get_datamodule(
        cls,
        config: FLP2T5Codec.Config,
        train_id: int,
        val_id: int,
        batch_size: int,
        num_workers: int,
        shuffle_buffer: int = 10,
    ) -> pl.LightningDataModule:
        """Example 1:
        lyrics_seq_len = 7; leadsheet_seq_len = 5
        tokens: [l0, l1, l2, l3, eol pad pad][s0, s1, eos, pad, pad]

        input_id: l0, l1, l2, l3, eol pad pad, s0, s1, eos, pad
        target_id: s1, eos, pad, pad
        target_length: 2

        From base_modules.py:

        ```
        x = logits[:, -target_ids.size(1):, :]
        ```

        x will be the logits of inputs s0, s1, eos, pad

        ```
        loss_mask = sequence_mask(
            training_inputs['target_lengths'], max_len=target_ids.shape[1], device=target_ids.device)
        ```

        loss will be computed on the first 2 tokens
        x[:2] -> target_ids[:2]

        which is

        s0, s1 -> s1, eos

        Example 2:
        lyrics_seq_len = 0; leadsheet_seq_len = 5
        tokens: [][s0, s1, eos, pad, pad]

        input_id: s0, s1, eos, pad
        target_id: s1, eos, pad, pad
        target_length: 2

        x: logits of s0, s1, eos, pad
        x[:2] -> target_ids[:2], same as example 1

        So this should work for lyrics -> leadsheet and
        leadsheet modeling alone.
        """
        codec = cls(config=config)

        def preproc_each_sample(sample):
            builder = (
                BMDfsDictBuilder(sample)
                .pre_load_meta()
                .add_df_note(subsets=config.stems.split(","))
                .add_df_beat()
                .add_df_chord()
                .add_key()
                .quantize_chord_to_beat()
            )
            if config.include_phoneme:
                builder.add_df_lyrics()
            if config.include_section_indicator_each_bar:
                builder.add_df_section().quantize_section_to_downbeat()
            dfs_dict = builder.create_output()
            tokens = codec.encode(dfs_dict)

            ## Convert tokens to required model input format
            leadsheet_tokens = tokens[-config.leadsheet_seq_len:]
            pad_index = codec.indexer["pad"]
            pad_occ = np.where(leadsheet_tokens == pad_index)[0]
            target_length = pad_occ[0] if len(pad_occ) > 0 else config.leadsheet_seq_len

            ## BaseModule requires the following nested data as input
            return {
                "model_inputs": {
                    "input_ids": tokens[:-1],
                },
                "target_ids": leadsheet_tokens[1:],
                "target_lengths": target_length - 1,

                **codec._debug_info,
            }

        train_dataset = ParquetDataset(
            data_id=train_id,
            resampled=True,
        ).shuffle(shuffle_buffer).map(
            preproc_each_sample,
            handler=wds.warn_and_continue,
        )
        val_dataset = ParquetDataset(
            data_id=val_id,
            nodesplitter=return_self,
        ).map(
            preproc_each_sample,
            handler=wds.warn_and_continue,
        )
        return pl.LightningDataModule.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )