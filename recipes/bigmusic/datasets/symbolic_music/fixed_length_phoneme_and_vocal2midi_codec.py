from __future__ import annotations
import os
from typing import Dict, Any

import numpy as np
import webdataset as wds
import pytorch_lightning as pl

from samantha.dataio.parquet import ParquetDataset
from recipes.bigmusic.datasets.symbolic_music.base import (
    SymbolicMusicCodecBase,
    IndexerConfig,
)
from recipes.musiclm.datamodules.webdataset import return_self
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from recipes.bigmusic.inference.controllers import ControllerBase


class FixedLengthPhonemeAndVocal2MidiCodec(SymbolicMusicCodecBase):
    class Config(IndexerConfig):
        lyrics_seq_len: int
        leadsheet_seq_len: int

    def encode(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        try:
            leadsheet_tokens = self.chop_or_pad(
                self.encode_leadsheet(dfs_dict)[:-1],
                self.config.leadsheet_seq_len - 1,
            )
            leadsheet_tokens = np.append(leadsheet_tokens, self.indexer["eos"])
            if self.config.include_utterance_phoneme_tokens:
                lyrics_tokens = self.chop_or_pad(
                    self.encode_lyrics(dfs_dict)[:-1],
                    self.config.lyrics_seq_len - 1,
                )
                lyrics_tokens = np.append(lyrics_tokens, self.indexer["eol"])
                return np.concatenate([lyrics_tokens, leadsheet_tokens])
            return leadsheet_tokens
        except Exception as e:
            print(repr(e))
            from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip

    @classmethod
    def get_datamodule(
        cls,
        config: FixedLengthPhonemeAndVocal2MidiCodec.Config,
        train_id: int,
        val_id: int,
        batch_size: int,
        num_workers: int,
        shuffle_buffer: int = 10,
    ) -> pl.LightningDataModule:
        codec = cls(config=config)

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
            target_length = pad_occ[0] + 1 if len(pad_occ) > 0 else config.leadsheet_seq_len

            ## BaseModule requires the following nested data as input
            return {
                "model_inputs": {
                    "input_ids": tokens[:-1],
                },
                "target_ids": leadsheet_tokens,
                "target_lengths": target_length,
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