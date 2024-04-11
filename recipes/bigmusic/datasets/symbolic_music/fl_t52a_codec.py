from __future__ import annotations
import math
import torch

import webdataset as wds
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Any

from samantha.utils.webdataset import return_self
from samantha.dataio.parquet import ParquetDataset

from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from recipes.bigmusic.datasets.symbolic_music.fl_t5_codec import FLT5Codec
# from recipes.bigmusic.datasets.svs import rewrite_metadata


class FLT52ACodec(FLT5Codec):
    class Config(FLT5Codec.Config):
        audio_max_duration: int
        sample_rate: int = 24000
        conditions: str = "remi_leadsheet_tokens"
        semantic_frame_rate: int = 25
        audio_key: str = "wav"

        include_phoneme: bool = True

    def get_audio(self, dfs_dict: Dict[str, Any]):
        audio, sample_rate = dfs_dict["audio"]
        audio = audio.squeeze(0)
        # original audio length
        audio_len = len(audio)
        original_target_tokens_length = math.ceil(
            audio_len / sample_rate * self.config.semantic_frame_rate
        )
        # cls.duration += audio_len / sample_rate
        # cls.lentoken += len(tokens)
        # print("token rate", int(cls.lentoken/cls.duration), cls.duration) #for debug
        # Crop audio length
        pad_audio_len = math.ceil(self.config.audio_max_duration * sample_rate)
        clip_audio_len = min(pad_audio_len, audio_len)
        target_tokens_length = math.floor(
            clip_audio_len / sample_rate * self.config.semantic_frame_rate
        )

        # Pad or crop audio array
        audio = self.chop_or_pad(arr=audio,
            target_len=pad_audio_len,
            constant_values=0.0,
        )
        return audio, original_target_tokens_length, target_tokens_length

    @classmethod
    def get_datamodule(
        cls,
        config: FLT52ACodec.Config,
        train_id: int,
        val_id: int,
        batch_size: int,
        num_workers: int,
        shuffle_buffer: int = 10,
    ) -> pl.LightningDataModule:
        codec = cls(config=config)

        def preproc_each_sample(sample):
            dfs_dict = (
                BMDfsDictBuilder(sample)
                .pre_load_meta()
                .add_df_note(subsets=config.stems.split(","))
                .add_df_lyrics()
                .add_df_beat()
                .add_df_section()
                .add_df_chord()
                .add_audio(audio_key=config.audio_key)
                .quantize_chord_to_beat()
                .quantize_section_to_downbeat()
                .create_output()
            )
            tokens = codec.encode_leadsheet(dfs_dict)
            ## Convert tokens to required model input format
            leadsheet_tokens = codec.chop_or_pad(arr=tokens,
                target_len=config.leadsheet_seq_len,
            )

            audio, original_target_tokens_length, target_tokens_length = codec.get_audio(dfs_dict)

            # metadata = sample["meta"]
            # style_text = rewrite_metadata(metadata)

            ## BaseModule requires the following nested data as input
            data = {
                "remi_leadsheet_tokens": leadsheet_tokens,
                "original_remi_leadsheet_tokens_length": len(tokens),
                "target_audio": audio,
                # "style_text": style_text,
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

