from __future__ import annotations
import torch

import webdataset as wds
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Any

from samantha.utils.webdataset import return_self
from samantha.dataio.parquet import ParquetDataset

from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from recipes.bigmusic.datasets.symbolic_music.fl_t52a_codec import FLT52ACodec
# from recipes.bigmusic.datasets.svs import rewrite_metadata


class FLP2T52ACodec(FLT52ACodec):
    class Config(FLT52ACodec.Config):
        include_utterance_phoneme_tokens: bool = True

    @classmethod
    def get_datamodule(
        cls,
        config: FLP2T52ACodec.Config,
        train_id: int,
        val_id: int,
        batch_size: int,
        num_workers: int,
        shuffle_buffer: int = 10,
    ) -> pl.LightningDataModule:
        """
        """
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
                .quantize_chord_to_beat()
                .quantize_section_to_downbeat()
                .add_audio(audio_key=config.audio_key)
                .create_output()
            )
            ## Step 1: lyrics
            lyrics_tokens = codec.chop_or_pad(
                codec.encode_lyrics(dfs_dict),
                config.lyrics_seq_len,
            )
            # lyrics_tokens = np.append(lyrics_tokens, codec.indexer["eol"])
            # lyrics_tokens = torch.LongTensor(lyrics_tokens)

            ## Step 2: leadsheet
            tokens = codec.encode_leadsheet(dfs_dict)
            leadsheet_tokens = codec.chop_or_pad(arr=tokens,
                target_len=config.leadsheet_seq_len,
            )

            ## Step 3: audio
            audio, original_target_tokens_length, target_tokens_length = codec.get_audio(dfs_dict)

            data = {
                "lyrics_tokens": lyrics_tokens,
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
