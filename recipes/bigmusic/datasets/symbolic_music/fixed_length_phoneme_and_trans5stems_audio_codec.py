import math
import torch

import webdataset as wds
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Any, List
from recipes.bigmusic.utils.format_utils import normalize_text

from samantha.utils.webdataset import return_self
from samantha.dataio.parquet import ParquetDataset

from recipes.bigmusic.datasets.svs import pad_crop
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from recipes.bigmusic.datasets.symbolic_music.base_codec import IndexerConfig
from recipes.bigmusic.datasets.symbolic_music.fixed_length_trans5stem_and_lyric2audio_codec import FixedLengthTrans5StemsAndLyric2AudioCodec
from recipes.bigmusic.utils.common_utils import colorful_sequence
from recipes.bigmusic.datasets.symbolic_music.base import (
    SymbolicMusicCodecBase,
    QueuedEvents,
    Priority,
)
from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id
from recipes.bigmusic.datasets.svs import rewrite_metadata


class FixedLengthTransPhone25Stems2AudioCodec(FixedLengthTrans5StemsAndLyric2AudioCodec):
    duration = 0
    lentoken = 0
    passed = 0
    counter = 0

    @classmethod
    def get_datamodule(
        cls,
        config: IndexerConfig,
        train_id: int,
        val_id: int,
        batch_size: int,
        num_workers: int,
        shuffle_buffer: int = 10,
        fast_eval: bool = False,
    ) -> pl.LightningDataModule:
        codec = FixedLengthTransPhone25Stems2AudioCodec(config=config)

        def preproc_each_sample(sample):
            if cls.counter > 1e12:
                cls.counter = 0
            cls.counter += 1
            if fast_eval and cls.passed > 10:
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
            

            # Randomly select a sentence
            # And use all preceding sentences as training samples, including itself.
            last_sentence = codec.select_sentence_endpoint(dfs_dict, 
                                    min_sec=30, 
                                    max_sec=config.audio_max_duration
                                    )
            if last_sentence is None:
                return
            start_sec = 0
            end_sec = last_sentence.loc[0, 'end_time'] / 1000

            # original audio length
            audio, sample_rate = dfs_dict["audio"]
            audio = audio.squeeze(0)
            audio_len =  len(audio)
            original_target_tokens_length = math.ceil(
                audio_len / sample_rate * config.semantic_frame_rate
            )
            audio_duration = audio_len / sample_rate
            if audio_duration < config.audio_max_duration:
                return 
            # Crop audio
            audio = audio[int(start_sec*config.sample_rate):int(end_sec*config.sample_rate)]
            audio_len =  len(audio)
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

            # Extract leadsheet and lyric tokens accordingly
            lyrics_tokens, lyrics = codec.encode_lyrics(dfs_dict, start_sec=start_sec, end_sec=end_sec)
            if len(lyrics_tokens) == 0 or len(lyrics) <= 1:
                return

            tokens = codec.encode_leadsheet(dfs_dict, start_sec=start_sec, end_sec=end_sec)
            if tokens is None or len(tokens) == 0:
                return

            # If token is longer than limit, we drop the sample to avoid clipping
            if len(lyrics_tokens) > config.lyrics_seq_len:
                # print("len(lyrics_tokens)", len(lyrics_tokens))
                return
            if  len(tokens) > config.leadsheet_seq_len-1:
                # print("len(tokens)", len(tokens))
                return

            if cls.passed > 1e12:
                cls.passed = 0
            cls.passed += 1
            if cls.passed/cls.counter < 0.3 and cls.counter > 100:
                print(f"[fixed_length_phoneme_and_trans5stems_audio_codec.py] Only {100*cls.passed/cls.counter}% of example is using for training")
            ## Convert lyrics_tokens to required model input format
            lyrics_tokens = codec.chop_or_pad(arr=lyrics_tokens,
                target_len=config.lyrics_seq_len,
                constant_values=0,
            )
            lyrics_tokens = torch.LongTensor(lyrics_tokens)

            ## Convert tokens to required model input format
            leadsheet_tokens = codec.chop_or_pad(arr=tokens[:-1],
                target_len=config.leadsheet_seq_len - 1,
            )
            leadsheet_tokens = torch.LongTensor(np.append(leadsheet_tokens, codec.indexer["eos"]))


            uttid = sample['uttid']
            metadata = sample["meta"]
            style_text = rewrite_metadata(metadata)
            lyrics_normalized_text = normalize_text(lyrics)
            ## BaseModule requires the following nested data as input
            data = {
                "uttid": uttid,
                "lyrics_tokens": lyrics_tokens,
                "lyrics": lyrics,
                "remi_leadsheet_tokens": leadsheet_tokens,
                "original_remi_leadsheet_tokens_length": len(tokens),
                "target_audio": audio,
                "style_text": style_text,
                "lyrics_normalized_text": lyrics_normalized_text,
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

