from typing import Callable, List, Optional
import torch
from pathlib import Path
import webdataset as wds
import pytorch_lightning as pl
from recipes.bigmusic.datasets.transforms.lyrics import (
    LyricsTokenTransform, 
    VocalChromaTransform, 
    MCCMetadataTextTransform, 
    StyleTextT5Transform, 
    AddConditionsTransform,
    RenameAudioKeyTransform,
)
from recipes.bigmusic.datasets.transforms.lyrics_segment import LyricsSegmentTransforms, crop_pad_to_seq_length
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from webdataset.pipeline import DataPipeline
from torch.utils.data import DataLoader
from samantha.dataio.dataset import MultiIterableDataset
from recipes.musiclm.datamodules.webdataset import return_self
from torch.utils.data import default_collate
from torch.utils.data._utils.collate import collate, collate_tensor_fn, default_collate_fn_map
from recipes.musiclm.datasets.mcc import WrappedMCC40MDataset
from recipes.datasets.mcc.mix import MCCInstrumentalDataset
from samantha.dataio.batching import BucketBatcher
from recipes.bigmusic.datasets.index_lists import INDEX
from samantha.utils.hdfs_tools import hdfs_open, hdfs_loadtxt
from recipes.datasets.mcc.mix import LibriLightASRDataset, LibriTTSDataset
class LyricsDataset(WebPipeline):
    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        sample_duration: float,
        audio_keys: dict,
        audio_format: str = "mp3",
        max_num_segments: int = 10,
        shuffle_segments: bool = True,
        resampled: bool =True,
        shardshuffle: bool =True,
        handler: Callable = wds.warn_and_continue,
        **kwargs,
    ):
        dataset = IndexedWebDataset(
            url2index=url2index,
            handler=handler,
            resampled=resampled,
            shardshuffle=shardshuffle,
            **kwargs,
        )
        segment_transforms = LyricsSegmentTransforms(
            sample_rate=sample_rate,
            sample_duration=sample_duration,
            audio_keys=audio_keys,
            audio_format=audio_format,
            max_num_segments=max_num_segments,
            shuffle_segments=shuffle_segments,
            url2index=url2index,
            handler=handler
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=segment_transforms,
        )
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
        ]
        super().__init__(dataset, pipeline)

class LyricsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        predict_num_rounds=1,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.predict_num_rounds = predict_num_rounds

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return [DataLoader(self.predict_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)]*self.predict_num_rounds
    
    @classmethod
    def from_dataset_type(
        cls,
        dataset_type: str,
        batch_size: int,
        sample_rate=24000,
        sample_duration=10,
        shuffle_buffer_size: int = 100,
        lyrics_max_seq_len: int = 150,
        num_workers: int = 8,
        pin_memory: bool = True,
        music_types: str = 'mixture,instrumental',
        enable_punctuation: bool = False
    ):
        sample_duration = sample_duration if isinstance(sample_duration, (list, tuple)) else [sample_duration]
        if dataset_type == 'style_audio':
            train_dataset, valid_dataset = DefaultDatasets.Batched.style_audio_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation=enable_punctuation)
        elif dataset_type == 'mcc9m':
            train_dataset, valid_dataset = DefaultDatasets.Batched.mcc9m(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, enable_punctuation=enable_punctuation)
        elif dataset_type == 'style_text':
            train_dataset, valid_dataset = DefaultDatasets.Batched.style_text_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation=enable_punctuation)
        elif dataset_type == 'style_text_2M':
            train_dataset, valid_dataset = DefaultDatasets.Batched.style_text_2M_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types)
        elif dataset_type == 'style_text_300k':
            train_dataset, valid_dataset = DefaultDatasets.Batched.style_text_300k_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types)
        elif dataset_type == 'style_mixed':
            train_dataset, valid_dataset = DefaultDatasets.Batched.style_mixed_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation=enable_punctuation)
        elif dataset_type == 't5_token':
            train_dataset, valid_dataset = DefaultDatasets.Batched.t5_token_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types)
        elif dataset_type == 'wav_only':
            train_dataset, valid_dataset = DefaultDatasets.Batched.wav_only_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, music_types)
        return LyricsDataModule(train_dataset=train_dataset, validation_dataset=valid_dataset, num_workers=num_workers, pin_memory=pin_memory)

def pad_collate_tensor_fn(batch, *, collate_fn_map):
     max_length = max([x.shape[-1] for x in batch])
     batch = [crop_pad_to_seq_length(x, max_length, x.dtype, padding_value=0) for x in batch]
     return collate_tensor_fn(batch, collate_fn_map=collate_fn_map)

def dictionary_collate(batch):
    """Fixes pytorch's default collate which cannot handle dictionaries or null fields."""
    lyrics_collate_fn_map = {
        **default_collate_fn_map,
        torch.Tensor: pad_collate_tensor_fn
    }
    def remove_invalid_fields(item):
        def invalid_field(field): return field is None or isinstance(field, dict)
        return { k:v for k,v in item.items() if not invalid_field(v)}
    batch = [remove_invalid_fields(item) for item in batch]
    return collate(batch, collate_fn_map=lyrics_collate_fn_map)

class LyricsBucketBatcher(BucketBatcher):
    def __init__(
        self,
        *args,
        collate_fn = dictionary_collate,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn

    def __call__(self, it):
        for item in it:
            batch = self.collate_batch(item)
            if batch is not None:
                yield self.collate_fn(batch)

def default_bucket_batcher_fn(sample_rate, sample_duration, batch_size):
    sample_duration = sample_duration if isinstance(sample_duration, (list, tuple)) else [sample_duration]
    buckets_samples = [d*sample_rate for d in sample_duration]
    length_fn = lambda x: x["target_audio"].shape[-1]
    return LyricsBucketBatcher(buckets=buckets_samples, batch_size=batch_size, dynamic_batch=False, length_fn=length_fn)
    
def default_batch_fn(batch_size, collation_fn=dictionary_collate):
    return wds.batched(batch_size, collation_fn=collation_fn)

def transform_dataset(dataset, segment_transforms=(), batch_transforms=(), batch_fn=None, shuffle_buffer_size=None):
    if isinstance(dataset, DataPipeline):
        data_pipeline = dataset
    else:
        data_pipeline = DataPipeline(dataset)
    if shuffle_buffer_size is not None:
        data_pipeline.append(wds.shuffle(shuffle_buffer_size))
    for segment_transform in segment_transforms:
        data_pipeline.append(wds.map(segment_transform))
    if batch_fn is not None:
        data_pipeline.append(batch_fn)
        for batch_transform in batch_transforms:
            data_pipeline.append(wds.map(batch_transform))
    return data_pipeline

def infer_dataset_weights(index_lists):
    weights = []
    for index_list in index_lists:
        if index_list.startswith('hdfs'):
            line_count = len(hdfs_loadtxt(index_list))
        elif Path(index_list).exists():
            with open(index_list, 'r') as f:
                line_count = len(f.read().splitlines())
        else:
            print('Invalid local path. Could not infer dataset weights', index_list)
            return None
        weights.append(line_count)
    return weights


class DefaultDatasets():
    class Basic:
        @staticmethod
        def mcc60m_lossless_dataset(sample_rate, sample_duration, index_list=INDEX["US"]["MCCVocalB"], infer_weights=True):
            datasets = [
                LyricsDataset(
                    url2index=url2index,
                    sample_rate=sample_rate,
                    sample_duration=sample_duration,
                    audio_keys={ 'style_audio': 'audio.npy', 'target_audio': 'audio.npy'},
                    audio_format='npy',
                )
                for url2index in index_list
            ]
            weights = infer_dataset_weights(index_list) if infer_weights else None
            return MultiIterableDataset(
                datasets=datasets, 
                weights=weights,
                seed=2023,
            )
        
        @staticmethod
        def speech_dataset(sample_rate, sample_duration):
            librilight_ds = LibriLightASRDataset(
                url2index=INDEX["US"]["LIBRILIGHT"], 
                sample_rate=sample_rate, 
                min_duration=sample_duration[0], 
                max_duration=sample_duration[-1],
                resampled=True,
                shardshuffle=True,
            )

            libritts_ds = LibriTTSDataset(
                urls=INDEX["US"]["LIBRITTS"],
                sample_rate=sample_rate,
                min_duration=sample_duration[0], 
                max_duration=sample_duration[-1],
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )
            ds = MultiIterableDataset(
                datasets=[librilight_ds, libritts_ds], 
                weights=[50, 1],
                seed=2023,
            )
            return transform_dataset(ds, segment_transforms=[MCCMetadataTextTransform("Speech"), RenameAudioKeyTransform()])
        
        @staticmethod
        def mcc40m_lossless_dataset(sample_rate, sample_duration, index_list=INDEX["US"]["MCCInstrumental"]):
            mcc_instrumental = WrappedMCC40MDataset(
                url2index_list=index_list,
                sample_rate=sample_rate,
                duration=sample_duration[-1],
                audio_key='audio.npy',
                exclude_licenses = [],
                loudness_ratio_threshold=0.2,
                max_vocal_threshold=0.5,
                max_num_crops=10,
                weights=None,
                resampled=True,
                shardshuffle=True,
                use_pipe=False
            )
            return transform_dataset(mcc_instrumental, segment_transforms=[MCCMetadataTextTransform("Instrumental"), RenameAudioKeyTransform()])

        @staticmethod
        def resso_mss_dataset(sample_rate, sample_duration):
            karaoke_ds = LyricsDataset(
                url2index=INDEX["US"]["KARAOKE_TRAIN"],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ 'style_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
                audio_format='mp3',
            )
            resso_mss_ds = LyricsDataset(
                url2index=INDEX["US"]["RESSO_MSS"],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ 'style_audio': 'mss_acc', 'vocal_audio': 'mss_vocal'},
                audio_format='mp3',
            )
            return MultiIterableDataset(
                datasets=[karaoke_ds, resso_mss_ds],
                weights=[0.2, 0.8],
                seed=2023,
            )

        @staticmethod
        def karaoke_vocal_validation_dataset(sample_rate, sample_duration):
            return LyricsDataset(
                url2index=INDEX["US"]["KARAOKE_VALID"],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ 'style_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
                audio_format="mp3",
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                max_num_segments=1,
                shuffle_segments=False,
                use_pipe=False
            )

        @staticmethod
        def karaoke_validation_dataset(sample_rate, sample_duration):
            return LyricsDataset(
                url2index=INDEX["US"]["KARAOKE_VALID"],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ "target_audio": "full.mp3", "style_audio": "full.mp3" },
                audio_format="mp3",
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                max_num_segments=1,
                shuffle_segments=False,
                use_pipe=False
            )
        @staticmethod
        def mcc_validation_dataset(sample_rate, sample_duration):
            return LyricsDataset(
                url2index=INDEX["US"]["MCC60M_VALID_LABEL1"],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ 'style_audio': 'audio.npy', 'target_audio': 'audio.npy'},
                audio_format="npy",
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                max_num_segments=1,
                shuffle_segments=False,
                use_pipe=False
            )
        
    class Batched:
        @staticmethod
        def mcc9m(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, enable_punctuation=False, conditions="style_tag,lyrics_tokens"):
            ds = LyricsDataset(
                url2index=INDEX["US"]["MCC1M_EN_GT"],
                audio_keys={ 'style_audio': 'mp3', 'target_audio': 'mp3'},
                audio_format="mp3",
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                resampled=True,
                shardshuffle=True,
                use_pipe=False,
            )
            ds_batched = transform_dataset(
                dataset=ds,
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation)],
                batch_transforms=[AddConditionsTransform(conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )

            ds_valid = DefaultDatasets.Batched.default_validation_dataset(sample_rate, sample_duration, batch_size, lyrics_max_seq_len, conditions=conditions, enable_punctuation=enable_punctuation)
            return ds_batched, ds_valid
        @staticmethod
        def style_audio_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, style_condition="style_tag", enable_punctuation=False):
            # style_condition = style_audio for audio tower training, style_tag for MIR/Mulan on-the-fly tagging
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds = DefaultDatasets.Basic.mcc60m_lossless_dataset(sample_rate, sample_duration)
                mixture_ds_batched = transform_dataset(
                    dataset=mixture_ds,
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation)],
                    batch_transforms=[AddConditionsTransform(f"{style_condition},lyrics_tokens")],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mcc40m_lossless_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[],
                    batch_transforms=[AddConditionsTransform(style_condition)],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)

            ds_valid = DefaultDatasets.Batched.default_validation_dataset(
                sample_rate, sample_duration, batch_size, lyrics_max_seq_len, 
                conditions=f"{style_condition},lyrics_tokens",
                enable_punctuation=enable_punctuation
            )
            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0], ds_valid
            combined_ds = MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds, ds_valid

        @staticmethod
        def style_text_dataset(
            sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation=False,
            index_list=INDEX["US"]["MCCVocalB"], infer_weights=True
        ):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds = DefaultDatasets.Basic.mcc60m_lossless_dataset(sample_rate, sample_duration, index_list=index_list, infer_weights=infer_weights)
                mixture_ds_batched = transform_dataset(
                    # only mcc60 has metadata attached for converting to style_text
                    dataset=mixture_ds,
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation), MCCMetadataTextTransform("Vocal")],
                    batch_transforms=[AddConditionsTransform("style_text,lyrics_tokens")],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.6)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mcc40m_lossless_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[],
                    batch_transforms=[AddConditionsTransform("style_text")],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.4)

            if 'speech' in music_types:
                speech_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.speech_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation)],
                    batch_transforms=[AddConditionsTransform("style_text,lyrics_tokens")],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(speech_ds_batched)
                weights.append(0.4)

            ds_valid = DefaultDatasets.Batched.default_validation_dataset(
                sample_rate, sample_duration, batch_size, lyrics_max_seq_len, 
                conditions=f"style_text,lyrics_tokens",
                enable_punctuation=enable_punctuation
            )
            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0], ds_valid
            
            return MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            ), ds_valid
        
        @staticmethod
        def style_text_2M_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation=False):
            return DefaultDatasets.Batched.style_text_dataset(
                sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation,
                index_list=INDEX["US"]["MCCVocalB_2M"], infer_weights=False
            )
        @staticmethod
        def style_text_300k_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation=False):
            return DefaultDatasets.Batched.style_text_dataset(
                sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation,
                index_list=INDEX["US"]["MCCVocalB_300k"], infer_weights=False
            )
        
        @staticmethod
        def style_mixed_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types, enable_punctuation=False):
            import random
            class RandomConditionsTransform():
                def __init__(self, conditions=["style_tag,lyrics_tokens", "style_audio,lyrics_tokens"]):
                    self.conditions = conditions

                def __call__(self, item):
                    random_condition = random.choice(self.conditions)
                    return { **item, 'conditions': random_condition }

            mixture_ds = DefaultDatasets.Basic.mcc60m_lossless_dataset(sample_rate, sample_duration)
            mixture_ds_batched = transform_dataset(
                dataset=mixture_ds,
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation), MCCMetadataTextTransform("Vocal")],
                batch_transforms=[RandomConditionsTransform()],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            ds_valid = DefaultDatasets.Batched.default_validation_dataset(
                sample_rate, sample_duration, batch_size, lyrics_max_seq_len, 
                conditions=f"style_tag,lyrics_tokens",
                enable_punctuation=enable_punctuation
            )
            return mixture_ds_batched, ds_valid
        
        @staticmethod
        def t5_token_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds = DefaultDatasets.Basic.mcc60m_lossless_dataset(sample_rate, sample_duration)
                mixture_ds_batched = transform_dataset(
                    # only mcc60 has metadata attached for converting to style_text
                    dataset=mixture_ds,
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), StyleTextT5Transform()],
                    batch_transforms=[AddConditionsTransform("style_tokens,lyrics_tokens"), ],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mcc40m_lossless_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[StyleTextT5Transform()],
                    batch_transforms=[AddConditionsTransform("style_tokens")],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)
            ds_valid = transform_dataset(
                dataset=DefaultDatasets.Basic.mcc_validation_dataset(sample_rate, sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), StyleTextT5Transform()],
                batch_transforms=[AddConditionsTransform("style_tokens,lyrics_tokens")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0], ds_valid
            combined_ds =  MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds, ds_valid

        @staticmethod
        def vocal_conditional_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len):
            multitask_ds_batched = DefaultDatasets.Batched.style_audio_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len)

            vocal_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.resso_mss_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("lyrics_tokens,vocal_audio,vocal_chroma")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            mss_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.resso_mss_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("style_audio,lyrics_tokens,vocal_audio,vocal_chroma")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            combined_ds = MultiIterableDataset(
                [multitask_ds_batched, vocal_ds_batched, mss_ds_batched],
                num_samples=10_000_000, seed=2023,
                weights=[0.7, 0.1, 0.2]
            )
            ds_valid = transform_dataset(
                dataset=DefaultDatasets.Basic.karaoke_vocal_validation_dataset(sample_rate, sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("lyrics_tokens,vocal_audio,vocal_chroma")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
            return combined_ds, ds_valid

        @staticmethod
        def wav_only_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, music_types):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mcc60m_lossless_dataset(sample_rate, sample_duration),
                    segment_transforms=[],
                    batch_transforms=[],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mcc40m_lossless_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[],
                    batch_transforms=[],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)
            ds_valid = transform_dataset(
                dataset=DefaultDatasets.Basic.mcc_validation_dataset(sample_rate, sample_duration),
                segment_transforms=[],
                batch_transforms=[],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0], ds_valid
            combined_ds = MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds, ds_valid

        @staticmethod
        def default_validation_dataset(sample_rate, sample_duration, batch_size, lyrics_max_seq_len, conditions="style_text,lyrics_tokens", enable_punctuation=False):
            return transform_dataset(
                dataset=DefaultDatasets.Basic.mcc_validation_dataset(sample_rate, sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation), MCCMetadataTextTransform("Vocal")],
                batch_transforms=[AddConditionsTransform(conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
