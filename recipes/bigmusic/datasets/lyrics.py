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
    RandomConditionsTransform,
    AddConditionsTransform,
    RenameAudioKeyTransform,
    SemanticTokenLengthTransform
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
from functools import partial
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
        min_song_confidence: int=0.8,
        min_segment_confidence: int=0.8,
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
            min_song_confidence=min_song_confidence,
            min_segment_confidence=min_segment_confidence,
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
        sample_rate=24000,
        sample_duration: list = [10],
        batch_size: int = 16,
        shuffle_buffer_size: int = 0,
        lyrics_max_seq_len: int = 400,
        num_workers: int = 8,
        pin_memory: bool = True,
        dataset_types: str = 'mcc60m_vocal_style_text',
        dataset_weights = None,
        valid_dataset_type: str = 'mcc60m_groupA',
    ):
        # convert to list of durations
        sample_duration = sample_duration if isinstance(sample_duration, (list, tuple)) else [sample_duration]
        dataset_types = dataset_types if isinstance(dataset_types, (list, tuple)) else dataset_types.split(',')
        if dataset_weights is not None:
            assert len(dataset_types) == len(dataset_weights), "Number of datasets must match weights"
            
        # initialize datasets
        datasets = []
        for dataset_type in dataset_types:
            dataset_config = DATASET_CONFIGS[dataset_type]
            init_fn = dataset_config['init_fn']
            dataset = init_fn(
                sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, 
                **dataset_config['extra_args']
            )
            datasets.append(dataset)

        if len(datasets) > 1:
            train_dataset = MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=dataset_weights
            )
        else:
            train_dataset = datasets[0]
        
        # initialize validation dataset
        valid_dataset_config = DATASET_CONFIGS[valid_dataset_type]
        valid_dataset = valid_dataset_config['init_fn'](
            sample_rate, sample_duration, batch_size, lyrics_max_seq_len,
            **valid_dataset_config['extra_args']
        )
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
            if batch is not None and len(batch):
                yield self.collate_fn(batch)

def default_bucket_batcher_length_fn(item, sample_rate=24000, semantic_frame_rate=25):
    if 'target_tokens_length' in item:
        target_seq_length = item['target_tokens_length'] # value returned from SemanticTokenLengthTransform
    elif 'target_audio' in item:
        target_seq_length = int(item['target_audio'].shape[-1] / sample_rate * semantic_frame_rate) # fallback case
    else:
        raise ValueError("Batched item must target audio to determine length function")

    if 'lyrics_tokens_length' in item:
        input_seq_length = item['lyrics_tokens_length']
    elif 'lyrics_tokens' in item:
        input_seq_length = len(item['lyrics_tokens']) # fallback case
    else:
        input_seq_length = 0 # instrumental case
    return target_seq_length + input_seq_length

def default_bucket_batcher_fn(sample_rate, sample_duration, batch_size, semantic_frame_rate=25, lyrics_frame_rate=14):
    sample_duration = sample_duration if isinstance(sample_duration, (list, tuple)) else [sample_duration]
    buckets_samples = [d * semantic_frame_rate + d * lyrics_frame_rate for d in sample_duration]
    length_fn = partial(default_bucket_batcher_length_fn, sample_rate=sample_rate, semantic_frame_rate=semantic_frame_rate)
    return LyricsBucketBatcher(buckets=buckets_samples, maximum_bucket_size=buckets_samples[-1] * batch_size, dynamic_batch=True, length_fn=length_fn)
    
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
        def mcc60m_lossless_dataset(sample_rate, sample_duration, index_list=INDEX["US"]["MCCVocal"], infer_weights=True, **kwargs):
            datasets = [
                LyricsDataset(
                    url2index=url2index,
                    sample_rate=sample_rate,
                    sample_duration=sample_duration,
                    audio_keys={ 'style_audio': 'audio.npy', 'target_audio': 'audio.npy'},
                    audio_format='npy',
                    **kwargs
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
            if min(sample_duration) > 10:
                sample_duration = [10] + sample_duration
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
            return ds
        
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
            return mcc_instrumental

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
        def mcc_validation_dataset(sample_rate, sample_duration, url2index=INDEX["US"]["MCC60M_VALID_LABEL1"]):
            return LyricsDataset(
                url2index=url2index,
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
        def default_batched_vocal_dataset(
            # dataset params
            sample_rate, sample_duration,
            # batching params
            batch_size, shuffle_buffer_size, lyrics_max_seq_len,
            # index path
            index_list,
            # transform params
            enable_punctuation=False, style_conditions="style_tag,lyrics_tokens", infer_weights=False
        ):
            if isinstance(style_conditions, list): # multiple style conditions - for mixed style training. In that case, use random conditioning
                batch_transforms = [RandomConditionsTransform(style_conditions)]
            else:
                batch_transforms = [AddConditionsTransform(style_conditions)]
            ds = DefaultDatasets.Basic.mcc60m_lossless_dataset(sample_rate, sample_duration=sample_duration, index_list=index_list, infer_weights=infer_weights)
            ds_batched = transform_dataset(
                dataset=ds,
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation), SemanticTokenLengthTransform(), MCCMetadataTextTransform("Vocal")],
                batch_transforms=batch_transforms,
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            return ds_batched
        
        @staticmethod
        def default_batched_instrumental_dataset(
            # dataset params
            sample_rate, sample_duration,
            # batching params
            batch_size, shuffle_buffer_size, lyrics_max_seq_len,
            # index path
            index_list,
            # transform params
            style_conditions="style_tag"
        ):
            instrumental_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.mcc40m_lossless_dataset(sample_rate, sample_duration, index_list),
                segment_transforms=[RenameAudioKeyTransform(), SemanticTokenLengthTransform(), MCCMetadataTextTransform("Instrumental")],
                batch_transforms=[AddConditionsTransform(style_conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            return instrumental_ds_batched

        @staticmethod
        def default_batched_speech_dataset(
            # dataset params
            sample_rate, sample_duration,
            # batching params
            batch_size, shuffle_buffer_size, lyrics_max_seq_len,
            # transform params
            enable_punctuation=False, style_conditions="style_tag,lyrics_tokens"
        ):
            speech_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.speech_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                segment_transforms=[RenameAudioKeyTransform(), SemanticTokenLengthTransform(), LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation), MCCMetadataTextTransform("Speech")],
                batch_transforms=[AddConditionsTransform(style_conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            return speech_ds_batched
        
        @staticmethod
        def default_validation_dataset(sample_rate, sample_duration, batch_size, lyrics_max_seq_len, url2index, style_conditions="style_text,lyrics_tokens", enable_punctuation=False):
            return transform_dataset(
                dataset=DefaultDatasets.Basic.mcc_validation_dataset(sample_rate, sample_duration, url2index=url2index),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation), SemanticTokenLengthTransform(), MCCMetadataTextTransform("Vocal")],
                batch_transforms=[AddConditionsTransform(style_conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
        
        # Coarse/Diffusion model training
        @staticmethod
        def unfiltered_batched_vocal_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, index_list, min_song_confidence=0.0, min_segment_confidence=0.0):
            vocal_dataset = DefaultDatasets.Basic.mcc60m_lossless_dataset(
                sample_rate, sample_duration, index_list,
                min_segment_confidence=min_segment_confidence,
                min_song_confidence=min_song_confidence
            )
            return transform_dataset(
                dataset=vocal_dataset,
                segment_transforms=[],
                batch_transforms=[],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
        
        @staticmethod
        def unfiltered_batched_instrumental_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, index_list):
            return transform_dataset(
                dataset=DefaultDatasets.Basic.mcc40m_lossless_dataset(sample_rate, sample_duration, index_list),
                segment_transforms=[RenameAudioKeyTransform()],
                batch_transforms=[],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )

        @staticmethod
        def mcc9m(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, enable_punctuation=False, style_conditions="style_tag,lyrics_tokens"):
            index_list = INDEX["US"]["MCC1M_EN_GT"]
            return DefaultDatasets.Batched.default_batched_vocal_dataset(
                sample_rate, sample_duration, batch_size, shuffle_buffer_size, 
                index_list, lyrics_max_seq_len, enable_punctuation=enable_punctuation, style_conditions=style_conditions
            )
        
DATASET_CONFIGS = {
    "mcc60m_vocal_style_text": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocal"],
            "style_conditions": "style_text,lyrics_tokens",
            "enable_punctuation": True,
            "infer_weights": True
        }
    },
    "mcc60m_vocal_style_tag": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocal"],
            "style_conditions": "style_tag,lyrics_tokens",
            "enable_punctuation": True,
            "infer_weights": True
        }
    },
    "mcc60m_vocalB_style_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": True
        }
    },
    "mcc60m_2M_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_2M"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_1M_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_1M"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_500k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_500k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_300k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_300k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_300k_vocal_style_text": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_300k"],
            "style_conditions": "style_text,lyrics_tokens",
            "enable_punctuation": True,
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_vocal_style_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocal"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": True
        }
    },
    "mcc60m_vocal_style_mixed_tag_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocal"],
            "style_conditions": ["style_tag,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": True
        }
    },
    "mcc60m_vocal_style_mixed_text_tag_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocal"],
            "style_conditions": ["style_text,lyrics_tokens","style_tag,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "infer_weights": True
        }
    },
    "mcc40m_instrumental_style_text": {
        "init_fn": DefaultDatasets.Batched.default_batched_instrumental_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCInstrumental"],
            "style_conditions": "style_text",
        }
    },
    "mcc40m_instrumental_style_tag": {
        "init_fn": DefaultDatasets.Batched.default_batched_instrumental_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCInstrumental"],
            "style_conditions": "style_tag",
        }
    },
    "speech": {
        "init_fn": DefaultDatasets.Batched.default_batched_speech_dataset,
        "extra_args": {
            "enable_punctuation": False, # disable newline tokenization since we don't group utterances
            "style_conditions": "style_text,lyrics_tokens", # does not support style_tag
        }
    },
    # Unfiltered datasets - for coarse/diffusion training
    "mcc40m_instrumental_unfiltered": {
        "init_fn": DefaultDatasets.Batched.unfiltered_batched_instrumental_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCInstrumental"],
        }
    },
    "mcc60m_vocal_unfiltered": {
        "init_fn": DefaultDatasets.Batched.unfiltered_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocal"],
            "min_song_confidence": 0.0,
            "min_segment_confidence": 0.0
        }
    },
    # Validation datasets
    "mcc60m_groupA": {
        "init_fn": DefaultDatasets.Batched.default_validation_dataset,
        "extra_args": {
            "url2index": INDEX["US"]["MCC60M_VALID_GROUPA"],
            "style_conditions": "style_text,lyrics_tokens",
            "enable_punctuation": True
        }
    },
}
