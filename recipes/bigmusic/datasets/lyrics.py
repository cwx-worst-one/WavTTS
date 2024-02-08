from typing import Callable, List, Optional, Any
import torch
from pathlib import Path
import webdataset as wds
import pytorch_lightning as pl
from recipes.bigmusic.datasets.transforms.lyrics import (
    LyricsTokenTransform, 
    MCCMetadataTextTransform, 
    SSTKMetadataTextTransform,
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
from samantha.dataio.batching import BucketBatcher
from recipes.bigmusic.datasets.index_lists import INDEX
from samantha.utils.hdfs_tools import hdfs_open, hdfs_loadtxt
from recipes.datasets.mcc.mix import LibriLightASRDataset, LibriTTSDataset
from samantha.dataio.parquet import ParquetDataset
from functools import partial
from webdataset import filters, shardlists
import torch.distributed as dist
from recipes.bigmusic.datasets.utils.ddp_utils import distributed_subset

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
        nodesplitter: Any = shardlists.single_node_only,
    ):
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
        if isinstance(url2index, int):
            dataset = ParquetDataset(
                data_id=url2index,
                handler=handler,
                resampled=resampled,
                shardshuffle=shardshuffle,
                nodesplitter=nodesplitter
            )
            pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        elif isinstance(url2index, list):
            dataset = ParquetDataset(
                data_urls=url2index,
                handler=handler,
                resampled=resampled,
                shardshuffle=shardshuffle,
                nodesplitter=nodesplitter
            )
            pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        elif isinstance(url2index, str):
            dataset = IndexedWebDataset(
                url2index=url2index,
                handler=handler,
                resampled=resampled,
                shardshuffle=shardshuffle,
                use_pipe=False,
                nodesplitter=nodesplitter
            )
            pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        else:
            raise Exception(f"Unhandled url type for: {url2index}")
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
        if dist.is_available() and torch.distributed.is_initialized() and dist.get_world_size() > 1:
            dataset = distributed_subset(self.predict_dataset)
        else:
            dataset = self.predict_dataset
        return [DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)]*self.predict_num_rounds

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
        valid_dataset_type: str = 'val_mcc60m_groupA',
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
                sample_rate=sample_rate, 
                sample_duration=sample_duration, 
                batch_size=batch_size, 
                shuffle_buffer_size=shuffle_buffer_size, 
                lyrics_max_seq_len=lyrics_max_seq_len, 
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

def simple_collate_fn(batch, *, collate_fn_map):
    return batch

def dictionary_collate(batch, remove_invalid=True):
    """Fixes pytorch's default collate which cannot handle dictionaries or null fields."""
    lyrics_collate_fn_map = {
        **default_collate_fn_map,
        torch.Tensor: pad_collate_tensor_fn,
        list: simple_collate_fn,
        type(None): simple_collate_fn,
    }
    if remove_invalid:
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
    elif 'audio' in item:
        target_seq_length = int(item['audio'].shape[-1] / sample_rate * semantic_frame_rate) # fallback case
    else:
        raise ValueError("Batched item must target audio to determine length function")

    if 'lyrics_tokens_length' in item:
        input_seq_length = item['lyrics_tokens_length']
    elif 'lyrics_tokens' in item:
        input_seq_length = len(item['lyrics_tokens']) # fallback case
    else:
        input_seq_length = 0 # instrumental case
    return target_seq_length + input_seq_length

def default_bucket_batcher_fn(
    sample_rate,
    sample_duration,
    batch_size,
    semantic_frame_rate=25,
    lyrics_frame_rate=14,
    max_duration=None,
):
    token_frame_rate = semantic_frame_rate + lyrics_frame_rate
    sample_duration = sample_duration if isinstance(sample_duration, (list, tuple)) else [sample_duration]
    buckets_samples = [d * token_frame_rate for d in sample_duration]
    length_fn = partial(default_bucket_batcher_length_fn, sample_rate=sample_rate, semantic_frame_rate=semantic_frame_rate)
    if max_duration is None:
        maximum_bucket_size = buckets_samples[-1] * batch_size
    else:
        maximum_bucket_size = max_duration * token_frame_rate * batch_size
    return LyricsBucketBatcher(
        buckets=buckets_samples,
        maximum_bucket_size=maximum_bucket_size,
        dynamic_batch=True,
        length_fn=length_fn,
    )
    
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
        def indexed_webdataset(sample_rate, sample_duration, index_list=INDEX["US"]["MCCVocalB"], infer_weights=True, max_num_segments: Optional[int] = 10, **kwargs):
            datasets = [
                LyricsDataset(
                    url2index=url2index,
                    sample_rate=sample_rate,
                    sample_duration=sample_duration,
                    audio_keys={ 'style_audio': 'audio.npy', 'target_audio': 'audio.npy'},
                    audio_format='npy',
                    max_num_segments=max_num_segments,
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
        def vocal_parquet_dataset(sample_rate, sample_duration, index_list=INDEX["US"]["MCCVocalA_1M_Parquet"], audio_format='npy', **kwargs):
            return LyricsDataset(
                url2index=index_list,
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ 'style_audio': 'wav', 'target_audio': 'wav'},
                audio_format=audio_format,
                **kwargs
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
            )
        @staticmethod
        def indexed_validation_dataset(sample_rate, sample_duration, url2index=INDEX["US"]["MCC60M_VALID_GROUPA"]):
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
            )
        @staticmethod
        def parquet_validation_dataset(sample_rate, sample_duration, url2index=INDEX["US"]["MCC60M_VALID_GROUPA"]):
            return LyricsDataset(
                url2index=url2index,
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                # audio_keys={ 'style_audio': 'wav', 'target_audio': 'wav'},
                audio_keys={ 'style_audio': 'audio.npy', 'target_audio': 'audio.npy'},
                audio_format="npy",
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                max_num_segments=1,
                shuffle_segments=False,
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
            enable_punctuation=True, style_conditions="style_tag,lyrics_tokens", infer_weights=False,
            metadata_tfm_fn=partial(MCCMetadataTextTransform, "Vocal"),
            tokenizer_init_fn=LyricsTokenTransform.init_espeak_tokenizer,
            min_song_confidence=0.8, min_segment_confidence=0.8,
            **kwargs
        ):
            if isinstance(style_conditions, list): # multiple style conditions - for mixed style training. In that case, use random conditioning
                batch_transforms = [RandomConditionsTransform(style_conditions)]
            else:
                batch_transforms = [AddConditionsTransform(style_conditions)]
            ds = DefaultDatasets.Basic.indexed_webdataset(
                sample_rate, sample_duration=sample_duration, index_list=index_list, infer_weights=infer_weights,
                min_song_confidence=min_song_confidence, min_segment_confidence=min_segment_confidence, **kwargs,
            )
            ds_batched = transform_dataset(
                dataset=ds,
                segment_transforms=[SemanticTokenLengthTransform(sample_rate=sample_rate), metadata_tfm_fn(), tokenizer_init_fn(lyrics_max_seq_len, enable_punctuation=enable_punctuation)],
                batch_transforms=batch_transforms,
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            return ds_batched
        
        @staticmethod
        def batched_vocal_parquet_dataset(
            # dataset params
            sample_rate, sample_duration,
            # batching params
            batch_size, shuffle_buffer_size, lyrics_max_seq_len,
            # index path
            index_list,
            # transform params
            enable_punctuation=True, style_conditions="style_tag,lyrics_tokens",
            metadata_tfm_fn=partial(MCCMetadataTextTransform, "Vocal"),
            tokenizer_init_fn=LyricsTokenTransform.init_espeak_tokenizer,
            min_song_confidence=0.8, min_segment_confidence=0.8,
            audio_format='npy',
            
        ):
            if isinstance(style_conditions, list): # multiple style conditions - for mixed style training. In that case, use random conditioning
                batch_transforms = [RandomConditionsTransform(style_conditions)]
            else:
                batch_transforms = [AddConditionsTransform(style_conditions)]
            ds = DefaultDatasets.Basic.vocal_parquet_dataset(
                sample_rate, sample_duration=sample_duration, index_list=index_list,
                min_song_confidence=min_song_confidence, min_segment_confidence=min_segment_confidence,
                audio_format=audio_format
            )
            ds_batched = transform_dataset(
                dataset=ds,
                segment_transforms=[
                    SemanticTokenLengthTransform(), metadata_tfm_fn(),
                    tokenizer_init_fn(lyrics_max_seq_len, enable_punctuation=enable_punctuation), 
                ],
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
                segment_transforms=[RenameAudioKeyTransform(), SemanticTokenLengthTransform(sample_rate=sample_rate), MCCMetadataTextTransform("Instrumental")],
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
                segment_transforms=[RenameAudioKeyTransform(), SemanticTokenLengthTransform(sample_rate=sample_rate), LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len, enable_punctuation=enable_punctuation), MCCMetadataTextTransform("Speech")],
                batch_transforms=[AddConditionsTransform(style_conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            return speech_ds_batched
        
        @staticmethod
        def default_validation_dataset(
            sample_rate, sample_duration, batch_size, lyrics_max_seq_len, url2index, 
            metadata_tfm_fn=partial(MCCMetadataTextTransform, "Vocal"),
            tokenizer_init_fn=LyricsTokenTransform.init_espeak_tokenizer,
            style_conditions="style_text,lyrics_tokens", enable_punctuation=True):
            return transform_dataset(
                dataset=DefaultDatasets.Basic.indexed_validation_dataset(sample_rate, sample_duration, url2index=url2index),
                segment_transforms=[
                    tokenizer_init_fn(lyrics_max_seq_len, enable_punctuation=enable_punctuation), 
                    SemanticTokenLengthTransform(sample_rate=sample_rate), 
                    metadata_tfm_fn()],
                batch_transforms=[AddConditionsTransform(style_conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
        
        @staticmethod
        def default_validation_parquet_dataset(
            sample_rate, sample_duration, batch_size, lyrics_max_seq_len, url2index,
            metadata_tfm_fn=partial(MCCMetadataTextTransform, "Vocal"),
            tokenizer_init_fn=LyricsTokenTransform.init_espeak_tokenizer,
            style_conditions="style_text,lyrics_tokens", enable_punctuation=True,
        ):
            return transform_dataset(
                dataset=DefaultDatasets.Basic.parquet_validation_dataset(sample_rate, sample_duration, url2index=url2index),
                segment_transforms=[tokenizer_init_fn(lyrics_max_seq_len, enable_punctuation=enable_punctuation), SemanticTokenLengthTransform(), metadata_tfm_fn()],
                batch_transforms=[AddConditionsTransform(style_conditions)],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
        
        # Coarse/Diffusion model training
        @staticmethod
        def unfiltered_batched_vocal_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, index_list, min_song_confidence=0.0, min_segment_confidence=0.0):
            vocal_dataset = DefaultDatasets.Basic.indexed_webdataset(
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
        def unfiltered_batched_validation_dataset(sample_rate, sample_duration, batch_size, lyrics_max_seq_len, url2index):
            return transform_dataset(
                dataset=DefaultDatasets.Basic.indexed_validation_dataset(sample_rate, sample_duration, url2index=url2index),
                segment_transforms=[],
                batch_transforms=[],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
        
        @staticmethod
        def mcc9m(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, enable_punctuation=True, style_conditions="style_tag,lyrics_tokens"):
            index_list = INDEX["US"]["MCC1M_EN_GT"]
            return DefaultDatasets.Batched.default_batched_vocal_dataset(
                sample_rate, sample_duration, batch_size, shuffle_buffer_size, 
                index_list, lyrics_max_seq_len, enable_punctuation=enable_punctuation, style_conditions=style_conditions
            )
        
DATASET_CONFIGS = {
    "mcc60m_vocal_style_text": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalA"] + INDEX["US"]["MCCVocalB"],
            "style_conditions": "style_text,lyrics_tokens",
            "infer_weights": True
        }
    },
    "mcc60m_vocalB_style_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": True
        }
    },
    "mcc60m_vocalB_style_mixed_tag_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB"],
            "style_conditions": ["style_tag,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": True
        }
    },
    "mcc60m_vocalB_style_mixed_text_audio_2m": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": True,
            "min_song_confidence": 0.8,
            "min_segment_confidence": 0.75 # lowering segment confidence, for longer segments
        }
    },
    "mcc60m_vocalA_style_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalA"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": True
        }
    },
    "mcc60m_vocalA_parquet_style_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalA_1M_Parquet"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "min_song_confidence": 0.8,
            "min_segment_confidence": 0.75 # lowering segment confidence, for longer segments
        }
    },
    "cd_baby": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["CD_Baby"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "min_song_confidence": 0.7, # already filtered
            "min_segment_confidence": 0.75 # lowering segment confidence, for longer segments
        }
    },
    "cd_baby_cn": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["CN"]["CD_Baby"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "min_song_confidence": 0.7, # already filtered
            "min_segment_confidence": 0.75 # lowering segment confidence, for longer segments
        }
    },
    "cd_baby_cn_authorized": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["CN"]["CD_Baby_Authorized"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "min_song_confidence": 0.7, # already filtered
            "min_segment_confidence": 0.75 # lowering segment confidence, for longer segments
        }
    },
    "sstk_vocal": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["SSTK_Vocal"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "min_song_confidence": 0.75,
            "min_segment_confidence": 0.75, # lowering segment confidence, for longer segments
            "audio_format": "wav",
            "metadata_tfm_fn": SSTKMetadataTextTransform,
        }
    },
    "mcc60m_vocalA_style_mixed_text_audio_2m": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalA"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": True,
            "min_song_confidence": 0.8,
            "min_segment_confidence": 0.5 # lowering segment confidence, for longer segments
        }
    },
    "mixed_groupa_tt_pop": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalA_TT_POP"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "enable_punctuation": True,
            "min_song_confidence": 0.8,
            "min_segment_confidence": 0.1 # lowering segment confidence, for longer segments
        }
    },
    "mcc60m_2M_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_2M"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_2M_vocal_mixed_text_audio_CN": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["CN"]["MCCVocalB_2M"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
        }
    },
    "mcc60m_1M_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_1M"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_500k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_500k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_300k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_300k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_200k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_200k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_150k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_150k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_100k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_100k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_50k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_50k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_25k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_25k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
        }
    },
    "mcc60m_10k_vocal_mixed_text_audio": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["US"]["MCCVocalB_10k"],
            "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "infer_weights": False # already balanced
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
    # CN Mixed training
    "mixed_qq_zh": {
        "init_fn": DefaultDatasets.Batched.default_batched_vocal_dataset,
        "extra_args": {
            "index_list": INDEX["CN"]["Mixed_QQ_ZH"],
            # "style_conditions": ["style_tag,lyrics_tokens","style_audio,lyrics_tokens"],
            "style_conditions": ["style_tag,lyrics_tokens"],
            "enable_punctuation": True,
            "metadata_tfm_fn": partial(MCCMetadataTextTransform, "Category"),
            "tokenizer_init_fn": LyricsTokenTransform.init_sami_offline_tokenizer,
        }
    },
    # CN Mixed training
    "mixed_lowrisk_zh": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["CN"]["Mixed_LowRisk_ZH"],
            # "style_conditions": ["style_tag,lyrics_tokens","style_audio,lyrics_tokens"],
            "style_conditions": ["style_tag,lyrics_tokens"],
            "enable_punctuation": True,
            "metadata_tfm_fn": partial(MCCMetadataTextTransform, "Category"),
            "tokenizer_init_fn": LyricsTokenTransform.init_sami_offline_tokenizer,
            "min_song_confidence": 0.1,
            "min_segment_confidence": 0.1
        }
    },
    "mixed_mcc_groupa_en": {
        "init_fn": DefaultDatasets.Batched.batched_vocal_parquet_dataset,
        "extra_args": {
            "index_list": INDEX["CN"]["Mixed_MCC_GroupA_EN"],
            # "style_conditions": ["style_text,lyrics_tokens","style_audio,lyrics_tokens"],
            "style_conditions": ["style_text,lyrics_tokens"],
            "enable_punctuation": True,
            "metadata_tfm_fn": partial(MCCMetadataTextTransform, "Category"),
            "tokenizer_init_fn": LyricsTokenTransform.init_sami_offline_tokenizer,
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
            "index_list": INDEX["US"]["MCCVocalB"],
            "min_song_confidence": 0.0,
            "min_segment_confidence": 0.1
        }
    },
    "val_vocal_unfiltered": {
        "init_fn": DefaultDatasets.Batched.unfiltered_batched_validation_dataset,
        "extra_args": {
            "url2index": INDEX["US"]["MCC60M_VALID_GROUPA"],
        }
    },
    # Validation datasets
    "val_mcc60m_groupA": {
        "init_fn": DefaultDatasets.Batched.default_validation_dataset,
        "extra_args": {
            "url2index": INDEX["US"]["MCC60M_VALID_GROUPA"],
            "style_conditions": "style_text,lyrics_tokens",
        }
    },
    "val_mcc60m_groupA_cn": {
        "init_fn": DefaultDatasets.Batched.default_validation_dataset,
        "extra_args": {
            "url2index": INDEX["CN"]["MCC60M_VALID_GROUPA"],
            "style_conditions": "style_text,lyrics_tokens",
        }
    },
    "val_cn_soda": {
        "init_fn": DefaultDatasets.Batched.default_validation_dataset,
        "extra_args": {
            "url2index": INDEX["CN"]["SodaTest"],
            "style_conditions": "style_tag,lyrics_tokens",
        }
    },
}
