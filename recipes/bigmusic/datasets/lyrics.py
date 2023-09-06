from typing import Callable, List, Optional
import torch
import webdataset as wds
import pytorch_lightning as pl
from recipes.bigmusic.datasets.transforms.lyrics import (
    LyricsTokenTransform, 
    VocalChromaTransform, 
    MCCMetadataTextTransform, 
    MetadataT5Transform, 
    AddConditionsTransform,
    MCCInstrumentalBatchTransform,
    AddMulanVocalTagTransform
)
from recipes.bigmusic.datasets.transforms.lyrics_segment import LyricsSegmentTransforms
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from webdataset.pipeline import DataPipeline
from torch.utils.data import DataLoader
from samantha.dataio.dataset import MultiIterableDataset
from recipes.musiclm.datamodules.webdataset import return_self
from torch.utils.data import default_collate
from recipes.musiclm.datasets.mcc import WrappedMCC40MDataset

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
        handler: Callable = wds.warn_and_continue,
        **kwargs,
    ):
        dataset = IndexedWebDataset(
            url2index=url2index,
            handler=handler,
            **kwargs,
        )
        audio_transforms = LyricsSegmentTransforms(
            sample_rate=sample_rate,
            sample_duration=sample_duration,
            audio_keys=audio_keys,
            audio_format=audio_format,
            max_num_segments=max_num_segments,
            shuffle_segments=shuffle_segments,
            url2index=url2index,
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
        ]
        super().__init__(dataset, pipeline)

class WrappedLyricsDataset(MultiIterableDataset):
    def __init__(
        self,
        dataset_list: list,
        sample_rate: int,
        sample_duration: float,
        handler: Callable = wds.warn_and_continue,
        num_samples: int = 10_000_000,
        weights: List[int] = None,
        seed: int = 2023,
        **kwargs,
    ):
        datasets = []
        for item in dataset_list:
            dataset = LyricsDataset(
                url2index=item['url2index'],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys=item["audio_keys"],
                audio_format=item.get("audio_format", "mp3"),
                handler=handler,
                **kwargs,
            )
            datasets.append(dataset)
        if weights is None:
            weights = [1.0 for _ in range(len(datasets))]
        super().__init__(
            datasets=datasets,
            num_samples=num_samples,
            weights=weights,
            seed=seed,
        )

class LyricsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
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
    ):
        if dataset_type == 'style_audio':
            train_dataset = DefaultDatasets.Batched.style_audio_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types)
        elif dataset_type == 'style_text':
            train_dataset = DefaultDatasets.Batched.style_text_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types)
        elif dataset_type == 't5_token':
            train_dataset = DefaultDatasets.Batched.t5_token_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types)
        if dataset_type == 'wav_only':
            train_dataset = DefaultDatasets.Batched.wav_only_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, music_types)
        # TODO: (AS) change validation dataset based on dataset_type
        validation_dataset = DefaultDatasets.Batched.default_validation_dataset(sample_rate, sample_duration, batch_size, lyrics_max_seq_len)
        return LyricsDataModule(train_dataset=train_dataset, validation_dataset=validation_dataset, num_workers=num_workers, pin_memory=pin_memory)


class SingsongDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    @classmethod
    def from_dataset_type(
        cls,
        dataset_type: str,
        batch_size: int,
        sample_rate: int = 24000,
        sample_duration: int = 10,
        shuffle_buffer_size: int = 100,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        if dataset_type == 'resso':            
            train_dataset = DefaultDatasets.Batched.singsong_dataset(sample_rate, sample_duration, "train", batch_size, shuffle_buffer_size)
            validation_dataset = DefaultDatasets.Batched.singsong_dataset(sample_rate, sample_duration, "val", batch_size, shuffle_buffer_size)
        else:
            raise NotImplementedError
        return SingsongDataModule(train_dataset=train_dataset, validation_dataset=validation_dataset, num_workers=num_workers, pin_memory=pin_memory)


def dictonary_collate(batch):
    """Fixes pytorch's default collate which cannot handle dictionaries or null fields."""
    def remove_invalid_fields(item):
        def invalid_field(field): return field is None or isinstance(field, dict)
        return { k:v for k,v in item.items() if not invalid_field(v)}
    return default_collate([remove_invalid_fields(item) for item in batch])

def transform_dataset(dataset, segment_transforms=(), batch_transforms=(), batch_size=None, collation_fn=dictonary_collate, shuffle_buffer_size=None):
    data_pipeline = DataPipeline(dataset)
    if shuffle_buffer_size is not None:
        data_pipeline.append(wds.shuffle(shuffle_buffer_size))
    for segment_transform in segment_transforms:
        data_pipeline.append(wds.map(segment_transform))
    if batch_size is not None:
        data_pipeline.append(wds.batched(batch_size, collation_fn=collation_fn))
    for batch_transform in batch_transforms:
        data_pipeline.append(wds.map(batch_transform))
    return data_pipeline


class DefaultDatasets():
    class Basic:
        @staticmethod
        def mixture_dataset(sample_rate, sample_duration):
            return WrappedLyricsDataset(
                [
                    {
                        "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_train.tar_to_index.tsv", 
                        "audio_keys": { 'style_audio': 'full.mp3', 'target_audio': 'full.mp3'},
                    },
                    {
                        "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso.tar_to_index.tsv", 
                        "audio_format": "m4a", # resso is m4a format for some reason
                        "audio_keys": { 'style_audio': 'mp3', 'target_audio': 'mp3'},
                    },
                    {
                        "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc9m.tar_to_index.tsv", 
                        "audio_keys": { 'style_audio': 'mp3', 'target_audio': 'mp3'},
                    },
                    {
                        "url2index": "/mnt/bn/audio-diffusion/data/vocal_mcc/mcc_60m_url2index_final.tsv", 
                        "audio_keys": { 'style_audio': 'mp3', 'target_audio': 'mp3'},
                    },
                ],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                resampled=True,
                shardshuffle=True,
                use_pipe=True,
                weights=[0.05, 0.25, 1, 20] # 50k, 250k, 1m, 35m
            )
        @staticmethod
        def mcc60m_mixture_dataset(sample_rate, sample_duration):
            return LyricsDataset(
                url2index='/mnt/bn/audio-diffusion/data/vocal_mcc/mcc_60m_url2index_final.tsv',
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ 'target_audio': 'mp3', 'style_audio': 'mp3' },
                resampled=True,
                shardshuffle=True,
                # use_pipe=True,
            )
        @staticmethod
        def resso_mss_dataset(sample_rate, sample_duration, split="train"):
            if split == "train":
                return WrappedLyricsDataset(
                    [
                        {
                            "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_train.tar_to_index.tsv", 
                            "audio_keys": { 'style_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
                        },
                        {
                            "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso_mss.tar_to_index.tsv", 
                            "audio_keys": { 'style_audio': 'mss_acc', 'vocal_audio': 'mss_vocal'},
                        },
                    ],
                    sample_rate=sample_rate,
                    sample_duration=sample_duration,
                    resampled=True,
                    shardshuffle=True,
                    use_pipe=True,
                    weights=[0.2, 0.8]
                )
            elif split == "val":
                return WrappedLyricsDataset(
                    [
                        {
                            "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_valid.tar_to_index.tsv", 
                            "audio_keys": { 'style_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
                        }
                    ],
                    sample_rate=sample_rate,
                    sample_duration=sample_duration,
                    resampled=True,
                    shardshuffle=True,
                    use_pipe=True,
                    weights=[1]
                )
                
        @staticmethod
        def vocal_only_dataset(sample_rate, sample_duration):
            return WrappedLyricsDataset(
                [
                    {
                        "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_train.tar_to_index.tsv", 
                        "audio_keys": { 'target_audio': 'vocal.mp3', 'vocal_audio': 'vocal.mp3' },
                    },
                    {
                        "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso_mss.tar_to_index.tsv", 
                        "audio_keys": { 'target_audio': 'mss_vocal', 'vocal_audio': 'mss_vocal'},
                    },
                ],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                resampled=True,
                shardshuffle=True,
                weights=[0.1, 0.9]
            )
        @staticmethod
        def mcc40m_instrumental_dataset(sample_rate, sample_duration):
            return WrappedMCC40MDataset(
                [
                    "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/a.tsv",
                    "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/b.tsv",
                    "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/c.tsv",
                    "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/d.tsv",
                    "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/e.tsv",
                    "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/f.tsv",
                ],
                # [
                #     "/mnt/bn/audio-diffusion/data/genre_balanced_mcc_celer/a.tsv",
                #     "/mnt/bn/audio-diffusion/data/genre_balanced_mcc_celer/b.tsv",
                #     "/mnt/bn/audio-diffusion/data/genre_balanced_mcc_celer/c.tsv",
                #     "/mnt/bn/audio-diffusion/data/genre_balanced_mcc_celer/d.tsv",
                #     "/mnt/bn/audio-diffusion/data/genre_balanced_mcc_celer/e.tsv",
                #     "/mnt/bn/audio-diffusion/data/genre_balanced_mcc_celer/f.tsv",
                # ],
                sample_rate=sample_rate,
                duration=sample_duration,
                audio_key='mp3',
                exclude_licenses = [],
                loudness_ratio_threshold=0.2,
                max_num_crops=3,
                resampled=True,
                shardshuffle=True,
                use_pipe=True
            )
        @staticmethod
        def mc40m_filtered_instrumental_dataset(sample_rate, sample_duration):
            url2index = [
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.alternative-hip-hop+chinese-style+others",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.blues+childhood+country+devotional+k-pop+soundtrack+trance+world-music",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.classical",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.easy-listening",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.electronic+techno",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.folk+indie-folk",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.hip-hop-rap",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.jazz",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.new-age",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.pop",
                "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores+vad/genre_specific/npy_url2idx.txt.rock"
            ]
            weights = [1.0, 2.0, 4.0, 1.0, 2.0, 2.0, 2.0, 4.0, 2.0, 4.0, 2.0]
            return WrappedMCC40MDataset(
                url2index,
                sample_rate=sample_rate,
                duration=sample_duration,
                audio_key='audio.npy',
                exclude_licenses = [],
                loudness_ratio_threshold=0.2,
                max_vocal_threshold=0.5,
                max_num_crops=10,
                weights=weights,
                resampled=True,
                shardshuffle=True,
                use_pipe=True
            )
        
        @staticmethod
        def karaoke_validation_dataset(sample_rate, sample_duration):
            import os
            if os.path.exists('/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0131.tar'):
                url2index = {
                    '/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0131.tar': 
                    '/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0131.tar.index'
                }
            else:
                # Web:
                url2index = '/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/karaoke_valid.tar_to_index.tsv'

            return LyricsDataset(
                url2index,
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                audio_keys={ "target_audio": "full.mp3", "style_audio": "full.mp3" },
                audio_format="mp3",
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                max_num_segments=1,
                shuffle_segments=False,
                use_pipe=True
            )
        
    class Batched:
        @staticmethod
        def style_audio_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mixture_dataset(sample_rate, sample_duration),
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len)],
                    batch_transforms=[AddConditionsTransform("style_audio,lyrics_tokens")],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mc40m_filtered_instrumental_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[],
                    batch_transforms=[MCCInstrumentalBatchTransform(), AddConditionsTransform("style_audio")],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)

            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0]
            combined_ds = MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds

        @staticmethod
        def style_text_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds_batched = transform_dataset(
                    # only mcc60 has metadata attached for converting to style_text
                    dataset=DefaultDatasets.Basic.mcc60m_mixture_dataset(sample_rate, sample_duration),
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), MCCMetadataTextTransform(), AddMulanVocalTagTransform()],
                    batch_transforms=[AddConditionsTransform("style_text,lyrics_tokens")],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mc40m_filtered_instrumental_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[MCCMetadataTextTransform()],
                    batch_transforms=[MCCInstrumentalBatchTransform(), AddConditionsTransform("style_text")],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)

            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0]
            return MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds

        @staticmethod
        def t5_token_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds_batched = transform_dataset(
                    # only mcc60 has metadata attached for converting to style_text
                    dataset=DefaultDatasets.Basic.mcc60m_mixture_dataset(sample_rate, sample_duration),
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), MetadataT5Transform()],
                    batch_transforms=[AddConditionsTransform("style_tokens,lyrics_tokens"), ],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mc40m_filtered_instrumental_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[MetadataT5Transform()],
                    batch_transforms=[AddConditionsTransform("style_tokens"), MCCInstrumentalBatchTransform()],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)
            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0]
            combined_ds =  MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds

        @staticmethod
        def vocal_conditional_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types):
            multitask_ds_batched = DefaultDatasets.Batched.style_audio_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len)

            vocal_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.vocal_only_dataset(sample_rate, sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("lyrics_tokens,vocal_audio,vocal_chroma")],
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size
            )
            mss_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.resso_mss_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("style_audio,lyrics_tokens,vocal_audio")],
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size
            )
            combined_ds = MultiIterableDataset(
                [multitask_ds_batched, vocal_ds_batched, mss_ds_batched],
                num_samples=10_000_000, seed=2023,
                weights=[0.7, 0.1, 0.2]
            )
            return combined_ds

        @staticmethod
        def wav_only_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, music_types):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mixture_dataset(sample_rate, sample_duration),
                    segment_transforms=[],
                    batch_transforms=[],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mc40m_filtered_instrumental_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[],
                    batch_transforms=[MCCInstrumentalBatchTransform()],
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)
            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0]
            combined_ds = MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds

        @staticmethod
        def singsong_dataset(sample_rate, sample_duration, split, batch_size, shuffle_buffer_size):            
            mss_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.resso_mss_dataset(sample_rate=sample_rate, sample_duration=sample_duration, split=split),
                segment_transforms=[],
                batch_transforms=[AddConditionsTransform("style_audio,vocal_audio")],
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size
            )
            return mss_ds_batched
        
        @staticmethod
        def default_validation_dataset(sample_rate, sample_duration, batch_size, lyrics_max_seq_len, conditions="style_audio,lyrics_tokens"):
            return transform_dataset(
                dataset=DefaultDatasets.Basic.karaoke_validation_dataset(sample_rate, sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len)],
                batch_transforms=[AddConditionsTransform(conditions)],
                batch_size=batch_size,
                shuffle_buffer_size=None
            )
