import torch
import random
import webdataset as wds
from functools import partial

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torchaudio_augmentations import Compose
from recipes.bigmusic.datasets.lyrics import default_bucket_batcher_fn
from recipes.bigmusic.datasets.transforms.lyrics import SemanticTokenLengthTransform
from recipes.bigmusic.datasets.mix import DataModule
from recipes.musiclm.datasets.mcc import WrappedMCC40MDataset
from recipes.musiclm.datasets.karaoke import KaraokeDataset
from recipes.musiclm.datamodules.webdataset import wds_to_dict
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.utils.webdataset import return_self


def collate_fn(batch, sample_rate=24000, mixed_ratio=0.0):
    batch["target_audio"] = batch["audio"]
    del batch["audio"]
    batch["duration"] = batch["target_audio"].shape[-1] // sample_rate
    if "text" not in batch or (mixed_ratio > 0 and random.random() < mixed_ratio):
        batch["style_audio"] = batch["target_audio"]
        batch["conditions"] = "style_audio,duration"
    else:
        batch["style_text"] = batch["text"]
        batch["conditions"] = "style_text,duration"
        del batch["text"]
    return batch


class InstrumentalWebDataModule(DataModule):
    def __init__(
        self,
        dataset_name: str = "MCC40M_US",
        val_split: str = "SSTK_EVAL_US",
        sample_rate: int = 24000,
        duration: Union[float, List[float]] = 30.0,
        batch_size: int = 16,
        shuffle_buffer_size: int = 64,
        num_workers: int = 6,
        pin_memory: bool = True,
        collate_fn: Callable = collate_fn,
        min_length_ratio: float = 0.8,
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = True,
        sstk_filtered: Optional[str] = None,
        avoid_sound_effect: bool = True,
        avoid_vocal: bool = True,
        max_vocal_threshold: float = 0.5,
        overlap_vocal_threshold: float = 0.1,
        audio_metrics_filtered: bool = True,
        text_type: Optional[str] = None,
        max_num_crops: Optional[Union[int, List[int]]] = 3,
        crop_step_size: Optional[Union[float, List[float]]] = 10.0,
        additional_transforms: Optional[List] = None,
        keys=["audio", "text", "structure", "intensity"],
        mixed_ratio: float = 0.0,
        max_duration: Optional[int] = None,
        use_pipe: bool = False,
        seed: int = 555,
    ):
        if dataset_name == "MCC40M_US":
            hdfs_dir = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc/indexes_merge"
            train_urls_and_weights = [
                (f"{hdfs_dir}/nonvocal-A-alternative-rock+indie-pop+sertanejo+trap-rap.txt", 0.33),
                (f"{hdfs_dir}/nonvocal-A-blues+childhood+country+devotional+k-pop+soundtrack+trance+world-music.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-A-classical.txt", 1.33),
                (f"{hdfs_dir}/nonvocal-A-easy-listening.txt", 0.33),
                (f"{hdfs_dir}/nonvocal-A-electronic+techno.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-A-folk+indie-folk.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-A-hip-hop-rap.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-A-jazz.txt", 1.33),
                (f"{hdfs_dir}/nonvocal-A-new-age.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-A-pop.txt", 1.33),
                (f"{hdfs_dir}/nonvocal-A-rock.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-B-alternative-rock+indie-pop+sertanejo+trap-rap.txt", 0.33),
                (f"{hdfs_dir}/nonvocal-B-blues+childhood+country+devotional+k-pop+soundtrack+trance+world-music.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-B-classical.txt", 1.33),
                (f"{hdfs_dir}/nonvocal-B-easy-listening.txt", 0.33),
                (f"{hdfs_dir}/nonvocal-B-electronic+techno.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-B-folk+indie-folk.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-B-hip-hop-rap.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-B-jazz.txt", 1.33),
                (f"{hdfs_dir}/nonvocal-B-new-age.txt", 0.66),
                (f"{hdfs_dir}/nonvocal-B-pop.txt", 1.33),
                (f"{hdfs_dir}/nonvocal-B-rock.txt", 0.66),
            ]
        elif dataset_name == "SSTK_EVAL_US":
            train_urls_and_weights = [
                ("hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/shutterstock/all_url2idx_tag.txt", 1.0),
            ]
        elif dataset_name == "SSTK+MCC_EVAL_US":
            train_urls_and_weights = [
                ("hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/shutterstock/all_url2idx_tag_mcc.txt", 1.0),
            ]
        elif dataset_name == "SSTK_US":
            train_urls_and_weights = [(106, 1.0)]
        elif dataset_name == "SSTK_US_RICH_INSTRUMENTS":
            train_urls_and_weights = [
                (124, 1.0), # no instrument label
                (125, 1.0), # 1 instrument
                (126, 2.0), # 2 instruments
                (127, 3.0), # 3+ instruments
            ]
        elif dataset_name == "SSTK_US_GENRE_BALANCED":
            train_urls_and_weights = [(146, 1.0)]
        elif dataset_name == "SSTK_US_SFT":
            train_urls_and_weights = [(155, 1.0)]
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset_name}")

        if isinstance(crop_step_size, (list, tuple)):
            crop_step_size = [int(c * sample_rate) for c in crop_step_size]
        else:
            crop_step_size = int(crop_step_size * sample_rate)

        train_dataset = WrappedMCC40MDataset(
            url2index_list=[x[0] for x in train_urls_and_weights],
            weights=[x[1] for x in train_urls_and_weights],
            sample_rate=sample_rate,
            duration=duration,
            audio_key="audio.npy",
            min_length_ratio=min_length_ratio,
            normalize_audio=normalize_audio,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            aed_filtered=aed_filtered,
            sstk_filtered=sstk_filtered,
            avoid_sound_effect=avoid_sound_effect,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            overlap_vocal_threshold=overlap_vocal_threshold,
            audio_metrics_filtered=audio_metrics_filtered,
            text_type=text_type,
            max_num_crops=max_num_crops,
            crop_step_size=crop_step_size,
            max_duration=max_duration,
            additional_transforms=additional_transforms,
            resampled=True,
            shardshuffle=True,
            use_pipe=use_pipe,
            seed=seed,
        )
        train_dataset = WebPipeline(
            train_dataset,
            pipeline=[{"compose": [
                wds_to_dict(*keys),
                wds.map(SemanticTokenLengthTransform(sample_rate=sample_rate, audio_key="audio")),
                wds.shuffle(shuffle_buffer_size),
                default_bucket_batcher_fn(
                    sample_rate, duration, batch_size, lyrics_frame_rate=0, max_duration=max_duration
                ),
            ]}],
        )

        if val_split == "SSTK_EVAL_US":
            val_urls_and_weights = [
                ("hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/shutterstock/val_url2idx_tag.txt", 1.0),
            ]
        elif val_split == "SSTK_US":
            val_urls_and_weights = [(107, 1.0)]
        else:
            raise NotImplementedError(f"Unknown val split: {val_split}")

        validation_dataset = WrappedMCC40MDataset(
            url2index_list=[x[0] for x in val_urls_and_weights],
            weights=[x[1] for x in val_urls_and_weights],
            sample_rate=sample_rate,
            duration=duration,
            audio_key="audio.npy",
            min_length_ratio=min_length_ratio,
            normalize_audio=normalize_audio,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            aed_filtered=aed_filtered,
            sstk_filtered=sstk_filtered,
            avoid_sound_effect=avoid_sound_effect,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            overlap_vocal_threshold=overlap_vocal_threshold,
            audio_metrics_filtered=audio_metrics_filtered,
            text_type=text_type,
            max_num_crops=max_num_crops,
            crop_step_size=crop_step_size,
            max_duration=max_duration,
            additional_transforms=additional_transforms,
            resampled=False,
            shardshuffle=False,
            use_pipe=use_pipe,
            seed=seed,
            nodesplitter=return_self,
        )
        validation_dataset = WebPipeline(
            validation_dataset,
            pipeline=[{"compose": [
                wds_to_dict(*keys),
                wds.map(SemanticTokenLengthTransform(sample_rate=sample_rate, audio_key="audio")),
                default_bucket_batcher_fn(
                    sample_rate, duration, batch_size, lyrics_frame_rate=0, max_duration=max_duration
                ),
            ]}],
        )

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=partial(
                collate_fn,
                sample_rate=sample_rate,
                mixed_ratio=mixed_ratio,
            ),
            do_shuffle=False,
        )
        