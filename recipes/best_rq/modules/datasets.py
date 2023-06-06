from typing import Callable, List, Optional
from functools import partial
import torch
import webdataset as wds
from webdataset.pipeline import DataPipeline
from recipes.musiclm.transforms.musiclm import MCCTransforms, MusicLMTransforms
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.dataio.dataset import MultiIterableDataset

from core.dataset.preprocess.fbank import SpeedKaldiFbank
from core.dataset.preprocess.draw_batch import FbankCollate


def fbank_norm(item, mean, std):
    item["fbank"] = (item["fbank"] - mean) / std
    return item


def collation_fn(batch, fbank_collate):
    res = {}
    fbank_collate(batch, res)
    return res


class MCC40MDataset(WebPipeline):

    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        duration: float,
        shuffle_buffer_size: int,
        fbank_mean_path: str,
        fbank_std_path: str,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.5,
        aed_filtered: bool = True,
        avoid_sound_effect: bool = True,
        avoid_vocal: bool = True,
        max_vocal_threshold: float = 0.25,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: Optional[int] = 3,   # recommended for 30s crops
        handler: Callable = wds.warn_and_continue,
        **kwargs,
    ):
        dataset = IndexedWebDataset(
            url2index=url2index,
            handler=handler,
            **kwargs,
        )
        fbank_mean = torch.load(fbank_mean_path).numpy()
        fbank_std = torch.load(fbank_std_path).numpy()

        audio_transforms = MCCTransforms(
            n_samples=int(duration * sample_rate),
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            aed_filtered=aed_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            max_num_crops=max_num_crops,
            crop_step_size=int(duration * sample_rate / 5),
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        fbank_fn = SpeedKaldiFbank(in_key="audio", dither=1.0, dynamic_dither=True)
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
            {"map": [fbank_fn]},
            {"map": [partial(fbank_norm, mean=fbank_mean, std=fbank_std)]},
            {"shuffle": [shuffle_buffer_size]},
        ]
        super().__init__(dataset, pipeline)


class WrappedMCC40MDataset(DataPipeline):

    def __init__(
        self,
        url2index_list: list,
        sample_rate: int,
        duration: float,
        shuffle_buffer_size: int,
        sub_shuffle_buffer_size: int,
        batch_size: int,
        fbank_dim: int,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.1,
        aed_filtered: bool = True,
        avoid_sound_effect: bool = True,
        avoid_vocal: bool = True,
        max_vocal_threshold: float = 0.25,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: Optional[int] = 3,
        handler: Callable = wds.warn_and_continue,
        num_samples: int = -1,
        seed: int = 2023,
        **kwargs,
    ):
        mcc_datasets = [
            MCC40MDataset(
                url2index=url2index,
                sample_rate=sample_rate,
                duration=duration,
                shuffle_buffer_size=sub_shuffle_buffer_size,
                audio_key=audio_key,
                min_volume_threshold=min_volume_threshold,
                loudness_ratio_threshold=loudness_ratio_threshold,
                aed_filtered=aed_filtered,
                avoid_sound_effect=avoid_sound_effect,
                avoid_vocal=avoid_vocal,
                max_vocal_threshold=max_vocal_threshold,
                exclude_licenses=exclude_licenses,
                max_num_crops=max_num_crops,
                handler=handler,
                **kwargs,
            ) for url2index in url2index_list
        ]
        datasets = MultiIterableDataset(
            datasets=mcc_datasets,
            num_samples=num_samples,
            weights=[1.0 for _ in range(len(mcc_datasets))],
            seed=seed
        )
        fbank_collate = FbankCollate(fbank_dim=fbank_dim)
        super().__init__(
            datasets,
            wds.shuffle(shuffle_buffer_size),
            wds.batched(
                batch_size,
                collation_fn=partial(collation_fn, fbank_collate=fbank_collate)
            ),
        )


class KaraokeDataset(DataPipeline):

    def __init__(
        self,
        urls: str,
        sample_rate: int,
        duration: float,
        fbank_mean_path: str,
        fbank_std_path: str,
        fbank_dim: int,
        batch_size: int,
        audio_key: str = "acc.npy",
        sample_range_key: str = "acc_sample_range.npy",
        min_volume_threshold: float = 0.05,
        shuffle_buffer_size: int = 100,
        training: bool = True,
        **kwargs,
    ):
        fbank_mean = torch.load(fbank_mean_path).numpy()
        fbank_std = torch.load(fbank_std_path).numpy()
        karaoke_dataset = wds.WebDataset(
            urls=urls,
            **kwargs,
        )
        audio_transforms = MusicLMTransforms(
            n_samples=int(duration * sample_rate),
            audio_key=audio_key,
            sample_range_key=sample_range_key,
            min_volume_threshold=min_volume_threshold,
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        fbank_fn = SpeedKaldiFbank(in_key="audio.npy", dither=1.0, dynamic_dither=True)
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
            {"map": [fbank_fn]},
            {"map": [partial(fbank_norm, mean=fbank_mean, std=fbank_std)]},
        ]
        if training:
            pipeline.append({"shuffle": [shuffle_buffer_size]})

        dataset = WebPipeline(
            dataset=karaoke_dataset,
            pipeline=pipeline,
        )
        fbank_collate = FbankCollate(fbank_dim=fbank_dim)
        super().__init__(
            dataset,
            wds.batched(
                batch_size,
                collation_fn=partial(collation_fn, fbank_collate=fbank_collate)
            ),
        )
