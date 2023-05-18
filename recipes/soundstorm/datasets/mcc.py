from typing import Callable, List, Optional
import webdataset as wds
from recipes.musiclm.transforms.musiclm import MCCTransforms
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline

from samantha.dataio.dataset import MultiIterableDataset


class MCC40MDataset(WebPipeline):

    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        duration: float,
        shuffle_buffer_size: int,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.1,
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
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
            {"shuffle": [shuffle_buffer_size]},
        ]
        super().__init__(dataset, pipeline)


class WrappedMCC40MDataset(MultiIterableDataset):

    def __init__(
        self,
        url2index_list: list,
        sample_rate: int,
        duration: float,
        shuffle_buffer_size: int,
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
        datasets = [
            MCC40MDataset(
                url2index=url2index,
                sample_rate=sample_rate,
                duration=duration,
                shuffle_buffer_size=shuffle_buffer_size,
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

        super().__init__(
            datasets=datasets,
            num_samples=num_samples,
            weights=[1.0 for _ in range(len(datasets))],
            seed=seed,
        )
