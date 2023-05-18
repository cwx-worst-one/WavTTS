from functools import partial
from webdataset import WebDataset
from recipes.musiclm.transforms.musiclm import MusicLMTransforms
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.utils.datastructures import select_keys


class KaraokeDataset(WebPipeline):

    def __init__(
        self,
        urls: str,
        sample_rate: int,
        duration: float,
        batch_size: int,
        shuffle_buffer_size: int,
        audio_key: str = "acc.npy",
        sample_range_key: str = "acc_sample_range.npy",
        min_volume_threshold: float = 0.05,
        **kwargs,
    ):
        dataset = WebDataset(
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
        pipeline=[
            {"select": {"predicate": partial(
                select_keys,
                keys=[audio_key, sample_range_key]
            )}},
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
            {"shuffle": [shuffle_buffer_size]},
            {"to_tuple": ["audio.npy"]},
            {"batched": [batch_size]}
        ]
        super().__init__(dataset, pipeline)
