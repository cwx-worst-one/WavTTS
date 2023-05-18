from webdataset import WebDataset

from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from recipes.musiclm.transforms.musiclm import MusicLMTransforms
from samantha.dataio.webdataset.pipeline import WebPipeline


class KaraokeDataset(WebPipeline):
    def __init__(
        self,
        urls: str,
        sample_rate: int,
        duration: float,
        audio_key: str = "acc.npy",
        sample_range_key: str = "acc_sample_range.npy",
        min_volume_threshold: float = 0.05,
        **kwargs,
    ):
        dataset = WebDataset(urls=urls, **kwargs)
        audio_transforms = MusicLMTransforms(
            n_samples=int(duration * sample_rate),
            audio_key=audio_key,
            sample_range_key=sample_range_key,
            min_volume_threshold=min_volume_threshold,
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate, transforms=audio_transforms
        )
        pipeline = []
        pipeline.append("decode")
        pipeline.append({"compose": [preprocessor.train_buffer_preprocessor]})
        super().__init__(dataset, pipeline)
