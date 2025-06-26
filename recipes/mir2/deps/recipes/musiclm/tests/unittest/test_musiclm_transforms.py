import numpy as np
import pytest

from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from recipes.musiclm.transforms.musiclm import MusicLMTransforms

SR = 24000  # 24kHz


@pytest.mark.parametrize("num_crops", [0, 1, 2])
def test_multi_crops(num_crops):
    transforms = MusicLMTransforms(
        n_samples=10 * SR, audio_key="audio.npy", num_crops=num_crops
    )
    dataset = WebDatasetBufferPreprocessor(sample_rate=SR, transforms=transforms)
    buffer = []
    for _ in range(3):
        buffer.append({"audio.npy": np.random.randn(30 * SR).astype(np.float32)})
    items = [_ for _ in dataset.train_buffer_preprocessor(buffer)]
    assert len(items) == num_crops * len(buffer)
