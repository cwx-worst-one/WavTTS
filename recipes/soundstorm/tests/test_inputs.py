import pytest
import torch

from recipes.soundstorm.lightning.soundstorm import upsample_tokens
from tests.unittests.models.utils import ids_tensor


@pytest.mark.parametrize("batch_size", [1, 2])
def test_upsample_tokens(batch_size):
    semantic_frame_rate = 5
    audio_frame_rate = 10
    vocab_size = 1024
    semantic_seq_len = semantic_frame_rate * 3
    semantic_tokens = ids_tensor((batch_size, semantic_seq_len), vocab_size)

    upsample_rate = audio_frame_rate // semantic_frame_rate
    upsampled_tokens = upsample_tokens(semantic_tokens, rate=upsample_rate)
    assert upsampled_tokens.shape == (batch_size, semantic_seq_len * upsample_rate)
    assert torch.equal(upsampled_tokens[:, 0::2], upsampled_tokens[:, 1::2])
