import pytest
import torch

from recipes.soundstorm.lightning.soundstorm import SoundStorm, cosine_schedule
from tests.helpers.testing_utils import torch_device
from tests.unittests.models.utils import ids_tensor


@pytest.fixture
def soundstorm_model():
    return (
        SoundStorm(
            sample_rate=24000,
            n_embd=16,
            n_head=2,
            n_layer=2,
            conv_kernel_size=5,
            n_audio_samples=240000,
            sample_q_uniformly=False,
            audio_prompting=False,
            optimizer_class=None,
            scheduler_class=None,
            attention_kwargs={},
        )
        .eval()
        .to(torch_device)
    )


@pytest.mark.parametrize("batch_size", [1])
def test_decoding(soundstorm_model: SoundStorm, batch_size):
    vocab_size = 1024
    n_sec = 10
    audio_seq_len = soundstorm_model.audio_model.frame_rate * n_sec
    semantic_tokens = ids_tensor(
        (batch_size, soundstorm_model.semantic_model.frame_rate * n_sec),
        vocab_size,
        device=torch_device,
    )

    iterations = [16] + [1] * 15
    sampled_audio_tokens, _ = soundstorm_model.iterative_decoding(
        semantic_tokens, max_seq_len=audio_seq_len, iterations=iterations
    )

    assert sampled_audio_tokens.shape == (
        batch_size,
        soundstorm_model.audio_model.num_quantizers,
        audio_seq_len,
    )

    # we use greedy sampling at the last iteration, so there shouldn't be any masked tokens left
    assert (sampled_audio_tokens == soundstorm_model.mask_token_id).sum() == 0
