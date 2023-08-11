import pytest
import torch

from recipes.soundstorm2.lightning.soundstorm import SoundStorm
from recipes.soundstorm2.tests.test_model import soundstorm, SoundStormModelTester
from tests.helpers.testing_utils import torch_device
from tests.unittests.models.utils import ids_tensor


@pytest.mark.parametrize("batch_size", [1, 3])
def test_decoding(soundstorm: SoundStormModelTester, batch_size):
    soundstorm = soundstorm.create_and_test_model()
    vocab_size = 1024
    n_sec = 10

    audio_seq_len = soundstorm.audio_model.frame_rate * n_sec
    # semantic_model_frame_rate = 25
    # semantic_tokens = ids_tensor(
    #     (batch_size, semantic_model_frame_rate * n_sec),
    #     vocab_size,
    #     device=torch_device,
    # )

    seed_tokens = ids_tensor(
        (batch_size, 4, audio_seq_len), vocab_size, device=torch_device
    )

    iterations = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
    score_strategies = ["maskgit"] * len(iterations)
    sampled_audio_tokens, _ = soundstorm.iterative_decoding(
        seed_tokens=seed_tokens,
        max_seq_len=audio_seq_len,
        iterations=iterations,
        score_strategies=score_strategies,
    )

    assert sampled_audio_tokens.shape == (
        batch_size,
        soundstorm.audio_model.n_quantizers,
        audio_seq_len,
    )

    # we use greedy sampling at the last iteration, so there shouldn't be any masked tokens left
    assert (sampled_audio_tokens == soundstorm.mask_token_id).sum() == 0
