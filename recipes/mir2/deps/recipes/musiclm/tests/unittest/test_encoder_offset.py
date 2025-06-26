import pytest
import torch
from einops import repeat

from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import generate_offsets


@pytest.mark.parametrize("batch_size", [1, 2])
def test_offsets_single_channel(batch_size):
    seq_len = 4
    num_quantizers = 1
    vocab_size = 8

    audio_tokens = torch.tensor([[3, 2, 7, 0]])

    audio_tokens = repeat(audio_tokens, "q s -> b q s", b=batch_size)
    offsets = generate_offsets(
        seq_len,
        vocab_size,
        start_channel=0,
        end_channel=num_quantizers,
        device=audio_tokens.device,
    )

    audio_tokens = audio_tokens + offsets

    assert torch.equal(
        audio_tokens[batch_size - 1, 0],
        torch.tensor(
            [
                3 + vocab_size * 0,
                2 + vocab_size * 0,
                7 + vocab_size * 0,
                0 + vocab_size * 0,
            ]
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_offsets_multi_channel(batch_size):
    seq_len = 4
    num_quantizers = 3
    vocab_size = 8

    audio_tokens = torch.tensor([[3, 2, 7, 0], [1, 6, 0, 3], [4, 5, 0, 2]])

    audio_tokens = repeat(audio_tokens, "q s -> b q s", b=batch_size)
    offsets = generate_offsets(
        seq_len,
        vocab_size,
        start_channel=0,
        end_channel=num_quantizers,
        device=audio_tokens.device,
    )

    audio_tokens = audio_tokens + offsets

    assert torch.equal(
        audio_tokens[batch_size - 1, 0],
        torch.tensor(
            [
                3 + vocab_size * 0,
                2 + vocab_size * 0,
                7 + vocab_size * 0,
                0 + vocab_size * 0,
            ]
        ),
    )
    assert torch.equal(
        audio_tokens[batch_size - 1, 1],
        torch.tensor(
            [
                1 + vocab_size * 1,
                6 + vocab_size * 1,
                0 + vocab_size * 1,
                3 + vocab_size * 1,
            ]
        ),
    )
    assert torch.equal(
        audio_tokens[batch_size - 1, 2],
        torch.tensor(
            [
                4 + vocab_size * 2,
                5 + vocab_size * 2,
                0 + vocab_size * 2,
                2 + vocab_size * 2,
            ]
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_offsets_start_single_channel(batch_size):
    start_channel = 2
    seq_len = 4
    num_quantizers = 1
    vocab_size = 8

    audio_tokens = torch.tensor([[3, 2, 7, 0]])

    audio_tokens = repeat(audio_tokens, "q s -> b q s", b=batch_size)
    offsets = generate_offsets(
        seq_len,
        vocab_size,
        start_channel=start_channel,
        end_channel=start_channel + num_quantizers,
        device=audio_tokens.device,
    )

    audio_tokens = audio_tokens + offsets
    assert torch.equal(
        audio_tokens[batch_size - 1, 0],
        torch.tensor(
            [
                3 + vocab_size * 0,
                2 + vocab_size * 0,
                7 + vocab_size * 0,
                0 + vocab_size * 0,
            ]
        )
        + (start_channel * vocab_size),
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_offsets_start_multi_channel(batch_size):
    start_channel = 2
    seq_len = 4
    num_quantizers = 3
    vocab_size = 8

    audio_tokens = torch.tensor([[3, 2, 7, 0], [1, 6, 0, 3], [4, 5, 0, 2]])

    audio_tokens = repeat(audio_tokens, "q s -> b q s", b=batch_size)
    offsets = generate_offsets(
        seq_len,
        vocab_size,
        start_channel=start_channel,
        end_channel=start_channel + num_quantizers,
        device=audio_tokens.device,
    )

    audio_tokens = audio_tokens + offsets
    assert torch.equal(
        audio_tokens[batch_size - 1, 0],
        torch.tensor(
            [
                3 + vocab_size * 0,
                2 + vocab_size * 0,
                7 + vocab_size * 0,
                0 + vocab_size * 0,
            ]
        )
        + (start_channel * vocab_size),
    )
    assert torch.equal(
        audio_tokens[batch_size - 1, 1],
        torch.tensor(
            [
                1 + vocab_size * 1,
                6 + vocab_size * 1,
                0 + vocab_size * 1,
                3 + vocab_size * 1,
            ]
        )
        + (start_channel * vocab_size),
    )
    assert torch.equal(
        audio_tokens[batch_size - 1, 2],
        torch.tensor(
            [
                4 + vocab_size * 2,
                5 + vocab_size * 2,
                0 + vocab_size * 2,
                2 + vocab_size * 2,
            ]
        )
        + (start_channel * vocab_size),
    )
