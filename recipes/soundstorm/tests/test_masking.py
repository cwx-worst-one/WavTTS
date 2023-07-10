import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from recipes.soundstorm.lightning.masking_scheme import (
    SoundStormMaskingScheme,
    cosine_schedule,
)
from tests.unittests.models.utils import ids_tensor


def test_cosine_schedule():
    n_iterations = 16
    mask_ratios = []
    for i in range(n_iterations):
        mask_ratios.append(cosine_schedule(ratio=torch.tensor(i)))

    plt.plot(mask_ratios)
    plt.savefig("test_cosine_schedule.png")
    plt.close()


@pytest.mark.parametrize("batch_size", [1, 4])
def test_masking_scheme(batch_size):
    audio_codebook_size = 1024
    mask_token_id = audio_codebook_size + 1
    n_quantizers = 12
    max_seq_len = 1500
    audio_tokens = ids_tensor(
        (batch_size, n_quantizers, max_seq_len), vocab_size=audio_codebook_size
    )

    masking_scheme = SoundStormMaskingScheme(sample_q_uniformly=True, sample_t=False)
    masked_tokens, _, _ = masking_scheme(audio_tokens, mask_token_id)
    assert (masked_tokens == mask_token_id).sum() > 0

    masked_tokens[masked_tokens == mask_token_id] = -10000
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    for batch_idx in range(batch_size):

        ax[batch_idx].imshow(
            masked_tokens[batch_idx], interpolation="none", aspect="auto"
        )
        ax[batch_idx].set_ylabel("RVQ")
        ax[batch_idx].set_yticks(np.arange(0, n_quantizers))
        ax[batch_idx].set_yticklabels(np.arange(1, n_quantizers + 1))
        ax[batch_idx].set_title(f"Batch example {batch_idx}")

    plt.xlabel("Sequence length")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"test_masking_scheme.png")
    plt.close()
