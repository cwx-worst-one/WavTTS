from typing import Optional

import numpy as np
import pytest
import torch

from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import AcousticModel
from recipes.musiclm.lightning.acoustic_modeling.semantic_acoustic_model import (
    SemanticAcousticModel,
)
from recipes.musiclm.lightning.semantic_modeling.mulan_semantic_model import (
    MulanSemanticModel,
)
from recipes.musiclm.lightning.unified_model import UnifiedModel
from tests.helpers.testing_utils import torch_device

sample_rate = 24000
codebook_size = 1024
max_sequence_length = 10


@pytest.fixture
def coarse_model():
    return AcousticModel(
        n_embd=32,
        n_head=2,
        n_layer=1,
        codebook_size=codebook_size,
        n_codebooks=4,
        max_sequence_length=max_sequence_length,
        sample_rate=sample_rate,
        quantizers=[0, 1, 2, 3],
        cond_quantizers=[],
        optimizer_class=None,
        scheduler_class=None,
    ).to(torch_device)


@pytest.fixture
def fine_model() -> AcousticModel:
    return AcousticModel(
        n_embd=32,
        n_head=2,
        n_layer=2,
        codebook_size=codebook_size,
        n_codebooks=12,
        max_sequence_length=max_sequence_length,
        sample_rate=sample_rate,
        quantizers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        cond_quantizers=[0, 1, 2, 3],
        optimizer_class=None,
        scheduler_class=None,
    ).to(torch_device)


@pytest.fixture
def semantic_acoustic_model() -> SemanticAcousticModel:
    model = SemanticAcousticModel(
        n_embd=32,
        n_head=2,
        n_layer=1,
        codebook_size=codebook_size,
        n_codebooks=12,
        max_sequence_length=max_sequence_length,
        sample_rate=sample_rate,
        quantizers=[0, 1, 2, 3],
        cond_quantizers=[],
        optimizer_class=None,
        scheduler_class=None,
    ).to(torch_device)
    model.on_train_epoch_start()
    return model


@pytest.fixture
def semantic_model() -> MulanSemanticModel:
    model = MulanSemanticModel(
        n_embd=32,
        n_head=2,
        n_layer=1,
        codebook_size=codebook_size,
        n_codebooks=12,
        max_sequence_length=max_sequence_length,
        optimizer_class=None,
        scheduler_class=None,
    ).to(torch_device)
    model.on_train_epoch_start()
    return model


@pytest.fixture
def unified_model() -> UnifiedModel:
    model = UnifiedModel(
        n_embd=32,
        n_head=2,
        n_layer=1,
        codebook_size=codebook_size,
        n_codebooks=12,
        max_sequence_length=max_sequence_length,
        sample_rate=sample_rate,
        quantizers=[0, 1, 2, 3],
        cond_quantizers=[],
        optimizer_class=None,
        scheduler_class=None,
    ).to(torch_device)
    model.on_train_epoch_start()
    return model


@pytest.mark.parametrize("seq_len", [None, 8])
@pytest.mark.parametrize("prefix_len", [0, 5])
def test_sampling_acoustic_model(
    fine_model: AcousticModel, seq_len: Optional[int], prefix_len: int
):
    target_seq_len = fine_model.hparams.max_sequence_length
    if seq_len is not None:
        target_seq_len = seq_len

    prefix = None
    if prefix_len > 0:
        prefix = (
            torch.randint(
                0, codebook_size, (1, len(fine_model.input_quantizers), prefix_len)
            )
            .long()
            .to(fine_model.device)
        )

    audio = torch.randn(1, 240000, device=torch_device)
    fine_token_ids = fine_model.sample_with_audio_conditioning(
        audio, temperature=1.0, seq_len=seq_len, prefix=prefix
    )

    assert fine_token_ids.shape == (
        1,
        len(fine_model.input_quantizers),
        target_seq_len - prefix_len,
    )
    assert fine_token_ids.min() >= 0
    assert fine_token_ids.max() < fine_model.hparams.codebook_size


@pytest.mark.parametrize("seq_len", [None, 8])
@pytest.mark.parametrize("prefix_len", [0, 5])
def test_sampling_semantic_acoustic_model(
    semantic_acoustic_model: SemanticAcousticModel,
    seq_len: Optional[int],
    prefix_len: int,
):
    target_seq_len = semantic_acoustic_model.hparams.max_sequence_length
    if seq_len is not None:
        target_seq_len = seq_len

    prefix = None
    if prefix_len > 0:
        prefix = (
            torch.randint(
                0,
                codebook_size,
                (1, len(semantic_acoustic_model.input_quantizers), prefix_len),
            )
            .long()
            .to(semantic_acoustic_model.device)
        )

    audio = torch.randn(1, 240000, device=torch_device)
    coarse_token_ids = semantic_acoustic_model.sample_with_audio_conditioning(
        audio, temperature=1.0, seq_len=seq_len, prefix=prefix
    )
    assert coarse_token_ids.shape == (
        1,
        len(semantic_acoustic_model.input_quantizers),
        target_seq_len - prefix_len,
    )
    assert coarse_token_ids.min() >= 0
    assert coarse_token_ids.max() < semantic_acoustic_model.hparams.codebook_size


@pytest.mark.parametrize("seq_len", [None, 8])
@pytest.mark.parametrize("prefix_len", [0, 5])
def test_sampling_semantic_model(
    semantic_model: MulanSemanticModel, seq_len: Optional[int], prefix_len: int
):
    target_seq_len = semantic_model.hparams.max_sequence_length
    if seq_len is not None:
        target_seq_len = seq_len

    prefix = None
    if prefix_len > 0:
        prefix = (
            torch.randint(0, codebook_size, (1, prefix_len))
            .long()
            .to(semantic_model.device)
        )

    audio = torch.randn(1, 240000, device=torch_device)
    semantic_token_ids = semantic_model.sample_with_conditioning(
        cond_data=audio,
        temperature=1.0,
        data_type="music",
        seq_len=seq_len,
        prefix=prefix,
    )
    assert semantic_token_ids.shape == (1, target_seq_len - prefix_len)
    assert semantic_token_ids.min() >= 0
    assert semantic_token_ids.max() < semantic_model.hparams.codebook_size


@torch.no_grad()
def test_kv_cache_acoustic_model(fine_model: AcousticModel):
    n_samples = 100
    batch_size = 5
    model = fine_model.model

    audio = torch.randn(batch_size, 240000, device=torch_device)
    sampled_ids = fine_model.prepare_cond_token_ids(audio=audio)
    sampled_ids = sampled_ids[:, :1]

    for _ in range(n_samples):
        logits = model(sampled_ids)
        sampled = logits[:, -1].softmax(dim=-1).argmax(dim=-1, keepdim=True)
        sampled_ids = torch.cat((sampled_ids, sampled), dim=1)

    # 2. k/v cache without prefix
    sampled_ids_cache = sampled_ids[:, :1]

    kv_cache = model.transformer.init_cache()
    for idx in range(n_samples):
        logits = model(sampled_ids_cache[:, idx : idx + 1])
        sampled = logits[:, -1].softmax(dim=-1).argmax(dim=-1, keepdim=True)
        sampled_ids_cache = torch.cat((sampled_ids_cache, sampled), dim=1)
    model.transformer.deinit_cache()
    np.testing.assert_array_almost_equal(sampled_ids.cpu(), sampled_ids_cache.cpu())

    # 3. k/v cache with prefix
    prefix_len = 10
    sampled_ids_cache_prefix = sampled_ids[:, : prefix_len + 1].clone()

    kv_cache2 = model.transformer.init_cache()
    model(sampled_ids_cache_prefix[:, :prefix_len])

    for k, k2 in zip(kv_cache.values(), kv_cache2.values()):
        for i in range(prefix_len):
            np.testing.assert_allclose(k[:, i].cpu(), k2[:, i].cpu(), atol=1e-7)

    for idx in range(prefix_len, n_samples):
        logits = model(sampled_ids_cache_prefix[:, idx : idx + 1])
        sampled = logits[:, -1].softmax(dim=-1).argmax(dim=-1, keepdim=True)
        sampled_ids_cache_prefix = torch.cat((sampled_ids_cache_prefix, sampled), dim=1)
    model.transformer.deinit_cache()
    np.testing.assert_array_almost_equal(
        sampled_ids.cpu(), sampled_ids_cache_prefix.cpu()
    )


def test_sampling_unified_model(unified_model):
    input_audio = torch.randn(1, 240000, device=unified_model.device)

    coarse_token_ids = unified_model.sample_ssa(input_audio, input_audio, temperature=1)

    assert coarse_token_ids.shape == (
        1,
        len(unified_model.hparams.quantizers),
        unified_model.hparams.max_sequence_length,
    )
    assert coarse_token_ids.min() >= 0
    assert coarse_token_ids.max() < unified_model.hparams.codebook_size
