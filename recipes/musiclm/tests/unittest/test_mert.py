import pytest
import torch
import torch.utils.benchmark as benchmark
from pytorch_lightning import seed_everything

from recipes.musiclm.lightning.mert import MERTModel, QuantizedMERTModel


@pytest.mark.parametrize("batch_size", [1, 2])
def test_semantic_encoder_module(batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MERTModel()
    model = model.to(device)

    test_audio = torch.randn(batch_size, 1, 240000, device=device)
    semantic_token_ids = model(test_audio)
    assert semantic_token_ids.shape == (batch_size, 748, 1024)

    t0 = benchmark.Timer(
        stmt="model(test_audio)",
        setup="",
        globals={"test_audio": test_audio, "model": model},
    )

    print(t0.timeit(100))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_quantized_mert(batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_audio = torch.randn(batch_size, 1, 240000, device=device)
    n_codebooks = 12

    seed_everything(42)
    model_1 = QuantizedMERTModel(train_head=False, n_codebooks=n_codebooks).to(device)
    semantic_token_ids_1 = model_1(test_audio)
    assert semantic_token_ids_1.shape == (batch_size, n_codebooks)

    seed_everything(42)
    model_2 = QuantizedMERTModel(train_head=False, n_codebooks=n_codebooks).to(device)
    semantic_token_ids_2 = model_2(test_audio)
    assert torch.allclose(semantic_token_ids_1, semantic_token_ids_2)
