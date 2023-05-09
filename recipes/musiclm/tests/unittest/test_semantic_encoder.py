import pytest
import torch
import torch.utils.benchmark as benchmark

from recipes.musiclm.lightning.semantic_model import SemanticModel


@pytest.mark.parametrize("batch_size", [1, 8])
def test_semantic_encoder_module(batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SemanticModel()
    model = model.to(device)

    test_audio = torch.randn(batch_size, 240000, device=device)
    semantic_token_ids = model(test_audio, padding=False)
    assert semantic_token_ids.shape == (batch_size, 250)

    semantic_token_ids = model(test_audio, padding=True)
    assert semantic_token_ids.shape == (batch_size, model.n_frames)
    assert torch.equal(semantic_token_ids[:, 0], semantic_token_ids[:, 1])
    assert torch.equal(semantic_token_ids[:, -3], semantic_token_ids[:, -1])
    assert torch.equal(semantic_token_ids[:, -2], semantic_token_ids[:, -1])

    t0 = benchmark.Timer(
        stmt="model(test_audio)",
        setup="",
        globals={"test_audio": test_audio, "model": model},
    )

    print(t0.timeit(100))
