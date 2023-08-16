import torch
import torchaudio

from recipes.soundstorm2.lightning.bestrq import BestRQMelModel
from tests.helpers.testing_utils import torch_device


def test_bestrq_mel():
    batch_size = 1
    n_seconds = 10
    model = BestRQMelModel().to(torch_device)
    # model.model.vq.sync_codebook = False

    n_samples = n_seconds * model.sample_rate

    audio = torch.randn(batch_size, 1, n_samples, device=torch_device)
    tokens = model(audio)
    assert tokens.shape == (batch_size, model.n_frames(n_seconds))
