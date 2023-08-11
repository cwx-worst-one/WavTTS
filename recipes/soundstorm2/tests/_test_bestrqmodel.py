import torch

from recipes.soundstorm2.lightning.bestrq import BestRQMelCTCModel, BestRQMelModel
from tests.helpers.runif import RunIf
from tests.helpers.testing_utils import torch_device


@RunIf(min_cuda_gpus=1)
def test_bestrq_mel():
    batch_size = 1
    n_seconds = 10
    model = BestRQMelModel().to(torch_device)
    # model.model.vq.sync_codebook = False

    n_samples = n_seconds * model.sample_rate

    audio = torch.randn(batch_size, 1, n_samples, device=torch_device)

    tokens = model(audio)
    assert tokens.shape == (batch_size, model.n_frames(n_seconds))

    model = BestRQMelCTCModel().to(torch_device)
    tokens = model(audio)
    assert tokens.shape == (batch_size, model.n_frames(n_seconds))
