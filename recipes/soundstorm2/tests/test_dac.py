
import torch

from recipes.soundstorm2.lightning.dac import DACModel
from samantha.utils.audio import hz_to_samples
from tests.helpers.testing_utils import torch_device


def test_dac():
    batch_size = 1
    sample_rate = 44100
    n_samples = sample_rate * 10

    audio = torch.randn(batch_size, 1, n_samples, device=torch_device)
    dac = DACModel(sample_rate=sample_rate).to(torch_device)

    z, codes, latents, _, _ = dac.forward(audio, sample_rate)


    expected_samples = hz_to_samples(dac.frame_rate, dac.sample_rate)
    breakpoint()
    assert codes.shape == (batch_size, dac.num_quantizers, )
