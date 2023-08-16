import pytest
import torch
import torchaudio

from recipes.datasets.billboard_hot200 import BillboardHot200WebDataModule
from recipes.soundstorm2.lightning.dac import DACModel
from tests.helpers.testing_utils import torch_device


@pytest.mark.parametrize("n_seconds", [1, 4, 11])
def test_dac(n_seconds):
    batch_size = 1
    sample_rate = 44100
    n_samples = n_seconds * sample_rate

    pl_datamodule = BillboardHot200WebDataModule(
        sample_rate=sample_rate,
        batch_size=batch_size,
        shuffle_buffer_size=100,
        duration=n_seconds,
        num_workers=0,
    )
    train_loader = pl_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    audio = batch["audio"].to(torch_device)

    # audio = torch.randn(batch_size, 1, n_samples, device=torch_device)

    dac = DACModel(src_sample_rate=sample_rate, target_sample_rate=sample_rate).to(
        torch_device
    )

    codes = dac(audio)
    audio_r = dac.decode(codes)

    torchaudio.save("test.mp3", audio[0].cpu(), sample_rate)
    torchaudio.save("test-dac.mp3", audio_r[0].cpu(), sample_rate)

    # expected_frames = dac.n_frames(n_seconds)
    # assert codes.shape == (batch_size, dac.num_quantizers, expected_frames)
