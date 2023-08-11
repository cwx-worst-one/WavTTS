import pytest
import torch
import torchaudio
from tqdm import tqdm

from recipes.datasets.billboard_hot200 import BillboardHot200WebDataModule
from recipes.soundstorm2.lightning.dac import DACModel
from tests.helpers.testing_utils import torch_device
from tests.helpers.runif import RunIf

@pytest.mark.parametrize("n_seconds", [1, 4, 11])
@RunIf(min_cuda_gpus=1)
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


def test_dac_throughput():
    batch_size = 32
    n_seconds = 30
    sample_rate = 44100
    n_samples = n_seconds * sample_rate


    audio = torch.randn(batch_size, 2, n_samples, device=torch_device)

    dac = DACModel(src_sample_rate=sample_rate, target_sample_rate=sample_rate).to(torch_device)


    for _ in tqdm(range(100)):
        codes = dac(audio)
