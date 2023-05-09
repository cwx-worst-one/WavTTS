import numpy as np
import pytest
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from samantha.utils.hparams import DotDict


def load_config():

    hparams_file = "./recipes/audio_diffusion/conf/musiclm.yaml"

    # Load hyperparameter file with command-line overrides
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    return DotDict(hparams)


def initialize_dataloader(cfg):
    return cfg.pl_datamodule.train_dataloader()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.skip("Can only be run on Merlin for data verification")
def test_webdataset_audio(n_test_samples: int = 1):
    cfg = load_config()
    train_loader = initialize_dataloader()
    sample_rate = cfg.data_params.sample_rate
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == n_test_samples:
            break

        assert len(batch) == 3
        audio, emb = batch
        torchaudio.save(
            f"{batch_idx}-test_webdataset_audio-coarse-audio.mp3",
            audio[0],
            sample_rate=sample_rate,
        )


@pytest.mark.skip("Can only be run on Merlin for data verification")
def test_webdataset_audio_encode_decode(n_test_samples: int = 5):
    cfg = load_config()
    train_loader = initialize_dataloader(cfg)
    sample_rate = cfg.data_params.sample_rate
    pl_module = cfg.pl_module

    pl_module = pl_module.to(get_device())

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == n_test_samples:
            break

        audio, emb = batch
        audio = audio.to(pl_module.device)

        with torch.no_grad():
            audio_emb = pl_module.cuda_transforms(audio)
            dec_audio = pl_module.cuda_transforms.decode(audio_emb)

            audio_emb[:, 2:, :] = 0
            dec_coarse_only_audio = pl_module.cuda_transforms.decode(audio_emb)

            torchaudio.save(
                f"{batch_idx}-original-test_webdataset_audio.mp3",
                audio[0].cpu(),
                sample_rate=sample_rate,
            )
            torchaudio.save(
                f"{batch_idx}-decoded-test_webdataset_audio.mp3",
                dec_audio[0].cpu(),
                sample_rate=sample_rate,
            )

            torchaudio.save(
                f"{batch_idx}-decoded-coarse-only-test_webdataset.mp3",
                dec_coarse_only_audio[0].cpu(),
                sample_rate=sample_rate,
            )


@pytest.mark.skip()
def test_webdataset_audio_level():
    import matplotlib.pyplot as plt
    import pyloudnorm as pyln

    cfg = load_config()
    train_loader = initialize_dataloader(cfg)
    sample_rate = cfg.data_params.sample_rate
    meter = pyln.Meter(sample_rate)  # create BS.1770 meter

    ls = []
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        audio, emb = batch

        for a in audio:
            loudness = meter.integrated_loudness(a.permute(1, 0).numpy())
            ls.append(loudness)

            if loudness < -40:
                torchaudio.save(
                    f"test-audio-loudness-lvl-{loudness}.mp3", a, sample_rate
                )

        if batch_idx > 1000:
            break

    ls = np.array(ls)
    ls[ls == -np.inf] = 0
    plt.hist(ls, density=True, bins=100)
    plt.title("WebDataset audio loudness level (ITU-R BS.1770-4)")
    plt.savefig("test.png")
    plt.close()
