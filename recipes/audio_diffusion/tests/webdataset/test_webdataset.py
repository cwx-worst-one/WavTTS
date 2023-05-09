import pytest
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from samantha.utils.hparams import DotDict


def initialize_dataloader(cfg):
    return cfg.pl_datamodule.train_dataloader()


@pytest.mark.skip("Can only be run on Merlin for data verification")
def test_webdataset_audio(n_test_samples: int = 1):
    hparams_file = (
        "recipes/audio_diffusion/conf/"
        "140223-singsong/singsong_semantic-coarse2semantic-coarse.yaml"
    )

    # Load hyperparameter file with command-line overrides
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)
    train_loader = initialize_dataloader(cfg)
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
