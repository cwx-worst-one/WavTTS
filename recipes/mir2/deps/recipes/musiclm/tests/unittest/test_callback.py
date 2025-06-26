import pytest
from hyperpyyaml import load_hyperpyyaml

from recipes.musiclm.callbacks.frechet_audio_distance import (
    FrechetAudioDistanceCallback,
)
from sami_ai.utils.hparams import DotDict


@pytest.mark.skip("TODO")
def test_webdataset_audio():
    hparams_file = "./recipes/musiclm/conf/musiclm_coarse.yaml"

    # Load hyperparameter file with command-line overrides
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)

    cfg.pl_module = cfg.pl_module.to("cuda")
    callback = FrechetAudioDistanceCallback(
        dataset=cfg.validation_dataset,
        dataset_sample_rate=cfg.data_params.sample_rate,
        temperature=1.0,
    )
    callback.on_validation_end(None, cfg.pl_module)
