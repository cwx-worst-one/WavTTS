import os

from hyperpyyaml import load_hyperpyyaml

from samantha.utils.experiment import create_experiment_directory


def test_create_experiment_directory(tmp_path, global_datadir):
    hparams_file = os.path.join(global_datadir, "config.yaml")
    overrides = {"training_params": {"batch_size": 128}}
    create_experiment_directory(tmp_path, hparams_file, overrides)

    config_filename = os.path.join(tmp_path, "config.yaml")
    assert os.path.isfile(config_filename)

    # Load hyperparameters file with command-line overrides
    with open(config_filename, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    assert hparams["seed"] == 0
    assert hparams["training_params"]["batch_size"] == 128
