import os

import pytest

from samantha.utils.parser import parse_arguments


@pytest.mark.parametrize("action", ["fit", "validate", "test", "predict"])
def test_parse_arguments_config(global_datadir, action):
    test_config = os.path.join(global_datadir, "config.yaml")
    filename, run_opts, overrides = parse_arguments(
        [
            action,
            "--config",
            str(test_config),
            "--seed=3",
            "--data_dir",
            "TIMIT",
            "--training_params.batch_size=128",
        ]
    )
    assert filename == test_config
    assert run_opts.action == action
    assert overrides == {
        "data_dir": "TIMIT",
        "seed": 3,
        "training_params": {"batch_size": 128},
    }
