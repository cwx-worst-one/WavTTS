import os
import shutil
from typing import Dict

from hyperpyyaml import resolve_references

from samantha.utils.hdfs_helper import ishdfs, put


def create_experiment_directory(
    experiment_directory: str, hparams_file: str, overrides: Dict
):
    """Creates a SAMI AI experiment using the Trainer's log_dir attribute
    and saves the overridden config file in the created directory as
    `config.yaml`.

    Args:
        experiment_directory (str): Experiment directory path
        hparams_file (str): Original path to the hparams_file
        overrides (Dict): Overrides for the hparams_file
    """
    with open(hparams_file, encoding="utf-8") as f:
        resolved_yaml = resolve_references(f, overrides)

    hparams_save_path = os.path.join(experiment_directory, "config.yaml")
    os.makedirs("logs", exist_ok=True)
    local_path = os.path.join("logs", "config.yaml")

    try:
        with open(local_path, "w", encoding="utf-8") as w:
            shutil.copyfileobj(resolved_yaml, w)

        if ishdfs(hparams_save_path):
            put(local_path, hparams_save_path)
        else:
            shutil.copy(local_path, hparams_save_path)
    finally:
        os.remove(local_path)
