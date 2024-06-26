import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Dict

import git
from hyperpyyaml import resolve_references
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from samantha.utils.hdfs_helper import ishdfs, put

logger = logging.getLogger()


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
    temp_logs = tempfile.mkdtemp()
    local_path = os.path.join(temp_logs, "config.yaml")

    try:
        with open(local_path, "w", encoding="utf-8") as w:
            shutil.copyfileobj(resolved_yaml, w)

        if ishdfs(hparams_save_path):
            put(local_path, hparams_save_path)
        else:
            if not os.path.exists(hparams_save_path):
                shutil.copy(local_path, hparams_save_path)
    finally:
        shutil.rmtree(temp_logs)


@rank_zero_only
def save_dummy_model(trainer, pl_module, check_load_ckpt: bool = False):
    out_fp = os.path.join("model.ckpt")
    trainer.strategy.connect(pl_module)

    module_name = pl_module.__class__.__name__

    logger.info(f"({module_name}) Saving model checkpoint to: {out_fp}")
    trainer.save_checkpoint(out_fp)

    if check_load_ckpt:
        logger.info(f"({module_name}) Loading model checkpoint from: {out_fp}")
        tik = time.perf_counter()
        pl_module.load_from_checkpoint(out_fp)
        tok = time.perf_counter()
        logger.info(f"({module_name}) It took {tok - tik} to load the model checkpoint")


def get_git_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return repo.git.rev_parse(sha, short=7)


def get_version_name() -> str:
    git_hash = get_git_hash()
    datetime_str = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
    return os.path.join(git_hash, datetime_str)


def wandb_sanitize(s: str) -> str:
    return s.replace("/", "_")
