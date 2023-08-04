import sys
from unittest.mock import patch

import pytest

from samantha.main import main
from samantha.utils.hdfs_helper import rmdir, mkdir
import uuid


@pytest.fixture
def hdfs_ckpt_dir():
    return f"hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/ci/{uuid.uuid4().hex}"


@pytest.fixture
def fit_args(project_dir, hdfs_ckpt_dir):

    return (
        "prog",
        "fit",
        "--config",
        str(project_dir / "recipes" / "GPT2" / "conf" / "default.yaml"),
        "--run_opts.strategy=ddp",
        "--run_opts.enable_activation_ckpt=False",
        "--training_params.model=gpt2_tiny",
        f"--hdfs_model_ckpt.hdfs_path={hdfs_ckpt_dir}",
        "--trainer.profiler=null",
        "--training_params.num_epochs=1",
        "--training_params.steps_per_epoch=200",
    )


def test_gpt2_fit(fit_args, hdfs_ckpt_dir):
    assert mkdir(hdfs_ckpt_dir)

    with patch.object(sys, "argv", fit_args):
        main()
    rmdir(hdfs_ckpt_dir)
