"""A single interface for training, validation, testing, inference, and tuning.

Use `python3 -m samantha.main -h` for usage help.
"""

import logging
import sys

try:
    from cruise import CruiseTrainer
except Exception:
    CruiseTrainer = None
from hyperpyyaml import load_hyperpyyaml
from pytorch_lightning import Trainer

from samantha.utils.benchmark import benchmark_model
from samantha.utils.experiment import create_experiment_directory
from samantha.utils.hdfs_tools import hdfs_open
from samantha.utils.hparams import DotDict
from samantha.utils.parser import parse_arguments
from samantha.utils.result_io import (
    export_model,
    parse_post_to_sail_params,
    task_postprocess,
)

logger = logging.getLogger()


def _get_extra_action_params(run_opts, cfg, action):
    """Get extra action params from hparams."""
    action_extra_params = {
        "fit": {"ckpt_path": None},
        "validate": {"ckpt_path": None, "verbose": True},
        "test": {"ckpt_path": None, "verbose": True},
        "predict": {"ckpt_path": None, "return_predictions": None},
        "export": {"ckpt_path": None},
        "benchmark": {},
    }
    assert action in action_extra_params, f"Invalid action: {action}"
    extra_params = {
        k: vars(run_opts).get(k, v) for k, v in action_extra_params[action].items()
    }
    # Special logic for ckpt_path
    extra_params["ckpt_path"] = extra_params.get("ckpt_path", None) or cfg.get(
        "ckpt_path", None
    )
    return extra_params


def main():
    r"""Single entrypoint to SAMI AI training, testing, inference, and tuning.
    Arguments are read from `sys.argv[1:]`.

    Use `python3 -m samantha.main -h` additional usage details.

    Example
    -------
    `python3 -m samantha.main fit --config \
        recipes/sample_project/conf/default.yaml --run_opts.fast_dev_run True`
    """
    # CLI:
    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])
    action = run_opts.action

    # Load hyperparameter file with command-line overrides
    if hparams_file.startswith("hdfs"):
        with hdfs_open(hparams_file, "r") as fin:
            hparams = load_hyperpyyaml(fin, overrides)
    else:
        with open(hparams_file, "r", encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin, overrides)

    cfg = DotDict(hparams)

    trainer = cfg.trainer
    pl_datamodule = cfg.pl_datamodule
    pl_module = cfg.pl_module
    if isinstance(trainer, Trainer):
        extra_params = _get_extra_action_params(run_opts, cfg, action)
    elif CruiseTrainer is not None and isinstance(trainer, CruiseTrainer):
        extra_params = {}
    else:
        raise ("Unsupported trainer type.")

    # If post_to_sail is True, we limit to only one atomic action
    # Moreover, we have required params such as:
    # exp_id, arnold_output_dir, output_dir
    # If post_to_sail is False, the algo user can set a list of actions
    # e.g. ['fit', 'validate', 'test', 'predict']
    if not run_opts.post_to_sail:
        logger.info("Results will NOT be posted to sail.")
        # run the trainer functions e.g.: `train.fit()`,
        if action == "export":
            raise ValueError(
                "Exporting model is not supported when post_to_sail is False."
            )

        if action == "benchmark":
            # benchmark action
            benchmark_model(trainer, pl_module, pl_datamodule, run_opts.output_dir)
            return

        fn = getattr(trainer, action)
        fn(model=pl_module, datamodule=pl_datamodule, **extra_params)
    else:
        # need to post to sail
        logger.info("Results will be posted to sail.")
        parse_post_to_sail_params(run_opts)

        if action == "export" and trainer.is_global_zero:
            # export action
            export_model(pl_module, extra_params["ckpt_path"], run_opts.output_dir)
            return

        # only do 1 atomic action per call
        fn = getattr(trainer, action)
        # TODO: @wangxin.colin gather all output
        result = fn(model=pl_module, datamodule=pl_datamodule, **extra_params)

        if trainer.is_global_zero and action in ["fit", "test", "predict"]:
            task_postprocess(
                output_dir=trainer.log_dir if action == "fit" else None,
                action=action,
                trainer=trainer,
                result=result,
            )

    if not trainer.fast_dev_run and trainer.global_rank == 0:
        # Save hparams_file to "config.yaml" in experiment directory
        if isinstance(trainer, Trainer):
            log_dir = trainer.log_dir
        elif CruiseTrainer is not None and isinstance(trainer, CruiseTrainer):
            log_dir = trainer.default_root_dir

        create_experiment_directory(
            log_dir, hparams_file, overrides
        )  # pragma: no cover


if __name__ == "__main__":
    main()  # pragma: no cover
