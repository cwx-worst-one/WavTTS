r"""This module provides list of io functionalities to handle pipieline output format.
"""

import filecmp
import itertools
import json
import logging
import os
import shutil
from typing import List

from bytedance import easycycle

import samantha.utils.hdfs_helper as hh
import samantha.utils.infer_dtype as infer
from samantha.plugins.torch_io import LargeTorchCheckpointIO
from samantha.utils.sail import SAILInferProcessor, compile_and_upload_model

logger = logging.getLogger(__name__)


def compare_and_copy(src_files, dst_dir):
    r"""Compare and copy files from src_files to dst_dir.

    If the file in src_files is not in dst_dir, copy it to dst_dir.

    Args:
        src_files (list): list of source files
        dst_dir (str): destination directory

    """

    for src_file in src_files:
        bn = os.path.basename(src_file)
        dst_file = os.path.join(dst_dir, bn)
        if not os.path.isfile(dst_file) or not filecmp.cmp(src_file, dst_file):
            shutil.copyfile(src_file, dst_file)  # pragma: no cover


def prep_training_output(log_dir):
    r"""Generate metadata.json for training output.

    Args:
        log_dir (str): training log directory
    """

    # process checkpoints

    result = []
    arnold_output = os.getenv("ARNOLD_OUTPUT", None)
    ckpt_io = LargeTorchCheckpointIO()
    if "checkpoints" in hh.walk_one(log_dir):
        checkpoints_folder = os.path.join(log_dir, "checkpoints")
        sorted_ckpt_filenames = sorted(hh.list_dir(checkpoints_folder))
        for cur_ckpt_fn in sorted_ckpt_filenames:
            cur_ckpt_loc = os.path.join(checkpoints_folder, cur_ckpt_fn)
            cur_ckpt = ckpt_io.load_checkpoint(cur_ckpt_loc, map_location="cpu")
            monitor_keys = list(cur_ckpt["callbacks"].keys())
            metas = {}

            for monitor_key in monitor_keys:
                cur_callback = cur_ckpt["callbacks"][monitor_key]
                cur_monitor = cur_callback["monitor"]
                current_score = cur_callback["current_score"]
                if current_score is not None:
                    current_score = float(current_score)
                metas[cur_monitor] = str(current_score)

            ckpt_hdfs = (
                cur_ckpt_loc
                if hh.ishdfs(log_dir)
                else os.path.join(arnold_output, "checkpoints", cur_ckpt_fn)
            )
            cur_ret = easycycle.Checkpoint(
                name=cur_ckpt_fn, hdfs=ckpt_hdfs, extra=metas
            )
            result.append(cur_ret)

    # process log
    base_dir = log_dir if hh.ishdfs(log_dir) else arnold_output
    tfevents = []
    for event_file in hh.list_dir(log_dir):
        if "tfevents" in event_file:
            file_loc = os.path.join(base_dir, event_file)
            tfevents.append(file_loc)
    if result:
        result[-1].extra.update(
            {
                "tfevents": ",".join(tfevents),
                "hparams": os.path.join(base_dir, "hparams.yaml"),
            }
        )
    else:
        logger.warning("No result generated after training.")
    return result


def prep_test_output(metrics):
    r"""Generate metadata.json for test output.

    Args:
        metrics (dict): metrics to be dumped
    """

    groups = easycycle.get_ckpt_groups()
    assert len(groups) == 1
    metrics = {k: float(v) for k, v in metrics.items()}
    return easycycle.EvaluateResult(raw_result=json.dumps(metrics))


def prep_predict_output(predictions: List[infer.Container]):
    r"""Generate metadata.json for predict output.

    Args:
        predictions (List[infer.Container]): predictions to be dumped
    """

    groups = easycycle.get_ckpt_groups()
    assert len(groups) == 1
    predictions = list(itertools.chain(*predictions))
    predictions = [SAILInferProcessor.process(prediction) for prediction in predictions]
    return [
        easycycle.InferResult(ckpt_group_id=groups[0].ckpt_group_id, tuples=predictions)
    ]


def prep_export_output(exported_dict, output_dir):
    r"""Generate metadata.json for export output.

    Args:
        exported_dict (dict): exported dict
        output_dir (str): output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, path in exported_dict.items():
        if isinstance(path, list):
            compare_and_copy(path, output_dir)
        else:
            compare_and_copy([path], output_dir)

    compile_and_upload_model(output_dir)


def parse_post_to_sail_params(run_opts):
    r"""Parse parameters for post_to_sail.

    Args:
        run_opts (Namespace): run options
        is_export (bool): currently do export or not
    """

    # experiment_id is required for post_to_sail
    exp_id = os.environ.get("SAIL_EXPERIMENT_ID", None)
    logger.info("experiment_id is {}".format(exp_id))

    # arnold_output_dir is required for post_to_sail
    arnold_output_dir = run_opts.arnold_output_dir
    if arnold_output_dir:
        # override environment variable
        os.environ["ARNOLD_OUTPUT"] = arnold_output_dir
    else:
        arnold_output_dir = os.environ.get("ARNOLD_OUTPUT", None)

    if not arnold_output_dir:
        logger.error("Please provide arnold_output_dir when post to sail.")
        raise ValueError("No arnold_output_dir is provided")
    logger.info("arnold_output_dir is {}".format(arnold_output_dir))


def export_model(pl_module, ckpt_path: str, output_dir):
    r"""Export model to SAIL.

    Args:
        pl_module (LightningModule): LightningModule
        ckpt_path (str): checkpoint path
        output_dir (str): local output directory
    """

    if not output_dir:
        raise ValueError("Output dir must be provided when doing model export.")

    if not ckpt_path:
        raise ValueError("`ckpt_path` not provided in command line arguments.")

    if hasattr(pl_module, "load_checkpoint"):  # pragma: no cover
        pl_module = pl_module.load_checkpoint(ckpt_path)
    else:  # pragma: no cover
        pl_module = pl_module.load_from_checkpoint(ckpt_path)
    exported_path = pl_module.model.export(
        resource_loc=os.path.join(output_dir, "resource.xml")
    )

    task_postprocess(output_dir, action="export", exported_path=exported_path)


def task_postprocess(
    output_dir, action="", exported_path=None, trainer=None, result=None
):
    r"""Post process after task."""
    exp_id = os.getenv("SAIL_EXPERIMENT_ID", None)
    logger.info(f"action: {action}, exp_id: {exp_id}, output_dir: {output_dir}")

    arnold_output_dir = os.getenv("ARNOLD_OUTPUT")

    output = None
    if action == "export":
        prep_export_output(exported_path, output_dir)
        if not hh.ishdfs(output_dir):
            hh.sync_hdfs_dir(output_dir, arnold_output_dir)

    if action == "fit":
        output = prep_training_output(output_dir)
        if not hh.ishdfs(output_dir):
            hh.sync_hdfs_dir(output_dir, arnold_output_dir)

    if action == "test":
        output = prep_test_output(trainer.callback_metrics)

    if action == "predict":
        output = prep_predict_output(result)

    upload_artifacts(action, output)


def upload_artifacts(action, artifacts):
    if artifacts is None:
        logger.warning(f"No artifacts on action {action}")
        return
    host = easycycle.Host.CN
    username = os.getenv("ARNOLD_TRIAL_OWNER", "samantha")
    if action == "fit":
        merlin_job_id = os.getenv("MERLIN_JOB_ID", None)
        is_merlin = merlin_job_id is not None
        req = easycycle.RegisterRawModelReq()
        req.name = os.getenv("ModelName", "TestModel")
        req.owner = username
        req.train_dataset_id = os.getenv("TrainDatasetID", "null")
        req.train_task_id = (
            merlin_job_id if is_merlin else os.getenv("ARNOLD_TRIAL_ID", "null")
        )
        req.train_task_type = "MERLIN" if is_merlin else "ARNOLD"
        req.checkpoints = artifacts
        easycycle.register_raw_model(req)
        # easycycle.upload_train_product(host=host, user=username, ckpts=artifacts)
    elif action == "test":
        easycycle.upload_evaluate_product(host=host, user=username, result=artifacts)
    elif action == "predict":
        easycycle.upload_infer_product(host=host, user=username, results=artifacts)
    else:
        logger.warning(f"No uploading for action {action}")
        return
    logger.info(f"Finished {action} and posted the artifacts to sail.")
