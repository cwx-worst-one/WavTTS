import logging
import multiprocessing
import os
import tempfile
from datetime import timedelta
from multiprocessing.pool import Pool
from typing import Any, Optional

import pytorch_lightning as pl
from bytedance import easycycle
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from samantha.utils.common import get_git_revision_hash
from samantha.utils.envs import getenv_bool
from samantha.utils.hdfs_tools import hdfs_mkdir, hdfs_put

logger = logging.getLogger(__name__)


class HDFSModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        hdfs_path: Optional[_PATH] = None,
        num_sync_process: int = 5,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
    ):
        if hdfs_path is None:
            hdfs_path = os.getenv("ARNOLD_OUTPUT", None)
        self.hdfs_path = os.path.join(hdfs_path, "checkpoints")

        if dirpath is None:
            dirpath = tempfile.mkdtemp()
            rank_zero_warn(
                f"save ckeckpoints to tempdir={dirpath} cause it's not provided, "
                f"but will still sync all checkpoints into hdfs: {self.hdfs_path}"
            )

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        # record last.ckpt modify time.
        # use dict to support last-v1.ckpt, last-v2.ckpt, etc.
        # when reusing the same local checkpoint directory.
        self._last_mtime = dict()
        self._ckpt_history = set()
        self._worker_pool = Pool(num_sync_process)
        self.should_sync_platform = True

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        fs = get_filesystem(self.dirpath)
        if (isinstance(fs.protocol, str) and fs.protocol != "file") or (
            isinstance(fs.protocol, tuple) and fs.protocol[0] != "file"
        ):
            raise ValueError(
                f"Only support local path to save checkpoints, but got {self.dirpath}."
            )
        trainer.strategy.broadcast(self.dirpath)

        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)
            hdfs_mkdir(self.hdfs_path)
            self.register_model(trainer)

    def register_model(self, trainer):
        model_name = os.getenv("ModelName", None)
        model_arch = os.getenv("ModelArch", None)
        ckpt_type = os.getenv("CheckpointType", None)
        infer_type = os.getenv("InferType", None)
        packed_model = os.getenv("PackedModel", None)

        eval_step_start = os.getenv("EvalStepStart", "0")
        eval_step_interval = os.getenv("EvalStepInterval", "0")
        eval_objective_eval_uuid = os.getenv("EvalObjectiveEvalUUID", "")

        eval_runner_path = os.getenv("EvalRunnerPath", "")
        eval_diffusion_ckpt = os.getenv("EvalDiffusionCkpt", "")
        eval_vocoder_path = os.getenv("EvalVocoderPath", "")
        eval_branch_name = os.getenv("EvalBranchName", "")
        eval_branch_commit = os.getenv("EvalBranchCommit", "")
        eval_script = os.getenv("EvalScript", "")

        eval_params = {}
        if eval_runner_path != "":
            eval_params["RUNNER_PATH"] = eval_runner_path
        if eval_diffusion_ckpt != "":
            eval_params["DIFFUSION_CKPT_PATH"] = eval_diffusion_ckpt
        if eval_vocoder_path != "":
            eval_params["VOCODER_CKPT_PATH"] = eval_vocoder_path
        if eval_branch_name != "":
            eval_params["BRANCH_NAME"] = eval_branch_name
        if eval_branch_commit != "":
            eval_params["BRANCH_COMMIT"] = eval_branch_commit
        if eval_script != "":
            eval_params["EVAL_SCRIPT"] = eval_script

        training_run_id = os.getenv("TRAINING_RUN_ID", "0")

        if model_name is None or model_arch is None or ckpt_type is None:
            self.should_sync_platform = False
            logger.warning(
                f"Will not sync model checkpoints to platform cause some of those env "
                f"variables are not provided:\n"
                f"  ModelName={model_name}\n"
                f"  ModelArch={model_arch}\n"
                f"  CheckpointType={ckpt_type}\n"
                f"  InferType={infer_type}\n"
                f"  PackedModel={packed_model}"
            )
            return None
        wandb_run_id, project = "", ""
        if infer_type is not None or packed_model is not None:
            pool = multiprocessing.Pool(1)
            for _logger in trainer.loggers:
                if isinstance(_logger, WandbLogger):
                    project = _logger.name
                    name = _logger.experiment.name
                    wandb_run_id = pool.apply(
                        easycycle.register_evaluation_wandb,
                        kwds={"project": project, "name": name},
                    )
                    break
            pool.close()
        req = easycycle.RegisterRawModelReq()
        req.name = model_name
        req.model_type = os.getenv("ModelType", "Common")
        req.model_arch = model_arch
        req.ckpt_type = ckpt_type
        req.infer_type = infer_type
        req.train_task_params = easycycle.TrainTaskParams(
            task_marking=packed_model or "",
            train_task_id=os.getenv("ARNOLD_TRIAL_ID", "null"),
            train_dataset_id=os.getenv("DatasetID", "null"),
            train_task_type="MERLIN" if os.getenv("MERLIN_JOB_ID", None) else "ARNOLD",
            wandb_run_id=wandb_run_id,
            wandb_project_name=project,
            merlin_job_id=os.getenv("MERLIN_JOB_ID", "null"),
            commit_id=get_git_revision_hash(),
            training_run_id=int(training_run_id),
            eval_config=easycycle.TrainTaskParamsEvalConfig(
                int(eval_step_start), int(eval_step_interval), eval_objective_eval_uuid
            ),
            eval_params=eval_params,
        )
        req.owner = os.getenv("ARNOLD_TRIAL_OWNER", "samantha")
        easycycle.register_raw_model(req)

    @rank_zero_only
    def check_and_sync_checkpoints(self, global_step):
        def maybe_sync_last(ckpt_path):
            fn = os.path.basename(ckpt_path)
            if "last" not in fn:
                return False

            mtime = os.path.getmtime(ckpt_path)
            if mtime == self._last_mtime.get(fn, None):
                return False
            self._last_mtime.update({fn: mtime})
            return True

        ckpts = self.list_checkpoints()
        for ckpt in ckpts:
            if maybe_sync_last(ckpt) or ckpt not in self._ckpt_history:
                fn = os.path.basename(ckpt)
                self._ckpt_history.add(ckpt)
                hdfs_path = os.path.join(self.hdfs_path, fn)
                self._worker_pool.apply_async(
                    func=self._sync_checkpoint,
                    args=(
                        ckpt,
                        hdfs_path,
                        "last" in fn,
                        self.should_sync_platform,
                        global_step,
                    ),
                    error_callback=lambda e: logger.warning(f"{e}"),
                    callback=lambda e: logger.info(f"{e}"),
                )

    @classmethod
    def _sync_checkpoint(
        cls, local_path, hdfs_path, force, should_sync_platform, global_step
    ):
        hdfs_put(local_path, hdfs_path, force=force)
        logger.info(f"Synced {local_path=} to {hdfs_path=}.")
        if force:
            return
        if not should_sync_platform:
            return
        model_name = os.getenv("ModelName", None)
        req = easycycle.RegisterCkptsReq()
        req.raw_model_name = model_name
        req.creator = os.getenv("ARNOLD_TRIAL_OWNER", "samantha")
        req.train_task_id = os.getenv("ARNOLD_TRIAL_ID", "null")

        req.checkpoints = [
            easycycle.Checkpoint(
                name=os.path.basename(hdfs_path),
                hdfs=hdfs_path,
                extra={"step": str(global_step)},
                auto_valid=getenv_bool("AutoEval", False),
            )
        ]
        easycycle.register_ckpts(req)

    def list_checkpoints(self):
        ckpt_path = self.dirpath
        if self._fs.exists(ckpt_path):
            return {os.path.normpath(p) for p in self._fs.ls(ckpt_path, detail=False)}
        return set()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.check_and_sync_checkpoints(global_step=trainer.global_step)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.check_and_sync_checkpoints(global_step=trainer.global_step)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_validation_end(trainer, pl_module)
        self.check_and_sync_checkpoints(global_step=trainer.global_step)

    def teardown(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        self._worker_pool.close()
        self._worker_pool.join()

    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
        if (
            self.save_top_k != 0
            and self._fs.isdir(dirpath)
            and len(self._fs.ls(dirpath)) > 0
        ):
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
