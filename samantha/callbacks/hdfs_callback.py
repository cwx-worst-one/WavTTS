import logging
import os
from multiprocessing import Process, Queue

from pytorch_lightning.callbacks import Callback

from samantha.utils.hdfs_tools import hdfs_mkdir, hdfs_put

logger = logging.getLogger(__name__)


class HdfsSavingCallback(Callback):
    r"""This callback will sync local checkpoints to hdfs.

    User can specify this callback like any others in yaml config:

    .. code-block::yaml

        trainer: !new:pytorch_lightning.Trainer
            ...
            callbacks:
              - !new:samantha.callbacks.HdfsSavingCallback
                hdfs_path: hdfs://xxx
                save_dir: !ref <run_opts[tensorboard_dir]>
                name: !ref <run_opts[log_name]>
                version: !ref <run_opts[version]>
                n_log: 1000
                async_upload: True

    Args:
        hdfs_path (str): target hdfs path
        save_dir (str): save directory,
            same as :class:`pytorch_lightning.loggers.TensorBoardLogger`
        name (str): experiment name,
            same as :class:`pytorch_lightning.loggers.TensorBoardLogger`
        version (str): experiment version,
            same as :class:`pytorch_lightning.loggers.TensorBoardLogger`
        n_log (int): synchronize interval
        async_upload (bool): do async upload or not
    """

    def __init__(self, hdfs_path, save_dir, name, version, n_log, async_upload=False):
        self.hdfs_path = hdfs_path
        version = "version_{}".format(version)
        self.local_dir = os.path.join(save_dir, name, version, "checkpoints")
        if hdfs_path is not None:
            hdfs_mkdir(hdfs_path + "/checkpoints")
        self.n_log = n_log
        self._queue = None
        self._worker = None
        self._async_upload = async_upload
        self._ckpt_history = set()

    def on_after_backward(self, trainer, pl_module):
        global_rank = trainer.global_rank
        if self.hdfs_path is None or global_rank != 0:
            return
        if not self._async_upload:
            self._target_func(trainer.global_step)
        else:
            if self._queue is None:
                self._queue = Queue()
                self._worker = Process(target=self._target_wrapper, args=(self._queue,))
                self._worker.start()
            self._queue.put(trainer.global_step)

    def __del__(self):
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join()

    def _target_wrapper(self, queue):
        while True:
            global_step = queue.get()
            if global_step is None:
                break
            self._target_func(global_step)

    def _target_func(self, global_step):
        if global_step > 1 and global_step % self.n_log == 0:

            logger.info(f"Processing {global_step=} checkpoints")

            # sync checkpoints
            ckpt_paths = self._find_newest_ckpts()
            for p in ckpt_paths:
                local_path = os.path.join(self.local_dir, p)
                fn = os.path.basename(local_path)
                hdfs_path = self.hdfs_path + "/checkpoints/" + fn
                logger.info(f"Putting {local_path=} to {hdfs_path=}")

                hdfs_put(local_path, hdfs_path, force="last" in fn)

            # sync events
            hdfs_put(
                os.path.join(self.local_dir, "../", "events*"),
                self.hdfs_path + "/",
                force=True,
            )

            logger.info(f"Processed {global_step=} checkpoints")

    def _find_newest_ckpts(self):
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir, exist_ok=True)
            return []
        ckpts = os.listdir(self.local_dir)
        ckpts = [c for c in ckpts if c.endswith(".ckpt")]
        new_ckpts = []
        for ckpt in ckpts:
            fn = os.path.basename(ckpt)
            if ckpt not in self._ckpt_history or "last" in fn:
                new_ckpts.append(ckpt)
                self._ckpt_history.add(ckpt)
        return new_ckpts
