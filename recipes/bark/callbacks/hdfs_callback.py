import logging
import os
from multiprocessing import Process, Queue

from pytorch_lightning.callbacks import Callback

from recipes.bark.utils.hdfs_tools import hdfs_mkdir, hdfs_put, hdfs_rm

logger = logging.getLogger(__name__)


class HdfsSavingCallback(Callback):
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
            ckpt_paths = self._find_newest_ckpts()
            if len(ckpt_paths) >= 1:
                ckpt_paths = ckpt_paths[0:2]
                for i, p in enumerate(ckpt_paths):
                    local_path = os.path.join(self.local_dir, p)
                    if i == 0:
                        force = True
                        fn = "last.ckpt"
                    else:
                        force = False
                        fn = os.path.basename(local_path)
                    print(
                        "Put {} to {}".format(
                            local_path, self.hdfs_path + "/checkpoints/" + fn
                        )
                    )
                    if force:
                        hdfs_rm(self.hdfs_path + "/checkpoints/" + fn)
                    hdfs_put(
                        local_path, self.hdfs_path + "/checkpoints/" + fn, force=force
                    )
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
        ckpts = [c for c in ckpts if c.endswith(".ckpt") and "last" not in c]
        ckpts.sort(key=lambda x: -int(x.split("-")[1].split("=")[-1]))
        return ckpts

