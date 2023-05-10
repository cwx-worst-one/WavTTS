''' checkpoint hook. '''

from os import remove, path as osp
from core.utils import symlink, hdfs_put, hdfs_mkdir, hdfs_rm, hdfs_copy, dist_barrier
from .hook import HOOKS, Hook


@HOOKS.register_module()
class CheckpointHook(Hook):
    '''CheckpointHook.'''

    def __init__(
        self,
        interval=-1,
        save_optimizer=True,
        out_dir=None,
        max_hdfs_keep_ckpts=-1,
        max_local_keep_ckpts=-1,
        **kwargs
    ):
        '''init.'''
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_hdfs_keep_ckpts = max_hdfs_keep_ckpts
        self.max_local_keep_ckpts = max_local_keep_ckpts
        self.args = kwargs
        self.saved_hdfs_files = []
        self.saved_local_files = []

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        self._save_checkpoint(runner)

    def after_train_epoch(self, runner):
        if not runner.train_cfg.get('save_after_epoch', False):
            return
        self._save_checkpoint(runner)

    def after_val_epoch(self, runner):
        '''after_val_epoch.'''
        dist_barrier()
        runner.valid_log_buffer.get()

        flag = runner.update_best_metric()
        if not flag:
            # current model is not best, do nothing.
            return

        if (
            len(self.saved_local_files) == 0
            or 'step_{}.pth'.format(runner.iter + 1) != self.saved_local_files[-1]
        ):
            self._save_checkpoint(runner, flag)
            return

        if runner.rank != 0:
            return

        # this model is saved, just link
        chkpt_file = self.saved_hdfs_files[-1]
        symlink(chkpt_file, osp.join(self.out_dir, 'best.pth'))
        remote_save_root = runner.train_cfg.get('remote_save_root', None)
        if remote_save_root:
            hdfs_file = osp.join(
                remote_save_root,
                runner.train_cfg.save_dir,
                runner.train_cfg.save_name,
                'checkpoints/best.pth',
            )
            sync = runner.pipeline_cfg is None and runner.tensor_parallel_size == 1
            hdfs_put(
                osp.join(self.out_dir, chkpt_file), hdfs_file, sync=sync, retry=1, timeout=90 * 60
            )  # 90mins

    def _save_checkpoint(self, runner, is_best=False):
        '''
        save checkpoint file function for train and validation.
        Args:
            runner(Runner): should be a runner.
            is_best(bool): whether this model is best.
        Return:
            str: file name for checkpoint.
        '''
        if not self.out_dir:
            self.out_dir = osp.join(
                runner.train_cfg.save_root,
                runner.train_cfg.save_dir,
                runner.train_cfg.save_name,
                'checkpoints',
            )
        chkpt_file = runner.save_checkpoint(self.out_dir, **self.args)
        if is_best and runner.rank == 0 and runner.tensor_parallel_size == 1:
            symlink(chkpt_file, osp.join(self.out_dir, 'best.pth'))
            runner.best_pth = chkpt_file

        if not self.saved_hdfs_files or self.saved_hdfs_files[-1] != chkpt_file:
            self.saved_hdfs_files.append(chkpt_file)
            self.saved_local_files.append(chkpt_file)

        remote_save_root = runner.train_cfg.get('remote_save_root', None)
        if remote_save_root and runner.dp_rank == 0:
            remote_save_dir = osp.join(
                remote_save_root,
                runner.train_cfg.save_dir,
                runner.train_cfg.save_name,
                'checkpoints',
            )
            hdfs_mkdir(remote_save_dir)
            hdfs_file = osp.join(remote_save_dir, chkpt_file)
            local_file = osp.join(self.out_dir, chkpt_file)
            # the total time should less than NCCL default timeout value 1h.
            sync = runner.pipeline_cfg is None and runner.tensor_parallel_size == 1
            hdfs_put(local_file, hdfs_file, sync=sync, retry=1, timeout=100 * 60)  # 100mins
            if sync:
                latest_file = osp.join(remote_save_dir, 'latest.pth')
                hdfs_copy(hdfs_file, latest_file, force=True, retry=1, timeout=5 * 60)
                if is_best:
                    best_file = osp.join(remote_save_dir, 'best.pth')
                    hdfs_copy(hdfs_file, best_file, force=True, retry=1, timeout=5 * 60)
                    runner.best_pth = best_file

        # remove old hdfs checkpoints
        while (
            self.max_hdfs_keep_ckpts > 0 and len(self.saved_hdfs_files) > self.max_hdfs_keep_ckpts
        ):
            file_name = self.saved_hdfs_files.pop(0)
            if remote_save_root and runner.dp_rank == 0:
                hdfs_rm(osp.join(remote_save_dir, file_name))

        # remove old local checkpoints
        while (
            self.max_local_keep_ckpts > 0
            and len(self.saved_local_files) > self.max_local_keep_ckpts
        ):
            file_name = self.saved_local_files.pop(0)
            if runner.dp_rank == 0:
                remove(osp.join(self.out_dir, file_name))
