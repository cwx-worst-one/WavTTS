''' logging hook. '''
import os

from core.utils import get_rank, hdfs_mkdir, hdfs_put, logging
from .hook import HOOKS, Hook


@HOOKS.register_module()
class LoggingHook(Hook):
    '''LoggingHook.'''

    def __init__(self):
        '''init.'''

    def after_val_epoch(self, runner):
        '''after_val_epoch.'''
        if get_rank() != 0:
            return
        trial_id = os.getenv('ARNOLD_TRIAL_ID', None)
        role = os.getenv('ARNOLD_ROLE', 'worker')
        if trial_id is None or role != 'worker':
            return

        remote_save_root = runner.train_cfg.get('remote_save_root', None)
        if not remote_save_root:
            return
        hdfs_path = os.path.join(
            remote_save_root,
            runner.train_cfg.save_dir,
            runner.train_cfg.save_name,
            'logs',
            trial_id,
        )
        hdfs_mkdir(hdfs_path)

        log_dir = os.getenv('USER_LOGS_DIR') or os.getenv('LOG_DIRS')
        if log_dir is None:
            return
        for name in ['stdout', 'stderr']:
            local_file = os.path.join(log_dir, name)
            remote_file = os.path.join(hdfs_path, f'{role}_0_{name}')
            hdfs_put(local_file, remote_file, sync=True)
        logging.info('Upload Trial %s stdout and stderr to %s', trial_id, hdfs_path)
