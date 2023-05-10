'''
consist of some operations about Arnold log
'''
import os
from core.utils import Config, get_logger, hdfs_mkdir, hdfs_put
from train import parse_args


def upload_logs(cfg):
    '''the Arnold stdout and stderr will be uploaded to train_remote_save_root
    Args:
        the train.remote_save_root in global config
    '''
    trial_id = os.getenv('ARNOLD_TRIAL_ID', None)
    role = os.getenv('ARNOLD_ROLE', None)
    worker_id = os.getenv('DMLC_WORKER_ID', None)
    if trial_id is None or role != 'worker':
        return

    logger = get_logger(log_level='INFO')
    remote_save_root = cfg.train.get('remote_save_root', None)
    if not remote_save_root:
        return
    hdfs_path = os.path.join(
        remote_save_root,
        cfg.train.save_dir,
        cfg.train.save_name,
        'logs',
        trial_id,
    )
    hdfs_mkdir(hdfs_path)

    # sometimes USER_LOGS_DIR is None
    log_dir = os.getenv('USER_LOGS_DIR') or os.getenv('LOG_DIRS')
    if log_dir is None:
        return
    for name in ['stdout', 'stderr']:
        local_file = os.path.join(log_dir, name)
        remote_file = os.path.join(hdfs_path, f'{role}_{worker_id}_{name}')
        hdfs_put(local_file, remote_file, sync=True)
    logger.info('Upload Trial %s stdout and stderr to %s', trial_id, hdfs_path)


def main():
    '''main.'''
    args, unknown = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_list(unknown)
    upload_logs(cfg)


if __name__ == "__main__":
    main()
