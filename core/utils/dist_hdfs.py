'''dits hdfs'''
import os
import os.path as osp
from .hdfs import hdfs_get, hdfs_put
from .path import mkdir_or_exist
from .dist_util import get_local_rank, dist_barrier
from . import logging


def dist_hdfs_get(remote_file, local_dir='./', local_file='.', sync=True, retry=3):
    '''get remote files on hdfs to local file path.
    Args:
        @remote_file: remote path on hdfs to download.
        @local_dir: local file path to save files.
        @local_file: local file name to be saved
        @sync: whether do this shell command in background.
        @retry: retry times when fail.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    local_rank = get_local_rank()
    local_path = osp.join(local_dir, local_file)
    if local_rank == 0:
        mkdir_or_exist(local_dir)
        if local_file != '.':
            os.system('rm -f {}'.format(local_path))
        hdfs_get(remote_file, local_path, sync, retry)
    dist_barrier()
    if osp.exists(local_path):
        # NOTE: this is a temporary solution to avoid output to stderr
        logging.info("get %s from %s", local_path, remote_file)
        return local_path
    return None


def dist_hdfs_put(local_file, remote_dir, sync=False, force=True, retry=3):
    '''put local files to remote file path on hdfs.
    Args:
        @local_file: local file path to be uploaded.
        @remote_dir: remote path on hdfs.
        @sync: whether do this shell command in background.
        @force: whether user '-f'(force) option.
        @retry: retry times when fail.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    local_rank = get_local_rank()
    ret = -1
    if local_rank == 0:
        try:
            ret = hdfs_put(local_file, remote_dir, sync, force, retry)
        except Exception:
            return ret
    dist_barrier()
    if ret != -1:
        logging.info("put %s in %s", local_file, remote_dir)
    return ret


def dist_file_get(remote_file, local_dir='./', local_file='.', sync=True, retry=3):
    '''get files to local file path.
    Args:
        @remote_file: remote path on hdfs or drive or nas to download.
        @local_dir: local file path to save files.
        @local_file: local file name to be saved
        @sync: whether do this shell command in background.
        @retry: retry times when fail.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    if remote_file.startswith("hdfs://"):
        return dist_hdfs_get(remote_file, local_dir, local_file, sync, retry)
    # drive or nas
    local_rank = get_local_rank()
    local_path = osp.join(local_dir, local_file)
    if local_rank == 0:
        mkdir_or_exist(local_dir)
        if local_file != '.':
            os.system('rm -f {}'.format(local_path))
        os.system(f'cp {remote_file} {local_path}')
    dist_barrier()
    if osp.exists(local_path):
        # NOTE: this is a temporary solution to avoid output to stderr
        logging.info("get %s from %s", local_path, remote_file)
        return local_path
    return None
