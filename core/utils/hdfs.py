'''
Functions for packaging  hdfs command.
'''

import os
from core.utils import logging


def check_hdfs_path(path, func='hdfs_func'):
    '''check whether file path is starts with 'hdfs://'.
    Args:
        @path: file path
        @func: call in func name.
    '''
    if not path.startswith('hdfs://'):
        logging.warning("%s: remote path is not begin with 'hdfs://': %s", func, path)

    # If the domestic machine visit blacklist['CN'] is to report an error,
    # and the foreign machine will report the error when visiting blacklist['UNCN']
    blacklist = {'CN': ['harunava'], 'UNCN': ['haruna']}
    arnold_region = os.getenv('ARNOLD_REGION', 'CN')
    if arnold_region != 'CN':
        arnold_region = 'UNCN'
    hdfs_node = path.split('/')[2]
    if hdfs_node in blacklist[arnold_region]:
        raise IOError(f"{func}: the remote path can't be accessed because not in a region: {path}")


def hdfs_get(remote_file, local_dir='.', sync=True, retry=3):
    '''get remote files on hdfs to local file path.
    Args:
        @remote_file: remote path on hdfs to download.
        @local_dir: local file path to save files.
        @sync: whether do this shell command in background.
        @retry: retry times when fail.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    check_hdfs_path(remote_file, 'hdfs_get')
    cmd = 'hdfs dfs -get {} {}'.format(remote_file, local_dir)
    if not sync:
        cmd += ' &'

    ret = -1
    while ret != 0 and retry > 0:
        ret = os.system(cmd)
        retry -= 1
    if ret != 0:
        logging.warning("%s failed", cmd)
    return ret


def hdfs_put(local_file, remote_dir, sync=False, force=True, retry=30, timeout=5 * 60):
    '''put local files to remote file path on hdfs.
    Args:
        @local_file: local file path to be uploaded.
        @remote_dir: remote path on hdfs.
        @sync: whether do this shell command in background.
        @force: whether user '-f'(force) option.
        @retry: retry times when fail.
        @timeout: timeout in senconds, default is 5min
            It should less than NCCL default timeout value 1h.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    check_hdfs_path(remote_dir, 'hdfs_put')
    cmd = 'hdfs dfs -put '
    if timeout:
        cmd = 'timeout --signal=KILL {}s {}'.format(timeout, cmd)
    if force:
        cmd += '-f '
    cmd += '{} {}'.format(local_file, remote_dir)
    if not sync:
        cmd += ' &'

    ret = -1
    while ret != 0 and retry > 0:
        ret = os.system(cmd)
        retry -= 1
    if ret != 0:
        logging.warning("%s failed", cmd)
    return ret


def hdfs_mkdir(remote_dir, parents=True, retry=3):
    '''make directory on hdfs.
    Args:
        @remote_dir: remote directory path on hdfs
        @parents: whether use parents option, same as mkdir's '-p' option
        @retry: retry times when fail.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    check_hdfs_path(remote_dir, 'hdfs_mkdir')
    cmd = 'hdfs dfs -mkdir '
    if parents:
        cmd += '-p '
    cmd += remote_dir

    ret = -1
    while ret != 0 and retry > 0:
        ret = os.system(cmd)
        retry -= 1
    if ret != 0:
        logging.warning("%s failed", cmd)
    return ret


def hdfs_copy(src_file, dst_file, force=True, sync=True, retry=3, timeout=0):
    '''copy remote src file to remote dst file.
    Args:
        @src_file: remote source file path on hdfs.
        @dst_file: remote destinated file path on hdfs.
        @force: whether user '-f'(force) option.
        @sync: whether do this shell command in background.
        @retry: retry times when fail.
        @timeout: timeout in senconds, default is 0(no timeout).
            It should less than NCCL default timeout value 1h.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    check_hdfs_path(src_file, 'hdfs_copy')
    check_hdfs_path(dst_file, 'hdfs_copy')
    cmd = 'hdfs dfs -cp '
    if timeout:
        cmd = 'timeout --signal=KILL {}s {}'.format(timeout, cmd)
    if force:
        cmd += '-f '
    cmd += src_file + ' ' + dst_file
    if not sync:
        cmd += ' &'

    ret = -1
    while ret != 0 and retry > 0:
        ret = os.system(cmd)
        retry -= 1
    if ret != 0:
        logging.warning("%s failed", cmd)
    return ret


def hdfs_test(remote_path, mode='-e', retry=3):
    '''test hdfs path.
    Args:
        @remote_path: remote path, begin with 'hdfs://'
        @mode: -d  return 0 if @remote_path is a directory.
               -e  return 0 if @remote_path exists.
               -f  return 0 if @remote_path is a file.
               -s  return 0 if file @remote_path is greater than
                   zero bytes in size.
               -z  return 0 if file @remote_path is zero bytes in size,
                   else return 1.

    Return:
        0 if success.
        other if fail.
    '''
    check_hdfs_path(remote_path, 'hdfs_test')
    assert mode in ('-e', '-d', '-f', '-s', '-z')
    cmd = 'hdfs dfs -test {} {}'.format(mode, remote_path)

    ret = -1
    while ret != 0 and retry > 0:
        ret = os.system(cmd)
        retry -= 1
    if ret != 0:
        logging.warning("%s failed", cmd)
    return ret


def hdfs_rm(remote_path, force=True, recursive=True, sync=False, retry=3):
    '''remove path on hdfs.
    Args:
        @remote_path(str): remote file path on hdfs
        @force(bool): force flag for rm.
        @recursive(bool): Recursively deletes directories.
        @sync(bool): whether do this shell command in background.
        @retry(int): retry times when fail.

    Return:
        command line return value.
        0 if success, otherwise failure.
    '''
    check_hdfs_path(remote_path, 'hdfs_rm')
    cmd = 'hdfs dfs -rm '
    if force:
        cmd += '-f '
    if recursive:
        cmd += '-r '
    cmd += remote_path
    if not sync:
        cmd += ' &'

    ret = -1
    while ret != 0 and retry > 0:
        ret = os.system(cmd)
        retry -= 1
    if ret != 0:
        logging.warning("%s failed", cmd)
    return ret


def hdfs_ls(remote_path, ptype='dir'):
    '''get hdfs file list.
    Args:
        @remote_path: remote path, begin with 'hdfs://'
        @ptype: dir  get files list in the dir if @remote_path is a directory.
                file get file path if @remote_path is a file

    Return:
        files list.
    '''
    items = []
    check_hdfs_path(remote_path, hdfs_ls)
    assert ptype in ('dir', 'file')
    p = os.popen("hdfs dfs -ls %s" % remote_path)
    for line in p.read().splitlines():
        s = line.strip().split()
        if len(s) > 0 and s[-1].startswith('hdfs'):
            items.append(s[-1])
    return items
