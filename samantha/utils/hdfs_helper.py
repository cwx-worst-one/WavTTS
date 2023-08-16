r"""This module provides awesome HDFS relevant utilities."""

import contextlib
import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

import command

logger = logging.getLogger(__name__)

HDFS = "hdfs dfs"
FAILED_TO_LIST_DIRECTORY_MSG = "No such file or directory"


class HdfsException(Exception):
    pass


def _run_command(cmd):
    try:
        ret = command.run(cmd.split())
        return ret
    except command.CommandException as e:
        return e


def ishdfs(path: Union[str, Path]):
    if isinstance(path, Path):
        path = path.as_posix()
    return path.startswith("hdfs:")


@contextlib.contextmanager
def hopen(path: str, mode="r"):  # pragma: no cover
    """Open a file from local/hdfs.

    Args:
        path (str): Path of the file, can be either a local file or a hdfs file.
        mode (str): Open mode.
    """
    if ishdfs(path):
        cmd = f"{HDFS} -text {path}"
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()  # type: ignore
        pipe.kill()
        pipe.wait()
        return
    else:
        yield open(path, mode)


def append(msg, hdfs_path):
    """
    append is diff from other methods. It involves `echo` thus
    needs to set `shell=True`
    """
    cmd = f"echo {msg} | {HDFS} -appendToFile - {hdfs_path}"
    res = subprocess.call(cmd, shell=True)
    return res == 0


def mkdir(path: str):
    """Create a directory on local or hdfs.

    Args:
        path (str): directory to be created on local or hdfs.
    """

    if ishdfs(path):
        cmd = f"{HDFS} -mkdir -p {path}"
        res = _run_command(cmd)
        result = res.exit == 0
    else:
        Path(path).mkdir(parents=True, exist_ok=True)
        result = True
    return result


def touch(path: str):
    r"""Touch a file.

    Args:
        path (str): local or hdfs path.
    """
    if isfile(path):
        cmd = f"{HDFS} -touchz {path}"
        res = _run_command(cmd)
        result = res.exit == 0
    else:
        Path(path).touch()
        result = True
    return result


def isfile(hdfs_path: str) -> bool:
    """Check if a hdfs path is a file.

    Args:
        hdfs_path (str): Path to be checked.
    """

    if ishdfs(hdfs_path):
        cmd = f"{HDFS} -test -f {hdfs_path}"
        res = _run_command(cmd)
        return res.exit == 0
    return os.path.isfile(hdfs_path)


def isdir(hdfs_path: str) -> bool:
    """Check if a hdfs path is a directory.

    Args:
        hdfs_path (str): Path to be checked.
    """

    cmd = f"{HDFS} -test -d {hdfs_path}"
    res = _run_command(cmd)
    return res.exit == 0


def rm(hdfs_path: str) -> bool:
    """Remove a file on hdfs.

    Args:
        hdfs_path (str): The file to be removed.
    """

    cmd = f"{HDFS} -rm -f {hdfs_path}"
    res = _run_command(cmd)
    return res.exit == 0


def rmdir(path: str) -> bool:
    """Remove a directory on hdfs.

    Args:
        path (str): The directory to be removed.
    """

    if ishdfs(path):
        cmd = f"{HDFS} -rm -r -f {path}"
        res = _run_command(cmd)
        result = res.exit == 0
    else:
        shutil.rmtree(path)
        result = True
    return result


def put(local_path: str, hdfs_path: str) -> bool:
    """Upload a local file to hdfs.

    Args:
        local_path (str): The local file to be uploaded.
        hdfs_path (str): A file path on hdfs, will be overwritten if already exist.
    """

    if isfile(hdfs_path):
        logger.warning(f"{hdfs_path} has already exist and will be overwritten.")
        rm(hdfs_path)
    mkdir(os.path.dirname(hdfs_path))
    cmd = f"{HDFS} -put {local_path} {hdfs_path}"
    res = _run_command(cmd)
    if res.exit != 0:
        logger.warning(
            f"Failed to put {local_path} to {hdfs_path} with message {res.message}"
        )
    return res.exit == 0


def get(hdfs_path: str, local_path: str) -> bool:
    """Download a hdfs file to local.

    Args:
        hdfs_path (str): A file path on hdfs to be downloaded.
        local_path (str): The targeted local directory.
    """

    if not isfile(hdfs_path):
        raise ValueError(f"{hdfs_path} does not exist.")
    cmd = f"{HDFS} -get {hdfs_path} {local_path}"
    res = _run_command(cmd)
    return res.exit == 0


def put_many(local_paths: List[str], hdfs_path: str) -> bool:
    """Upload local files to hdfs.

    Args:
        local_paths (List[str]): The local files to be uploaded.
        hdfs_path (str): A directory on hdfs.
    """

    if not isdir(hdfs_path):
        raise ValueError(f"{hdfs_path} is not a directory.")
    for p in local_paths:
        bp = os.path.basename(p)
        hf = f"{hdfs_path}/{bp}"
        if isfile(hf):
            logger.warning(f"{hf} has already exist and will be overwritten.")
            rm(hf)
    cmd = f"{HDFS} -put {' '.join(local_paths)} {hdfs_path}"
    res = _run_command(cmd)
    return res.exit == 0


def sync_hdfs_dir(src_dir, dst_dir, retry_times=10, multi_processing=True):
    r"""This function will sync stuff to hdfs, the default destination
    is `ARNOLD_OUTPUT` which set by ARNOLD trial.

    Args:
        src_dir (str): local directory
        dst_dir (str): destination hdfs folder
        retry_times (int): number of retry time if sync failed.
        multi_processing (bool): sync files with multi-processes or not.

    Returns:
       List: destination hdfs paths
    """

    if multi_processing:
        from multiprocessing import Pool

        hdfs_files = []
        worker_pool = Pool(5)
        worker_ret = []

        def inner_sync(src_dir, dst_dir, retry_times):
            for fn in os.listdir(src_dir):
                local_file = os.path.join(src_dir, fn)
                if os.path.isdir(local_file):
                    nested_dst_dir = os.path.join(dst_dir, fn)
                    mkdir(nested_dst_dir)
                    inner_sync(local_file, nested_dst_dir, retry_times)
                else:
                    worker_ret.append(
                        worker_pool.apply_async(
                            func=sync_hdfs, args=(local_file, dst_dir, retry_times)
                        )
                    )

        inner_sync(src_dir, dst_dir, retry_times)
        for ret in worker_ret:
            hdfs_files.append(ret.get())
    else:
        hdfs_files = []
        for fn in os.listdir(src_dir):
            local_file = os.path.join(src_dir, fn)
            hdfs_files.append(sync_hdfs(local_file, dst_dir, retry_times))
    return hdfs_files


def sync_hdfs(src, dst_dir=None, retry_times=10):
    r"""This function will sync stuff to hdfs, the default destination
    is `ARNOLD_OUTPUT` which set by ARNOLD trial.

    Args:
        src (str): local file which will be synced.
        dst_dir (str): destination hdfs folder.
        retry_times (int): number of retry time if sync failed.

    Returns:
       destination hdfs path.
    """
    if dst_dir is None:
        dst_dir = os.getenv("ARNOLD_OUTPUT")

    if dst_dir is None:
        logger.warning("No dst_dir specified.")
        return None
    logger.info(f"Syncing output from {src} to {dst_dir}")
    basename = os.path.basename(src)
    hdfs_path = os.path.join(dst_dir, basename)
    rt = retry_times
    while rt > 0:
        if put(src, hdfs_path):
            return hdfs_path
        rt -= 1
    else:
        logger.warning(
            f"Failed to put file to {hdfs_path} after retrying {retry_times}"
            f" times, but you can still get the file at {src}."
        )
        return None


def hdfs_ls(hdfs_path: str):
    """
    Returns list of HDFS directory entries (absolute paths).

    Args:
        hdfs_path (str): hdfs directory

    Returns:
       out (list): list of hdfs paths in directory
    """
    logger.info("Listing HDFS directory " + hdfs_path)
    cmd = f"hdfs dfs -ls {hdfs_path} | awk '{{print $8}}'"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    (out, err) = proc.communicate()
    err = err.decode()

    if proc.returncode != 0:
        errmsg = (
            'Failed to list HDFS directory "'
            + hdfs_path
            + '", return code '
            + str(proc.returncode)
        )
        logger.error(errmsg)
        logger.error(err)
        if FAILED_TO_LIST_DIRECTORY_MSG not in err:
            raise HdfsException(errmsg)
        return []
    elif err:
        logger.debug("stderr:\n" + err.decode())

    out = out.splitlines()
    out = [elem.decode() for elem in out if elem]

    return out


def hdfs_getsize(hdfs_path: str):
    """
    Returns size of HDFS file (absolute paths).

    Args:
        hdfs_path (str): hdfs filepath

    Returns:
       out (int): number of bytes
    """
    logger.info("HDFS filesize " + hdfs_path)
    cmd = f"hdfs dfs -du -s {hdfs_path} | awk '{{print $1}}'"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    (out, err) = proc.communicate()

    if proc.returncode != 0:
        errmsg = (
            'Failed to get HDFS size "'
            + hdfs_path
            + '", return code '
            + str(proc.returncode)
        )
        logger.error(errmsg)
        logger.error(err)
        if FAILED_TO_LIST_DIRECTORY_MSG not in err:
            raise HdfsException(errmsg)
        return []
    elif err:
        logger.debug("stderr:\n" + err)

    out = out.decode().strip()
    return int(out)


def list_dir(path: str):
    r"""List files under path.

    Args:
        path (str): local or hdfs path.

    Returns:
        List: filenames under path
    """
    if ishdfs(path):
        return hdfs_ls(path)
    return os.listdir(path)


def glob_files(pattern: str):
    r"""List all files which names match the pattern

    Args:
        pattern: file name pattern

    Returns:
        List: file paths match the pattern
    """
    if ishdfs(pattern):
        return hdfs_ls(pattern)
    return glob.glob(pattern)


def walk_one(path: str):
    r"""List level one directory under path.

    Args:
        path (str): local or hdfs path.
    """
    if ishdfs(path):
        paths = []
        for p in hdfs_ls(path):
            if isdir(p):
                paths.append(p)
        return paths
    return next(os.walk(path))[1]


def exists(path):
    if not ishdfs(path):
        return os.path.exists(path)
    cmd = f"{HDFS} -test -e {path}"
    return _run_command(cmd).exit == 0


class HdfsFile:
    """A wrapper of HDFS file and make it open like local file.

    Args:
        hdfs_path (str): path to hdfs
    """

    def __init__(self, hdfs_path):
        self.path = hdfs_path

    def write(self, msg):
        r"""Append message to an opened hdfs file handler"""
        if not append(msg, self.path):
            logger.warning(f"Failed to write msg to {self.path}")
