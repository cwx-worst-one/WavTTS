import logging
import os
import subprocess

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def hdfs_ls(hdfs_path: str):
    """
    Returns list of HDFS directory entries (absolute paths).

    Args:
        hdfs_path (str): hdfs directory

    Returns:
       out (list): list of hdfs paths in directory
    """
    cmd = f"hdfs dfs -ls {hdfs_path} | awk '{{print $8}}'"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    (out, err) = proc.communicate()
    try:
        if isinstance(err, bytes):
            err = err.decode()
        if not isinstance(err, str):
            err = str(err)
    except Exception as exn:
        err = ""

    if proc.returncode != 0:
        errmsg = (
            'Failed to list HDFS directory "'
            + hdfs_path
            + '", return code '
            + str(proc.returncode)
        )
        return []
    elif err:
        logger.warn(err)

    out = out.splitlines()
    out = [elem.decode() for elem in out if elem]

    return out


def split(l, chunk_size):
    for i in range(0, len(l), chunk_size):
        yield l[i : i + chunk_size]


def transfer(src: str, target: str):
    p = subprocess.Popen(["hdfs", "dfs", "-get", src, target])
    (out, err) = p.communicate()

    # if out is not None:
    #     logger.warn(out)

    # if err is not None:
    #     logger.warn(err)


if __name__ == "__main__":
    ## Parquet
    # NUM_DRIVES = 5
    # DATA_SHARDS = sorted(hdfs_ls("hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_everynoise_N937k_Lmix_mp3/data"))
    # split_shards = np.array_split(DATA_SHARDS, NUM_DRIVES)

    # for drive_no in range(0, NUM_DRIVES):
    #     target = f"/mnt/bd/everynoise-{drive_no}/data"
    #     os.makedirs(target, exist_ok=True)
    #     shards = split_shards[drive_no]
    #     Parallel(n_jobs=32, prefer="processes")(
    #         delayed(transfer)(fp, target) for fp in tqdm(shards)
    #     )

    ## WebDataset
    DATA_SHARDS = sorted(
        hdfs_ls(
            "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/shards/billboard_hot_200-v2_normalised_-16LUFS/*/*"
        )
    )

    target = "/mnt/bd/billboard-44k/data"
    os.makedirs(target, exist_ok=True)
    Parallel(n_jobs=32, prefer="processes")(
        delayed(transfer)(fp, target) for fp in tqdm(DATA_SHARDS)
    )
