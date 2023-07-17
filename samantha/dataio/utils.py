""" data utils. """

import os
import subprocess


def run_command(cmd):
    """run command, get output"""
    rets = None
    with subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    ) as p:
        rets = p.communicate()[0]
    return rets


def list_tensorbundle_files(cmd, cur_pattern):
    """list tensorbundle files"""
    cmd = f"{cmd} {cur_pattern}*.index"
    text_wrapper = run_command(cmd)
    ret_file_list = []
    for elem in text_wrapper.strip().split():
        if elem.startswith("hdfs://") or elem.startswith("/mnt"):
            ret_file_list.append(elem[:-6])
    return ret_file_list


def list_files(cmd, cur_pattern):
    """list files"""
    cmd = f"{cmd} {cur_pattern}"
    text_wrapper = run_command(cmd)
    ret_file_list = []
    for elem in text_wrapper.strip().split():
        if elem.startswith("hdfs://") or elem.startswith("/mnt"):
            ret_file_list.append(elem)
    return ret_file_list


# pylint: disable="redefined-outer-name"
def file_pattern(data_root, file_pattern, file_num):
    """deal file pattern"""
    if file_num is None:
        # Situation-1: {data_root}/{file_pattern}
        if "*" not in file_pattern:
            return [os.path.join(data_root, file_pattern)]
        # Situation-2: {data_root}/{file_pattern}*
        if data_root.startswith("hdfs://"):  # hdfs_files
            cmd = "hdfs dfs -ls"
        elif data_root.startswith("/mnt"):
            cmd = "ls"
        cur_pattern = os.path.join(data_root, file_pattern)
        file_list = list_tensorbundle_files(cmd, cur_pattern)
        if len(file_list) <= 0:
            file_list = list_files(cmd, cur_pattern)
        return file_list

    # Situation-3: {data_root}/{file_pattern}{idx}
    file_list = []
    for file_idx in range(file_num):
        file_name = os.path.join(data_root, file_pattern + str(file_idx))
        file_list.append(file_name)
    return file_list


def sort_data_sources(file_list):
    """sort data sources"""
    parquet_files = []
    kv_files = []
    wds_files = []
    for path in file_list:
        if ".parquet" in path:
            parquet_files.append(path)
        elif ".tar" in path:
            wds_files.append(path)
        else:
            kv_files.append(path)
    data_sources = []
    source_types = []
    if kv_files:
        data_sources.append(kv_files)
        source_types.append("falcon")
    if parquet_files:
        data_sources.append(parquet_files)
        source_types.append("parquet")
    if wds_files:
        data_sources.append(wds_files)
        source_types.append("webdataset")
    return data_sources, source_types
