""" data utils. """

import os
import subprocess

from bytedance.easycycle import get_dataset_collection_info
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.exceptions import MisconfigurationException
from pyarrow.parquet import ParquetFile

from samantha.dataio import remote_io
from samantha.dataio.webdataset.ra_wds import expand_urls


@remote_io.remote_load(0)
def parse_data_url_fn(fn):
    urls = []
    with open(fn, "r", encoding="utf-8") as fi:
        for line in fi:
            wds_path = line.strip()
            if wds_path == "":
                continue
            urls.append(wds_path)
    return urls


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


def __expand_paths(path_lst):
    if path_lst is None:
        return None
    paths = []
    for lst in path_lst:
        paths.extend(expand_urls(lst))
    return paths


def parse_data_urls(data_id=None, data_urls=None, use_url_lst=False):
    if data_id is not None and data_urls is not None:
        raise MisconfigurationException(
            f"Combination of parameters {data_id=} and {data_urls=} should be mutually "
            f"exclusive."
        )
    if data_id is None and data_urls is None:
        raise MisconfigurationException("User must specify either data_id or data_urls")

    if data_id is not None:
        os.environ["DatasetID"] = str(data_id)
        path_list = get_dataset_collection_info(data_id)
        paths = [v["data"] for v in path_list]
        data_urls = __expand_paths(paths)
        if path_list[0]["index"]:
            idx_paths = [v["index"] for v in path_list]
            idx_urls = __expand_paths(idx_paths)
            return list(zip(data_urls, idx_urls))
        else:
            return data_urls

    if isinstance(data_urls, str):
        if use_url_lst:
            data_urls = parse_data_url_fn(data_urls)
        else:
            data_urls = [data_urls]
    if not isinstance(data_urls, list):
        raise TypeError(
            f"Expecting data_urls either be str or list, but got"
            f" {data_urls=}, {type(data_urls)=}"
        )
    return __expand_paths(data_urls)


def parquet_reader(url, fs=None):
    if fs is None:
        fs = get_filesystem(url)
    parquet_file = ParquetFile(url, filesystem=fs)
    row_group_num = parquet_file.num_row_groups
    row_groups = list(range(row_group_num))
    for group_no, row_group in enumerate(row_groups):
        group_data = parquet_file.read_row_group(row_group)
        group_datas = group_data.to_pandas()
        for row in group_datas.iterrows():
            item = row[1].to_dict()
            yield group_no, item
    parquet_file.close()
