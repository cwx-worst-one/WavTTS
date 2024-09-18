""" data utils. """

import json
import logging
import os
import re
import subprocess
import warnings
from collections import Counter
from multiprocessing.pool import ThreadPool

import braceexpand
import numpy as np
from bytedance.easycycle import get_dataset_collection_info
from tqdm import tqdm
from webdataset.shardlists import split_by_node
from webdataset.utils import pytorch_worker_info

try:
    from bytedance.easycycle import get_dataset_collection_info_v2
except Exception:
    warnings.warn(
        "Could not import get_dataset_collection_info_v2 from bytedance.easycycle, please upgrade your package."
    )
    get_dataset_collection_info_v2 = None

try:
    from bytedance.easycycle import get_dataset_collection_info_v3_with_idc
except Exception:
    warnings.warn(
        "Could not import get_dataset_collection_info_v3_with_idc from bytedance.easycycle,"
        "please upgrade your package."
    )
    get_dataset_collection_info_v3_with_idc = None

from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.exceptions import MisconfigurationException
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from samantha.dataio import remote_io
from samantha.utils.hdfs_helper import fast_glob_files

logger = logging.getLogger(__name__)


def expand_urls(urls):
    if isinstance(urls, str):
        if "*" in urls:
            return fast_glob_files(urls)
        else:
            urllist = urls.split("::")
            result = []
            for url in urllist:
                result.extend(braceexpand.braceexpand(url))
            return result
    else:
        return list(urls)


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
    paths, futures = [], []
    freq = Counter(path_lst)
    pool = ThreadPool(20)
    for lst in freq:
        futures.append(pool.apply_async(func=expand_urls, args=(lst,)))
    pool.close()
    pool.join()
    for future, lst in zip(futures, freq):
        paths.extend(future.get() * freq[lst])
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
        paths = [v["data"].replace("\n", " ") for v in path_list]
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


def parquet_reader(
    url, fs=None, columns=None, sample_limit=None, meta=None, need_group_no=True
):
    if fs is None:
        fs = get_filesystem(url)

    stream = fs.open(url, skip_instance_cache=True)
    parquet_file = ParquetFile(stream)

    # meta: {url: [num_row_group, [unvisit_row_group_list]]}
    # could be empty at beginning
    if meta is None:
        meta = {}

    # cache num_row_group to save io
    if url not in meta:
        meta_data = parquet_file.metadata
        meta[url] = [(meta_data.num_row_groups, meta_data.num_rows), []]
    num_row_groups = meta[url][0][0]

    if sample_limit is None:
        # for row_group in tqdm(range(num_row_groups), desc=url):
        for row_group in range(num_row_groups):
            group_data = parquet_file.read_row_group(row_group, columns=columns)
            group_datas = group_data.to_pandas()
            for row in group_datas.iterrows():
                item = row[1].to_dict()
                if need_group_no:
                    yield row_group, item
                else:
                    yield item
    else:
        if 0 < sample_limit <= 1:
            sample_limit = int(meta[url][0][1] * sample_limit)
        while sample_limit > 0:
            if not meta[url][1]:
                meta[url][1] = list(range(num_row_groups))
            row_group = np.random.choice(meta[url][1])
            meta[url][1].remove(row_group)

            group_data = parquet_file.read_row_group(row_group, columns=columns)
            group_datas = group_data.to_pandas()
            for row in group_datas.iterrows():
                item = row[1].to_dict()
                sample_limit -= 1
                if sample_limit <= 0:
                    break
                if need_group_no:
                    yield row_group, item
                else:
                    yield item
    parquet_file.close()
    stream.close()


def resolve_data_urls(data_id=None, data_urls=None):
    r"""Glob input urls (list of dict), each key indicates one feature or index,
    and produce corresponding samples (list of dict), each dict represent one
    shard parquet.

    .. example::

        >>> data_urls = [{"index": "hdfs://index_1/*.parquet", "data": "hdfs://data/*.parquet", "feat1": "hdfs://feat1/*.parquet", ...}, ...]  # noqa
        >>> output = resolve_data_urls(data_urls=data_urls)
        >>> output
        >>> [
        >>>   {"index": "hdfs://index_1/shard_0.parquet", "data": "hdfs://data/shard_0.parquet", "feat1": "hdfs://feat1/shard_0.parquet"}, # noqa
        >>>   {"index": "hdfs://index_1/shard_1.parquet", "data": "hdfs://data/shard_1.parquet", "feat1": "hdfs://feat1/shard_1.parquet"}, # noqa
        >>>   ...
        >>> ]


    Args:
        urls(List[Dict[str, str]]): input urls

    Returns:
        List[Dict[str, str]]

    """

    if data_id is not None and data_urls is not None:
        raise MisconfigurationException(
            f"Combination of parameters {data_id=} and {data_urls=} should be mutually "
            f"exclusive."
        )
    if data_id is None and data_urls is None:
        raise MisconfigurationException("User must specify either data_id or data_urls")

    if data_id is not None:
        os.environ["DatasetID"] = str(data_id)

        if get_dataset_collection_info_v3_with_idc is not None:
            data_id = str(data_id)
            if ":" in data_id:
                logger.info(f"treat dataset id as idc format, {data_id=}")
            data_urls = get_dataset_collection_info_v3_with_idc(data_id)["origin"][
                "paths"
            ]
        elif get_dataset_collection_info_v2 is not None:
            data_urls = get_dataset_collection_info_v2(data_id)["origin"]["paths"]
        else:
            data_urls = get_dataset_collection_info(data_id)

    columns = None
    for url in data_urls:
        if columns is None:
            columns = set(url.keys())
        else:
            columns.intersection_update(url.keys())

    if "repetitions" in columns:
        columns.remove("repetitions")

    if "index" not in columns:
        raise ValueError(
            f"Data must contain key 'index', but only got {columns=},"
            f" all data_urls: {json.dumps(data_urls, indent=2)}"
        )

    # recode data repeat time, avoiding glob files repetitive
    frequency, uniqed_data_urls = uniq_data_urls(data_urls)

    tpool = ThreadPool(20)
    rets = []

    for url in uniqed_data_urls:
        rets.append(
            tpool.apply_async(
                func=_resolve_one_url,
                args=(url, columns),
                error_callback=lambda exc: logger.error("error on parse", exc_info=exc),
            )
        )
    tpool.close()

    resolved_url_dict = {}
    for ret in tqdm(rets, desc="parse urls"):
        index, cur_data, utterances = ret.get()
        # cur_data: Dict[col_name, Dict[uttid, file]]
        for name, file_dict in cur_data.items():
            drop_utt = set(file_dict.keys()).difference(utterances)
            for du in drop_utt:
                del file_dict[du]

            cur_files = list(file_dict.values()) * frequency[index]
            if name not in resolved_url_dict:
                resolved_url_dict[name] = cur_files
            else:
                resolved_url_dict[name].extend(cur_files)
    tpool.join()

    return [
        dict(zip(columns, item))
        for item in zip(*[resolved_url_dict[k] for k in columns])
    ]


def uniq_data_urls(data_urls):
    if get_dataset_collection_info_v2 is not None:
        frequency = {url["index"]: int(url.pop("repetitions", 1)) for url in data_urls}
        return frequency, data_urls
    else:
        frequency = Counter(url["index"] for url in data_urls)
        uniqed_data_urls, memory = [], set()
        for url in data_urls:
            index = url["index"]
            if index in memory:
                continue
            uniqed_data_urls.append(url)
            memory.add(index)
        return frequency, uniqed_data_urls


def _resolve_one_url(url, columns):
    index = url["index"]

    index_version = re.findall(r".*(index_\d+).*", index)[0]
    ARNOLD_BASE_DIR = os.getenv("ARNOLD_BASE_DIR", "")
    if not ARNOLD_BASE_DIR.startswith("hdfs://"):
        # maybe on merlin devbox, use RUNTIME_IDC_NAME instead
        RUNTIME_IDC_NAME = os.getenv("RUNTIME_IDC_NAME", "")
        if RUNTIME_IDC_NAME == "maliva":
            ARNOLD_BASE_DIR = "hdfs://harunava"
        else:
            ARNOLD_BASE_DIR = "hdfs://haruna"

    # record unique common utterance
    utterances, cur_data = None, {}
    for name, pattern in url.items():
        if name not in columns:
            logger.warning(f"drop feature={name}, cause some datasets do not have it")
            continue
        prefix = re.split(
            r"\*", pattern.removeprefix(ARNOLD_BASE_DIR).replace("//", "/")
        )[0]
        prefix = f"{ARNOLD_BASE_DIR}{prefix}"
        files = {
            ele.removeprefix(prefix).replace(f".{index_version}", ""): ele
            for ele in fast_glob_files(pattern)
        }
        cur_data[name] = files

        if utterances is None:
            utterances = set(files.keys())
        else:
            utterances.intersection_update(files.keys())
    return index, cur_data, utterances


def resolve_data_sources(data_id=None, data_urls=None):
    data_path = parse_data_urls(data_id, data_urls)
    return sort_data_sources(data_path)


def split_urls_by_nodes(urls):
    logger.info(f"Splitting {len(urls)} urls by node...")
    # split the url's by nodes, so that duplicate shards cannot be sampled!
    # 1. sort the url's, so that each node receives the same list:

    if type(urls[0]) is dict:
        urls = sorted(urls, key=lambda u: u["data"])
    else:
        urls = sorted(urls)

    # 2. split url's by node:
    node_urls = list(split_by_node(urls, group=None))

    # 3. logging:
    rank, world_size, worker, num_workers = pytorch_worker_info(group=None)

    if type(urls[0]) is dict:
        preview_urls_first = "\n".join([u["data"] for u in node_urls[:10]])
        preview_urls_last = "\n".join([u["data"] for u in node_urls[-10:]])
    else:
        preview_urls_first = "\n".join(node_urls[:10])
        preview_urls_last = "\n".join(node_urls[-10:])

    logger.info(
        f"{rank=} {world_size=} considering node urls {len(node_urls)} of total {len(urls)}"
    )
    logger.info(
        f"{rank=} {world_size=} Preview of first 10 urls:\n{preview_urls_first}\n"
    )
    logger.info(
        f"{rank=} {world_size=} Preview of last 10 urls:\n{preview_urls_last}\n"
    )
    return node_urls
