import argparse
import json
import logging
import multiprocessing
import os
from multiprocessing import Manager, Pool

import pyarrow
from bytedance import easycycle
from bytedance.easycycle import get_dataset_info
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetWriter

from samantha.dataio.utils import parquet_reader

logger = logging.getLogger(__name__)
# for randomizing manager server port
multiprocessing.util.abstract_sockets_supported = False
# for multiple filesystem instance
multiprocessing.set_start_method("spawn", force=True)


def index_parquet_writer(q, fs):
    while True:
        item = q.get()
        if item is None:
            break
        data, path = item
        dir = os.path.dirname(path)
        fs.makedirs(dir)
        scheme = pyarrow.Table.from_pylist(data).schema
        tb = pyarrow.Table.from_pylist(data)
        writer = ParquetWriter(path, scheme, filesystem=fs)
        writer.write_table(tb)
        writer.close()
        logger.info(f"Done {path=}")


def read(q, url, PREFIX, fs, partitions, index_version):
    index = []
    path = url.replace(
        f"/data/{partitions[0]}=", f"/index_{index_version}/{partitions[0]}="
    ).replace(".parquet", f".index_{index_version}.parquet")
    for group_no, item in parquet_reader(url, fs=fs):
        duration = len(item["audio"]) / 2 / 24000
        meta = json.loads(item["meta"])
        meta.update({"duration": duration})
        meta_item = {
            "uttid": item["uttid"],
            "text": item["text"],
            "meta": json.dumps(meta),
            "row_group_no": group_no,
            "data_file": url.replace(PREFIX, "/".join([".."] * (1 + len(partitions)))),
        }
        index.append(meta_item)
    q.put((index, path))


def get_index_version(path, fs):
    version = 1
    for item in fs.listdir(path):
        basename = item["name"].split("/")[-1]
        if "index" in basename:
            version = int(basename.split("_")[-1])
    return version


def get_partition(path, fs):
    partitions, suffix = [], None
    while path is not None:
        for item in fs.listdir(path):
            if item["type"] == "directory":
                basename = item["name"].split("/")[-1]
                if "=" in basename:
                    partitions.append(basename.split("=")[0])
                path = item["name"]
                break
            else:
                path = None
                suffix = item["name"].split(".")[-1]
                break
    return partitions, suffix


def main(args):

    ROOT_PATH = get_dataset_info(args.dataset_id)
    filesystem = get_filesystem(ROOT_PATH)
    partitions, suffix = get_partition(ROOT_PATH, filesystem)
    new_version = get_index_version(ROOT_PATH, filesystem) + 1

    url_pattern = f"{ROOT_PATH}/data/{'/'.join(['*'] * len(partitions))}/*.{suffix}"
    logger.info(f"{url_pattern=}, {new_version=}")
    urls = filesystem.glob(url_pattern)
    r_pool = Pool(args.num_reader)
    w_pool = Pool(args.num_writer)

    q = Manager().Queue(10240)
    for url in urls:
        r_pool.apply_async(
            func=read,
            args=(q, url, ROOT_PATH, filesystem, partitions, new_version),
            error_callback=lambda x: logger.error(x),
        )
    for _ in range(args.num_writer):
        w_pool.apply_async(
            func=index_parquet_writer,
            args=(q, filesystem),
            error_callback=lambda x: logger.error(x),
        )

    r_pool.close()
    r_pool.join()
    for _ in range(args.num_writer):
        q.put(None)

    w_pool.close()
    w_pool.join()

    easycycle.register_dataset_version(
        dataset_id=args.dataset_id,
        version_num=new_version,
        creator_username="wangxin.colin",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--num_reader", type=int, default=10)
    parser.add_argument("--num_writer", type=int, default=10)
    args = parser.parse_args()
    main(args)
