import argparse
import json
import logging
import multiprocessing
import os
import time
from cProfile import label
from email import header
from multiprocessing import Manager, Pool

import pyarrow
from bytedance import easycycle
from bytedance.easycycle import get_dataset_info
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetWriter

from samantha.dataio.parquet import ParquetWriter
from samantha.dataio.utils import parquet_reader
from scripts.utils.bigspeech import get_partition

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
        writer = ParquetWriter(filename=path, filesystem=fs, verbose=True)
        for item in data:
            writer.write(item)
        writer.close()


def is_float(e):
    try:
        float(e)
        return True
    except Exception as e:
        return False


def read(q, url, PREFIX, fs, partitions, index_version):
    index = []
    path = url.replace(
        f"/index_{index_version}/{partitions[0]}=",
        f"/index_{index_version}_new/{partitions[0]}=",
    ).replace(f".index_{index_version}", "")
    for group_no, item in parquet_reader(url, fs=fs):
        meta = json.loads(item["meta"])
        while not isinstance(meta, dict):
            meta = json.loads(meta)
        labels = meta.get("labels", "")
        if labels:
            labels = labels.split("\n")
            labels = [lab.split("\t") for lab in labels]
            header = labels[1]
            if is_float(header[-1]):
                if len(labels[-1]) == 2:
                    labels[-2][-1] = labels[-1][-1]
                    labels = labels[:-1]
                labels = "\n".join(["\t".join(lab[:-1]) for lab in labels])
                meta["labels"] = labels

        item["meta"] = json.dumps(meta, ensure_ascii=False)
        index.append(item)
    q.put((index, path))


def get_index_version(path, fs):
    version = 1
    for item in fs.listdir(path):
        basename = item["name"].split("/")[-1]
        if "index" in basename:
            version = int(basename.split("_")[-1])
    return version


def main(args):

    ROOT_PATH = args.root_path
    if ROOT_PATH is None:
        ROOT_PATH = get_dataset_info(args.dataset_id)
    filesystem = get_filesystem(ROOT_PATH)
    if filesystem.exists(f"{ROOT_PATH}/index_{args.index_version}_bak"):
        return
    partitions, suffix = get_partition(ROOT_PATH, filesystem)

    url_pattern = f"{ROOT_PATH}/index_{args.index_version}/{'/'.join(['*'] * len(partitions))}/*.{suffix}"
    logger.info(f"{url_pattern=}, {args.index_version=}")
    urls = filesystem.glob(url_pattern)
    logger.info(f"{len(urls)=}")
    r_pool = Pool(args.num_reader)
    w_pool = Pool(args.num_writer)

    q = Manager().Queue(128)
    for url in urls:
        r_pool.apply_async(
            func=read,
            args=(q, url, ROOT_PATH, filesystem, partitions, args.index_version),
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
    # filesystem.mv(
    #     f"{ROOT_PATH}/index_{args.index_version}",
    #     f"{ROOT_PATH}/index_{args.index_version}_bak",
    # )
    # filesystem.mv(
    #     f"{ROOT_PATH}/index_{args.index_version}_new",
    #     f"{ROOT_PATH}/index_{args.index_version}",
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--num_reader", type=int, default=10)
    parser.add_argument("--num_writer", type=int, default=10)
    parser.add_argument("--index_version", type=int, default=1)
    args = parser.parse_args()
    main(args)
