#!/usr/bin/python3
import argparse
import logging
import multiprocessing
import os
import time
import warnings
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
import requests
import torch
import torch.distributed as dist
import tqdm
from bytedance.easycycle import get_dataset_info
from lightning_fabric.utilities.cloud_io import get_filesystem

from samantha.dataio.utils import parquet_reader
from samantha.utils.distributed import is_global_zero
from samantha.utils.watch import elapsed_time
from scripts.data_processing.audio.utils import (
    Consumer,
    download_model,
    feature_name_mapping,
    load_model,
    model_path_patten,
)
from scripts.data_processing.bigtts.gen_duration import get_partition
from scripts.utils.bigspeech import get_dataset_name

warnings.simplefilter(action="ignore", category=FutureWarning)


logger = logging.getLogger(__name__)
multiprocessing.set_start_method("spawn", force=True)


def packing_callback(package_id, dest_dir, status="success"):
    domains = {"US": "bigspeech.byteintl.net", "CN": "bigspeech.bytedance.net"}
    region = os.environ.get("ARNOLD_REGION")
    logger.info(f"{region=}")
    url = f"https://{domains[region]}/platform/api/v3/data/dataset_collection/packed_callback"  # noqa
    data = {"packaging_id": package_id, "status": status, "dest_dir": dest_dir}
    logger.info(f"{data=}")
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise ConnectionError(f"request failed {response=}")
    logger.info(f"callback successed {response=}")


class Worker:
    def __init__(
        self, model_path_pattern, local_rank, feature_type, ckpt_path, **kwargs
    ) -> None:
        self.device = f"cuda:{local_rank}"
        self.local_rank = local_rank
        self.model_path = None
        if model_path_pattern:
            self.model_path = model_path_pattern % local_rank
        elif ckpt_path:
            self.model_path = ckpt_path
        self.data_urls = []
        self.output_urls = []
        self.ckpt_urls = []
        self.feature_type = feature_type
        self.kwargs = kwargs

    def en_task(self, data_url, output_url, ckpt_url):
        self.data_urls.append(data_url)
        self.output_urls.append(output_url)
        self.ckpt_urls.append(ckpt_url)

    def load_model(self):
        return load_model(self.feature_type)(
            model_path=self.model_path, device=self.device
        )


@torch.no_grad()
def run(
    worker: Worker,
    processor_idx,
    sample_rate,
    fs,
    dataset_name,
    feature_type,
    batch_size=128,
    domain="data",
):
    if not worker.data_urls:
        return
    device = worker.device
    torch.cuda.set_device(device)
    local_rank = worker.local_rank
    model = worker.load_model()
    prefix = f"[{local_rank=} {processor_idx=}]"
    logger.info(f"{prefix} {len(worker.data_urls)=}")
    for data_url, output_url, ckpt_url in zip(
        worker.data_urls, worker.output_urls, worker.ckpt_urls
    ):
        logger.info(f"{prefix} {data_url=}, {output_url=}, {ckpt_url=}")

    for idx, (data_url, output_url, ckpt_url) in enumerate(
        tqdm.tqdm(
            zip(worker.data_urls, worker.output_urls, worker.ckpt_urls),
            total=len(worker.data_urls),
            desc=device,
        )
    ):
        prefix_info = f"{prefix} [{idx}/{len(worker.data_urls)}]"
        try:
            consumer = Consumer(
                output_url,
                filesystem=fs,
                feature_type=feature_type,
                target_sample_rate=sample_rate,
                domain=domain,
                **worker.kwargs,
            )
            uttids, batch = [], []
            # duration for audio, character count for text
            total_count = 0
            st = time.perf_counter()
            for data_item in parquet_reader(data_url, fs, need_group_no=False):
                x, count = consumer.preprocess(data_item, device)
                if x is None:
                    continue
                total_count += count
                batch.append(x)
                uttids.append(data_item["uttid"])
                if len(batch) < batch_size:
                    continue

                consumer.consume(uttids, model, batch, device, dataset_name)
                batch, uttids = [], []

            consumer.consume(uttids, model, batch, device, dataset_name)
            consumer.close()
            fs.touch(ckpt_url)
            ed = time.perf_counter()
            rtf = (ed - st) / (1e-5 + total_count)
            logger.info(f"{prefix_info} {rtf=:.4f}")
        except Exception as e:
            logger.warning(f"{prefix_info} Error: {data_url=}", exc_info=e)


def check_url(filesystem, data_url, output_url, ckpt_url, force_update):
    return (
        (force_update or not filesystem.exists(ckpt_url))
        and filesystem.exists(data_url)
    ), (data_url, output_url, ckpt_url)


def generate_work_urls(
    ROOT_PATH, feature_type, feature_version, filesystem, force_update, index_version
):

    num_nodes = int(os.getenv("ARNOLD_WORKER_NUM", 1))
    nproc_per_node = int(os.getenv("ARNOLD_WORKER_GPU", 1))
    global_rank = int(os.getenv("RANK", 0))
    partitions, suffix = get_partition(ROOT_PATH, filesystem)
    domain = "data"
    feature_domain = ""
    if index_version is not None:
        domain = f"index_{index_version}"
        feature_domain = domain

    url_pattern = f"{ROOT_PATH}/{domain}/{'/'.join(['*'] * len(partitions))}/*.{suffix}"
    urls = filesystem.glob(url_pattern)
    thread_pool = ThreadPool(100)
    remain_urls = []
    rets = []
    ARNOLD_BASE_DIR = os.getenv("ARNOLD_BASE_DIR", "")
    for url in urls:
        data_url = os.path.join(ARNOLD_BASE_DIR, url.lstrip("/"))
        output_url = data_url.replace(
            os.path.join(ROOT_PATH, domain),
            os.path.join(
                ROOT_PATH, f"features/{feature_domain}/{feature_type}_{feature_version}"
            ),
        )
        output_bn = os.path.basename(output_url)
        output_dn = os.path.dirname(output_url)
        ckpt_url = f"{output_dn}/SUCCESS/{output_bn}"
        assert data_url != output_url, f"{data_url=} -- {output_url=}"
        rets.append(
            thread_pool.apply_async(
                func=check_url,
                args=(filesystem, data_url, output_url, ckpt_url, force_update),
            )
        )
    thread_pool.close()
    for ret in tqdm.tqdm(rets, desc=f"{global_rank=}"):
        flag, url = ret.get()
        if flag:
            remain_urls.append(url)
        else:
            logger.info(f"skipping file {url[1]}")
    thread_pool.join()

    dist.barrier()
    logger.info(f"[{global_rank=}] {num_nodes=} {nproc_per_node=} reaching the barrier")
    work_urls = np.array_split(remain_urls, num_nodes * nproc_per_node)[global_rank]
    return work_urls


@elapsed_time
def main(args):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    for feature in args.features:
        feature_type, feature_version = (
            "_".join(feature.split("_")[:-1]),
            feature.split("_")[-1],
        )
        for dataset_id in args.dataset_ids:
            logger.info(
                f"[{local_rank=}] processing {dataset_id}-{feature}-{feature_name_mapping(feature_type)}"
            )
            ROOT_PATH = get_dataset_info(dataset_id)
            dataset_name = get_dataset_name(dataset_id)
            filesystem = get_filesystem(ROOT_PATH)

            work_urls = generate_work_urls(
                ROOT_PATH,
                feature_type,
                feature_version,
                filesystem,
                args.force,
                args.index_version,
            )

            n_worker = args.num_processor
            workers = [
                Worker(
                    model_path_pattern=model_path_patten(feature_type, feature_version),
                    local_rank=local_rank,
                    feature_type=feature_type,
                    ckpt_path=args.ckpt_path,
                    trim=not args.not_trim,
                    freq=args.freq,
                )
                for _ in range(n_worker)
            ]

            for i, (data_url, output_url, ckpt_url) in enumerate(work_urls):
                workers[i % n_worker].en_task(data_url, output_url, ckpt_url)

            domain = "data" if args.index_version is None else "index"
            pool = Pool(n_worker)
            for i, worker in enumerate(workers):
                pool.apply_async(
                    func=run,
                    args=(
                        worker,
                        i,
                        args.target_sr,
                        filesystem,
                        dataset_name,
                        feature_type,
                        args.batch_size,
                        domain,
                    ),
                )

            pool.close()
            pool.join()

            logger.info(
                f"[{local_rank=}] processed {dataset_id}-{feature}-{feature_name_mapping(feature_type)}"
            )
            if args.package_id:
                dist.barrier()
                if is_global_zero():
                    feature_domain = ""
                    if args.index_version is not None:
                        domain = f"index_{args.index_version}"
                        feature_domain = domain
                    feature_dst_dir = os.path.join(
                        ROOT_PATH,
                        f"features/{feature_domain}/{feature_type}_{feature_version}",
                    )
                    packing_callback(args.package_id, feature_dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", nargs="+", type=str, required=True)
    parser.add_argument("--features", nargs="+", type=str, default="wavevae_1.0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--package_id", type=str, default=None)
    parser.add_argument("--num_processor", type=int, default=1)
    parser.add_argument("--not_trim", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--index_version", type=int, default=None)
    parser.add_argument("--freq", type=int, default=40)
    args = parser.parse_args()
    backend = "nccl" if torch.cuda.is_available() else "mpi"
    dist.init_process_group(backend=backend)
    args.ckpt_path = download_model(args.ckpt_path)
    main(args)
