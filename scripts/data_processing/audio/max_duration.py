import argparse
import json
import logging
import warnings

import tqdm
import wandb
from bytedance.easycycle import get_dataset_info
from lightning_fabric.utilities.cloud_io import get_filesystem

from samantha.dataio.utils import parquet_reader
from scripts.data_processing.bigtts.gen_duration import get_partition
from scripts.utils.bigspeech import find_all_parts, get_dataset_name

warnings.simplefilter(action="ignore", category=FutureWarning)


logger = logging.getLogger(__name__)


def main(args):
    dataset_id = args.dataset_id
    ROOT_PATH = get_dataset_info(dataset_id)
    filesystem = get_filesystem(ROOT_PATH)
    partitions, suffix = get_partition(ROOT_PATH, filesystem)
    url_pattern = f"{ROOT_PATH}/data/{'/'.join(['*'] * len(partitions))}/*.{suffix}"
    urls = filesystem.glob(url_pattern)
    index = f"index_{args.index_version}"
    parts = args.parts
    if parts is None:
        parts = find_all_parts(f"{ROOT_PATH}/data", filesystem=filesystem)
    max_duration, total_duration = -1, 0
    ds_name = get_dataset_name(dataset_id)
    run = wandb.init(project="colin-duration", name=f"{dataset_id}-{ds_name}")
    durations = []
    for url in tqdm.tqdm(urls, desc=ds_name):
        for part in parts:
            if part in url:
                url = "hdfs://haruna" + url
                index_url = url.replace(
                    f"{ROOT_PATH}/data/", f"{ROOT_PATH}/{index}/"
                ).replace(".parquet", f".{index}.parquet")
                cur_max_duration = 0
                for item in parquet_reader(index_url, filesystem, need_group_no=False):
                    duration = json.loads(item["meta"])["duration"]
                    total_duration += duration
                    cur_max_duration = max(cur_max_duration, duration)
                    max_duration = max(max_duration, duration)
                    durations.append(duration)
                total_duration_h = total_duration / 3600
                logger.info(
                    f"{cur_max_duration=}, {max_duration=}, {total_duration_h=:.4f}"
                )
    run.log({"durations": wandb.Histogram(durations)})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_version", type=int, default=1)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--parts", type=str, nargs="+", default=None, required=False)
    args = parser.parse_args()
    logger.info(f"launch with {args=}")
    main(args)
