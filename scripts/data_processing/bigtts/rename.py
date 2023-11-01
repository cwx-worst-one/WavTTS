import argparse
import logging
from multiprocessing import Manager

from bytedance.easycycle import get_dataset_info
from lightning_fabric.utilities.cloud_io import get_filesystem

from scripts.utils.bigspeech import get_partition

logger = logging.getLogger(__name__)


def main(args):
    from tqdm import tqdm

    ROOT_PATH = get_dataset_info(args.dataset_id)
    filesystem = get_filesystem(ROOT_PATH)
    partitions, suffix = get_partition(ROOT_PATH, filesystem)

    url_pattern = f"{ROOT_PATH}/index_{args.index_version}/{'/'.join(['*'] * len(partitions))}/*.{suffix}"
    logger.info(f"{url_pattern=}, {args.index_version=}")
    urls = filesystem.glob(url_pattern)
    logger.info(f"{len(urls)=}")

    q = Manager().Queue(128)
    for url in tqdm(urls):
        path = url.replace(f".index_{args.index_version}", "")
        filesystem.mv(url, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--index_version", type=int, default=1)
    args = parser.parse_args()
    main(args)
