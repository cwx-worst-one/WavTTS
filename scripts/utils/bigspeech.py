import argparse
import logging
import os

import requests
from bytedance import easycycle
from lightning_fabric.utilities.cloud_io import get_filesystem

logger = logging.getLogger(__name__)


def find_all_parts(root_path, filesystem=None):
    if filesystem is None:
        filesystem = get_filesystem(root_path)
    _, parts, _ = next(filesystem.walk(root_path, maxdepth=1))
    return parts


def get_dataset_name(dataset_id):
    return easycycle.get_dataset_detail(dataset_id)["dataset"]["name"]


def get_partition(path, fs):
    partitions, suffix = [], None
    path = f"{path}/data"
    while path is not None:
        for item in fs.listdir(path):
            if item["type"] == "directory":
                basename = item["name"].split("/")[-1]
                if "=" in basename:
                    partitions.append(basename.split("=")[0])
                path = item["name"]
                break
            elif not item["name"].endswith("SUCCESS"):
                path = None
                suffix = item["name"].split(".")[-1]
                break
    return partitions, suffix


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--feature", type=str, default="wavevae_1.0")
    parser.add_argument("--package_id", type=str, default=None)
    parser.add_argument("--index_version", type=int, default=None)
    args = parser.parse_args()

    logger.info(f"Launch with {args=}")
    feature_type, feature_version = (
        "_".join(args.feature.split("_")[:-1]),
        args.feature.split("_")[-1],
    )
    ROOT_PATH = easycycle.get_dataset_info(args.dataset_id)
    filesystem = get_filesystem(ROOT_PATH)
    if args.package_id:
        feature_domain = ""
        if args.index_version is not None:
            domain = f"index_{args.index_version}"
            feature_domain = domain
        feature_dst_dir = os.path.join(
            ROOT_PATH,
            f"features/{feature_domain}/{feature_type}_{feature_version}",
        )
        packing_callback(args.package_id, feature_dst_dir)
