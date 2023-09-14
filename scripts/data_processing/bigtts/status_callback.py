import argparse
import logging
import os

import requests

logger = logging.getLogger(__name__)


def packing_callback(packaging_id, dest_dir, status="success"):
    region = os.environ.get("ARNOLD_REGION")
    logger.info(f"{region=}")
    domains = {"US": "bigspeech.byteintl.net", "CN": "bigspeech.bytedance.net"}
    url = f"https://{domains[region]}/platform/api/v3/data/dataset_collection/packed_callback"  # noqa

    data = {"packaging_id": packaging_id, "status": status, "dest_dir": dest_dir}
    logger.info(f"{data=}")
    # 可选：设置请求头
    # headers = {
    #     'Content-Type': 'application/json',
    #     'x-use-ppe': '1',
    #     'x-tt-env': 'ppe_bigspeech_zqd'
    # }
    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 检查响应状态码
    if response.status_code == 200:
        logger.info(f"callback successed {response=}")
    else:
        logger.error(f"request failed {response=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_id", type=str, nargs="+", required=True)
    parser.add_argument("--dest_dir", type=str, nargs="+", required=True)
    args = parser.parse_args()
    for pid, ddir in zip(args.package_id, args.dest_dir):
        packing_callback(packaging_id=pid, dest_dir=ddir)
