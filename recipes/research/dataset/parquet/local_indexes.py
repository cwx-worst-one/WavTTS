import os
from glob import glob
from typing import List


def resolve_data_url_from_index_url(index_shard: str, data_urls: List[str]):
    index_shard = os.path.basename(index_shard)
    for d in data_urls:
        if index_shard in d:
            return d
    return None


def everynoise_data_urls():
    index_urls = sorted(glob("/mnt/bd/everynoise-1/index_1/*.parquet"))
    data_urls = sorted(glob("/mnt/bd/everynoise-*/data/*.parquet"))

    data_urls = [
        {"index": i, "data": resolve_data_url_from_index_url(i, data_urls)}
        for i in index_urls
    ]
    return list(filter(lambda i: i["data"] is not None, data_urls))


def playlist_v5_data_urls():
    index_urls = sorted(glob("playlist_v5/index_1/*.parquet"))
    data_urls = sorted(glob("playlist_v5/data/*.parquet"))

    data_urls = [
        {"index": i, "data": resolve_data_url_from_index_url(i, data_urls)}
        for i in index_urls
    ]
    return list(filter(lambda i: i["data"] is not None, data_urls))


if __name__ == "__main__":
    enoi = everynoise_data_urls()
