import pandas as pd
import ujson
from tqdm import tqdm
from typing import List
from samantha.dataio.data_bucket import data_bucket
from samantha.utils.hdfs_tools import hdfs_loadtxt_cache, hdfs_loadtxt

class IndexReader:

    def __init__(self, url2index: str, cache: bool = False):
        url2indexes = hdfs_loadtxt(url2index)

        url2indexes = [u.split("\t")[1] for u in url2indexes]

        print(f"Loaded {len(url2indexes)} index urls")

        indexes = []
        for u in tqdm(url2indexes):
            index = self.load_index(u, cache=cache)
            indexes.extend(index)

        self.indexes = pd.DataFrame(indexes)

    @staticmethod
    def load_index(index_url: str, cache: bool = False) -> List[dict]:
        if cache:
            raw_index = hdfs_loadtxt_cache(index_url)
        else:
            raw_index = hdfs_loadtxt(index_url)
        index = {}
        for i in raw_index:
            key, index_data = i.split("\t", 1)

            index_data = ujson.loads(index_data)
            index_data["index_url"] = index_url
            index_data["key"] = key

            if key in index:
                raise Exception("key already present in index")

            index[key] = index_data
        return list(index.values())

if __name__ == "__main__":
    url2index = data_bucket(
        "data/music/billboard_hot200_v2/24000hz/test/20231026_genre/url2index.txt"
    )

    reader = IndexReader(url2index)
