import os
from glob import glob

from tqdm import tqdm

from samantha.utils.hdfs_tools import hdfs_get

if __name__ == "__main__":

    os.makedirs("index_files", exist_ok=True)
    hdfs_get(
        "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/speech/librilight/large/*.index",
        "index_files",
    )
    index_fps = glob("./index_files/*.index")
    all_index = []
    for fp in tqdm(index_fps):
        with open(fp) as f:
            index = f.read().splitlines()
        all_index.extend(index)

    unique_index = set(all_index)
    print(len(all_index), len(unique_index))
