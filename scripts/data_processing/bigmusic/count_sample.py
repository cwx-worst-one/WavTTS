import multiprocessing as mp
import os


def count_one(idx):
    if idx.startswith("hdfs://"):
        result = os.popen(f"hdfs dfs -cat {idx}").read().split("\n")
    else:
        result = open(idx, "r").read().split("\n")
    return len(result)


if __name__ == "__main__":
    path = "/mnt/bn/audio-diffusion/data/mcc_pgc_600k/gpt_url2idx.txt"
    if path.startswith("hdfs://"):
        result = os.popen(f"hdfs dfs -cat {path}").read().split("\n")
    else:
        result = open(path, "r").read().split("\n")
    result = [r for r in result if r]
    print(len(result))

    pool = mp.Pool(80)

    total = 0
    idxes = []
    for line in result:
        tar, idx = line.split("\t")
        idxes.append(idx)

    for count in pool.map(count_one, idxes):
        total += count
    print(total)
