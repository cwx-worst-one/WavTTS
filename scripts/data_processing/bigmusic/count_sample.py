import os
import multiprocessing as mp


def count_one(idx):
    result = os.popen(f"hdfs dfs -cat {idx}").read().split("\n")
    return len(result)


if __name__ == "__main__":
    path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/kaiping/2023-09-20/url2index.txt"
    result = os.popen(f"hdfs dfs -cat {path}").read().split("\n")
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
