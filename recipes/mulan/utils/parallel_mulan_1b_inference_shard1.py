import os
from multiprocessing import Pool

params = []

total_N = 32
for i in range(0, 8):
    cmd = f"python3 recipes/mulan/batch_mulan_1b_inference.py --shard_id {i} --n_shards {total_N}"
    params.append(cmd)


def do(cmd):
    os.system(cmd)


print(params)
with Pool(len(params)) as p:
    print(p.map(do, params))

