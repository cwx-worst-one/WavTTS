import re
import time
from multiprocessing import Manager
from multiprocessing.dummy import Pool
from multiprocessing.pool import Pool
from turtle import st

import webdataset as wds
from lightning_fabric.utilities.cloud_io import get_filesystem
from tqdm import tqdm

from samantha.dataio.webdataset.ra_wds import WebDataset
from samantha.utils.hdfs_helper import exists


def fn_sub(item, queue):
    item = item.strip()
    k, tar_k = item.split("/")[-1].split(".parquet.")
    tar_k = tar_k.strip(".SUCCESS")
    tar_path = re.sub("/[^/]*$", f"/{tar_k}_%06d.tar", item.replace("index_1", "data"))
    tar_paths = []
    for i in range(10000):
        cur_path = tar_path % i
        if exists(cur_path):
            tar_paths.append(cur_path)
        else:
            break
    queue.put((k, tar_paths))


def fn(index_lst):
    with open(index_lst) as f:
        index_wvae = f.readlines()
    index_wvae_dict = {}
    pool = Pool(100)
    queue = Manager().Queue()
    for item in tqdm(index_wvae):
        pool.apply_async(func=fn_sub, args=(item, queue))

    while True:
        try:
            k, tar_paths = queue.get(timeout=10)
            if k in index_wvae_dict:
                index_wvae_dict[k].extend(tar_paths)
            else:
                index_wvae_dict[k] = tar_paths
            print(f"{len(index_wvae_dict)=}")
        except Exception as e:
            print("done")
            break
    pool.close()
    pool.join()
    return index_wvae_dict


def read_tar(tar_lst):
    wds = WebDataset(tar_lst)
    samples = {}
    for sample in wds:
        samples[sample["__key__"]] = sample
    return samples


def write_tar(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        samples, tar_path = item
        fs = get_filesystem(tar_path)
        stream = fs.open(tar_path, mode="wb")
        tarstream = wds.TarWriter(stream)
        for sample in samples:
            tarstream.write(sample)
        tarstream.close()
        print(f"done {tar_path=}")


def merge(k, tar_paths1, tar_paths2, queue):
    samples_1 = read_tar(tar_paths1)
    samples_2 = read_tar(tar_paths2)
    samples_merge = []
    part = re.findall("part=[0-9]*", tar_paths1[0])[0]
    for sk, sv in samples_1.items():
        sv.update({"bn_3.1": samples_2[sk]["bns"]})
        samples_merge.append(sv)
    out_path = f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/data/bigtts/test_package/{part}/{k}.tar"
    print(f"merge {out_path=}")
    queue.put((samples_merge, out_path))


index_wvae_dict = fn("wvae.index")
index_wvae_31_dict = fn("wvae_3.1.index")

print(f"{len(index_wvae_dict)=}, {len(index_wvae_31_dict)=}")
print(list(index_wvae_dict.items())[:10])

wpool = Pool(50)
rpool = Pool(10)
queue = Manager().Queue()
start = time.perf_counter()
for k, tar_paths in tqdm(index_wvae_dict.items()):
    rpool.apply_async(
        func=merge,
        args=(k, tar_paths, index_wvae_31_dict[k], queue),
        error_callback=lambda x: print("rpool:", x),
    )

wpool.apply_async(
    func=write_tar, args=(queue,), error_callback=lambda x: print("wpool:", x)
)


rpool.close()
rpool.join()

for _ in range(10):
    queue.put(None)

wpool.close()
wpool.join()

end = time.perf_counter()
print(f"cost: {end - start:.4f}")
