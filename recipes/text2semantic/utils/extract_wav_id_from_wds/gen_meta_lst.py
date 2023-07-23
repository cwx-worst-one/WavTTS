#!/usr/bin/env python
# -*- coding:utf-8 -*-
#***********************************************
#      Filename: gen_meta_lst.py
#        Author: Jeff Pan
#         Email: panjunjie.jeff@bytedance.com
#   Description: --
#        Create: 2023-05-29 16:27:05
# Last Modified: 2023-05-29
#***********************************************

import os
import sys
import re
import json
from collections import defaultdict
from tqdm import tqdm
import glob
import subprocess
from multiprocessing import Pool

def get_hdfs_lst(path, key=""):
    try:
        if path.startswith("hdfs://"):
            res = subprocess.check_output(
                f"hdfs dfs -ls \"{path}\"",
                shell=True
            ).decode()
            res = list(map(lambda x: x.split()[-1],
                    filter(lambda x: x!="", res.split("\n"))))
        else:
            res = glob.glob(path)
        return res, key
    except subprocess.CalledProcessError as e:
        print(e)
        return None

def gen_valid_keys(lst):
    res = []
    for f in lst:
        with open(f, "r", encoding="utf-8") as fi:
            data = json.load(fi)
            res.extend(data.keys())
    return res

def main(args):
    res = {}
    idx2wds = {}
    
    if args.chunks:
        wds_fs, _ = get_hdfs_lst(os.path.join(args.wds, "*.tar.orig"))
    else:
        wds_fs, _ = get_hdfs_lst(os.path.join(args.wds, "*.tar"))

    p = Pool(args.jobs)
    bar = tqdm(total=len(wds_fs))
    update = lambda *args: bar.update()
    jobs = []
    print("loop wds_fs")
    for wds in tqdm(wds_fs):
        # json_lst = get_hdfs_lst(os.path.join(args.meta, f"{idx}*.json"))
        if args.chunks:
            key = wds
            idx = key.split('/')[-1][:-9]
            pattern = f"MergedWithAed_{idx}.*.json"
        else:
            key = wds
            idx = key.split('/')[-1][:-4]
            pattern = f"{idx}.json"

        jobs.append(
            p.apply_async(get_hdfs_lst,
                          args=(os.path.join(args.meta, pattern),
                                key),
                          callback=update))
        # res[idx] = json_lst
    p.close()
    p.join()
    json_miss_num = 0
    print("getting res...")
    for j in tqdm(jobs):
        info = j.get()
        if info is not None:
            json_lst, key = info
            if args.chunks:
                idx = key.split('/')[-1][:-9]
            else:
                idx = key.split('/')[-1][:-4]
            if json_lst:
                res[key] = json_lst
                # valid_keys.extend(gen_valid_keys(json_lst))
                if args.chunks:
                    idx2wds[key] = os.path.join(args.wds, idx + ".tar.orig")
                else:
                    idx2wds[key] = os.path.join(args.wds, idx + ".tar")
        else:
            json_miss_num += 1
    os.makedirs(args.outputs, exist_ok=True)
    with open(os.path.join(args.outputs, "wds2meta.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outputs, "wds.lst"), "w", encoding="utf-8") as fo:
        for line in idx2wds.values():
            fo.write(f"{line}\n")
    
    print("json_miss_num: ", json_miss_num)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta",
        default="",
        dest="meta", help="")
    parser.add_argument("--wds",
        default="",
        dest="wds", help="")
    parser.add_argument("--jobs",
        default=50, type=int,
        dest="jobs", help="")
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    parser.add_argument("--chunks",
        default=False, const=True, nargs="?",
        dest="chunks", help="") # chunk for resso podcast
    args = parser.parse_args()
    main(args)
