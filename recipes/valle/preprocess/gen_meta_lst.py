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
        res = subprocess.check_output(
            f"hdfs dfs -ls \"{path}\"",
            shell=True
        ).decode()
        res = list(map(lambda x: x.split()[-1],
                filter(lambda x: x!="", res.split("\n"))))
        return res, key
    except subprocess.CalledProcessError as e:
        print("Warning:", e)
        return None

def get_json_lst(c1, c2, sup):
    res = get_hdfs_lst(c1, sup)
    if res is None:
        res = get_hdfs_lst(c2, sup)
        return res

def main(args):
    res = {}
    idx2wds = []
    try:
        wds_fs, _ = get_hdfs_lst(os.path.join(args.wds, "*.tar.orig"))
    except:
        wds_fs, _ = get_hdfs_lst(os.path.join(args.wds, "*.tar"))

    json_miss_num = 0
    for wds in tqdm(wds_fs):
        idx = re.sub(".orig", "", os.path.basename(wds))
        pattern = f"MergedWithAed_{idx}.json"
        f1 = os.path.join(args.meta, pattern)
        pattern = f"{idx}.json"
        f2 = os.path.join(args.meta, pattern)
        if os.path.exists(f1):
            res[wds] = [f1]
            idx2wds.append(wds)
        elif os.path.exists(f2):
            res[wds] = [f2]
            idx2wds.append(wds)
        else:
            json_miss_num += 1

    os.makedirs(args.outputs, exist_ok=True)
    with open(os.path.join(args.outputs, "wds2meta.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outputs, "wds.lst"), "w", encoding="utf-8") as fo:
        for line in idx2wds:
            fo.write(f"{line}\n")

    with open(os.path.join(args.outputs, "wds_valid.lst"), "w", encoding="utf-8") as fo:
        for line in idx2wds:
            fo.write(f"{line}\n")
            break

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
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    args = parser.parse_args()
    main(args)
