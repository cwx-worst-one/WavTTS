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

def get_hdfs_lst(path):
    res = subprocess.check_output(
        f"hdfs dfs -ls \"{path}\"",
        shell=True
    ).decode()
    res = list(map(lambda x: x.split()[-1],
            filter(lambda x: x!="", res.split("\n"))))
    return res

def main(args):
    wds_fs = get_hdfs_lst(args.wds)
    os.makedirs(os.path.dirname(args.outputs), exist_ok=True)
    with open(args.outputs, "w", encoding="utf-8") as fo:
        for line in wds_fs:
            fo.write(f"{line}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wds",
        default="",
        dest="wds", help="")
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    args = parser.parse_args()
    main(args)
