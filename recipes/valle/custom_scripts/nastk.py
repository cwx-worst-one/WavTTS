#!/usr/bin/env python
# -*- coding:utf-8 -*-
#***********************************************
#      Filename: nastk.py
#        Author: Jeff Pan
#         Email: panjunjie.jeff@bytedance.com
#   Description: --
#        Create: 2023-04-26 14:39:43
# Last Modified: 2023-04-26
#***********************************************

import os
import sys
import re
import json
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import subprocess

def check_type(path):
    if not path:
        return None
    if path.startswith("hdfs"):
        return path
    elif path.startswith("/mnt"):
        volume = path.split("/")[3]
        path = "/".join(path.split("/")[4:])
        if volume == "sa-audiogen":
            return f"bytenas://cn:{volume}/{volume}/{path}"
        else:
            return f"bytenas://cn:{volume}/{path}"

    else:
        path = os.path.abspath(path)
        return f"file://{path}"

def nastk(op, src, tgt=None, nastk=True, display=True):
    if tgt is not None:
        if display:
            print(f"nastk {op} {src} {tgt}")
        if nastk:
            subprocess.check_call(f"nastk {op} {src} {tgt}", shell=True)
        else:
            subprocess.check_call(f"{op} {src} {tgt}", shell=True)
    else:
        if display:
            print(f"nastk {op} {src}")
        if nastk:
            subprocess.check_call(f"nastk {op} {src}", shell=True)
        else:
            subprocess.check_call(f"{op} {src}", shell=True)

def addr_preprocess(addr):
    if addr.startswith("bytenas://"):
        if re.findall("cn:sa-audiogen", addr):
            return re.sub("bytenas://cn:sa-audiogen/sa-audiogen/",
                              "/mnt/bn/sa-audiogen/", addr)
        return re.sub(r"bytenas://cn:(.*?)/", r"/mnt/bn/\1/", addr)
    elif addr.startswith("file://"):
        return re.sub("file://", "", addr)
    else:
        return addr

def main(args):
    src = addr_preprocess(args.src)
    tgt = addr_preprocess(args.tgt)
    op = args.op
    if op[0] == "split":
        p = Pool(32)
        chunk_size = int(op[1])
        cnt = 0
        fs = os.listdir(src)
        bar = tqdm(total=len(fs))
        update = lambda *args: bar.update()
        chunk_id = 0
        for f in fs:
            if cnt % chunk_size == 0:
                chunk_id += 1
                cnt = 0
                sub_tgt = f"{tgt}/{chunk_id:0>5}"
                os.makedirs(sub_tgt, exist_ok=True)
            src_fn = os.path.join(src, f)
            tgt_fn = os.path.join(sub_tgt, f)
            p.apply_async(nastk,
                          args=("cp", src_fn, tgt_fn, False, False,),
                          callback=update)
            cnt += 1
        print(f"Total Chunk {chunk_id}")
        p.close()
        p.join()
    else:
        op = " ".join(args.op)
        src = check_type(src)
        tgt = check_type(tgt)
        nastk(op, src, tgt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        default="", nargs="+",
        dest="op", help="")
    parser.add_argument(
        "--src", "-s",
        default="", required=True,
        dest="src", help="")
    parser.add_argument(
        "--tgt", "-t",
        default="", required=False,
        dest="tgt", help="")
    args = parser.parse_args()
    main(args)
