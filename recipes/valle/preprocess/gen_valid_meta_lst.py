import os
import sys
import re
import json
from collections import defaultdict
from tqdm import tqdm
import glob
from multiprocessing import Pool
import subprocess
import tempfile
import shutil

def get_lst(fn, fon):
    with open(fn, "r", encoding="utf-8") as fi, \
        open(fon, "w", encoding="utf-8") as fo:
        data = json.load(fi)
        lst = list(data.keys())
        assert len(lst) == len(set(lst)), f"{fn}, {len(lst)} != {len(set(lst))}"
        for line in lst:
            fo.write(f'{line}\n')

def get_fs(fn):
    fs = []
    with open(fn, "r", encoding="utf-8") as fi:
        data = json.load(fi)
        for k, v in data.items():
            fs.extend(v)
    return fs

def main(args):
    cnt = 0
    p = Pool(64)
    # fs = glob.glob(os.path.join(args.inputs, "*.json"))
    fs = get_fs(args.inputs)
    bar = tqdm(total=len(fs))
    update = lambda *args: bar.update()
    if args.debug:
        fs = fs[:10]
    tmp_dir = tempfile.mkdtemp()
    for f in tqdm(fs):
        fon = os.path.join(tmp_dir, os.path.basename(f))
        get_lst(f, fon)
        # p.apply_async(get_lst, args=(f, fon,), callback=update)
    # p.close()
    # p.join()
    subprocess.check_call(f"cat {tmp_dir}/* > {args.outputs}",
        shell=True)
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
        default="",
        help="inputs file/dir")
    parser.add_argument("--outputs",
        default="",
        help="outputs file/dir")
    parser.add_argument("--debug",
        default=False, const=True, nargs="?",
        help="debug mode")
    args = parser.parse_args()
    main(args)