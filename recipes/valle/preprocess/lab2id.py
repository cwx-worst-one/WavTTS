import os
import sys
import re
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import traceback
import glob
import logging

from recipes.valle.datasets.sami_tacolabel import enc_taco_label_no_bytes

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def convert_tacolab_to_text_id(tacolab, text_converter_dict):
    with HiddenPrints():
        metas = enc_taco_label_no_bytes(
            None, 
            tacolab, 
            {"use_prsdword": False, "forced_refix": True})
    text_id =  metas[0].astype(np.int64) * 1_000_000_000 + \
                metas[1].astype(np.int64) * 1_000_000 + \
                metas[2].astype(np.int64) * 1_000 + \
                metas[3].astype(np.int64)
    text_id = np.asarray([text_converter_dict[str(x)] for x in text_id]).astype(np.int64)
    return text_id

def tacolab_postprocess(labels):

    def is_phone(ph):
        return ph not in set(["sil", "sp", "pau", "<unk>"])
    def is_en(lab):
        return lab.startswith("E")

    try:
        short_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t1"

        idx = 0
        flag = False
        new_lab = []
        for label in labels:
            data = label.split("\t")
            if len(data) != 5:
                continue
            phone, tone, wordpost, wordcateg, prosody = data

            if flag:
                flag = False
                new_lab.append(short_sp)
            if is_phone(phone) and prosody != "0":
                idx += 1
                if is_en(phone) and prosody == "1":
                    flag = True
                    cur_lab = f"{phone}\t{tone}\t{wordpost}\t{wordcateg}\t0"
                else:
                    cur_lab = label
                new_lab.append(cur_lab)
            else:
                new_lab.append(label)
        return new_lab
    except Exception as e:
        traceback.print_exc()
        return None

def process(fn, fon, dct, debug):
    labs = []
    with open(fn, "r", encoding="utf-8") as fi:
        for line in fi:
            lab = line.strip()
            if len(lab) == 0:
                continue
            labs.append(lab)
    new_labs = tacolab_postprocess(labs)
    if new_labs is None:
        logging.warning(f"{fn} error, skipped!")
    
    text_id = convert_tacolab_to_text_id(new_labs, dct)
    np.save(fon, text_id)
    if debug:
        print(f"text_id:{text_id.shape}, lab:{len(labs)}, new_lab:{len(new_labs)}")
    return text_id

def main(args):
    os.makedirs(args.outputs, exist_ok=True)
    with open(args.text_dct, "r", encoding="utf-8") as fi:
        dct = json.load(fi)
    for f in tqdm(glob.glob(os.path.join(args.inputs, "*.lab"))):
        fon = os.path.join(args.outputs, f"{os.path.splitext(os.path.basename(f))[0]}.npy")
        text_id = process(f, fon, dct, args.debug)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
        default="",
        help="inputs file/dir")
    parser.add_argument("--outputs",
        default="",
        help="outputs file/dir")
    parser.add_argument("--text_dct",
        default="/mnt/bn/jeffus/repo/samantha/recipes/valle/datasets/dict/metaid_to_textid.json",
        help="text_dct file/dir")
    parser.add_argument("--debug",
        default=False, const=True, nargs="?",
        help="debug mode")
    args = parser.parse_args()
    main(args)