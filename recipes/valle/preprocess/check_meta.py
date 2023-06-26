#!/usr/bin/env python
# -*- coding:utf-8 -*-
#***********************************************
#      Filename: preprocess/process_meta.py
#        Author: Jeff Pan
#         Email: panjunjie.jeff@bytedance.com
#   Description: --
#        Create: 2023-05-26 10:21:41
# Last Modified: 2023-05-26
#***********************************************

import os
import sys
import re
import json
from collections import defaultdict
from tqdm import tqdm
from recipes.valle.utils.remote_io import remote_load
from recipes.valle.preprocess.gen_meta_lst import get_hdfs_lst
from multiprocessing import Pool

def return_time(t):
    if t < 20:
        return 20
    elif t < 40:
        return 40
    else:
        return 60

@remote_load(0)
def gen_json(fn, outdir):
    record = {
        "longform_length": defaultdict(int),
        "longform_length_cnt": defaultdict(int),
        "dialog_length": defaultdict(int),
        "dialog": defaultdict(int),
        "turn": defaultdict(int),
        "spk_num": defaultdict(int),
        "debug": defaultdict(int)
    }
    wds_name = os.path.splitext(os.path.basename(fn))[0]
    fon = os.path.join(outdir, f"{wds_name}.json")

    with open(fn, "r", encoding="utf-8") as fi:
        for jdx, (key, data) in enumerate(json.load(fi).items()):
            # fon = os.path.join(outdir, f"{key}.json")
            cur_spk_list = set([])
            pre_spk = None
            st, et = 0, 0
            turn = 1
            for idx, sent in enumerate(data["sent"]):
                cur_text = sent.get("text")
                cur_spk = sent.get("speaker_id")
                cur_st = sent.get("start_time")
                cur_et = sent.get("end_time")
                if cur_st < et:
                    print(f"Error: time error in {key}")
                    continue
                cur_tacolab = sent.get("labels")
                cur_tacolab = cur_tacolab.split("\n")

                debug = sent.get("debug")
                if debug:
                    record["debug"][debug] += 1

                length = cur_et - cur_st
                record["longform_length"][return_time(length)] += 1
                record["longform_length_cnt"][return_time(length)] += length
                if cur_st - et <= 1:
                    et = cur_et
                    if cur_spk != pre_spk:
                        turn += 1
                        cur_spk_list.add(cur_spk)
                else:
                    if len(cur_spk_list) != 0:
                        key = f"{turn}-{len(cur_spk_list)}"
                        record["dialog"][key] += 1
                        record["turn"][turn] += 1
                        record["spk_num"][len(cur_spk_list)] += 1
                        record["dialog_length"][return_time(et - st)] += 1

                    turn = 1
                    cur_spk_list = set([cur_spk])
                    st = cur_st
                    et = cur_et
                    pre_spk = cur_spk
            if len(cur_spk_list) != 0:
                key = f"{turn}-{len(cur_spk_list)}"
                record["dialog"][key] += 1
                record["turn"][turn] += 1
                record["spk_num"][len(cur_spk_list)] += 1
                record["dialog_length"][return_time(et - st)] += 1
    return record
    

def update_dict(dct, new_dct):
    for key in new_dct.keys():
        if key not in dct:
            dct[key] = defaultdict(int)
        for sub_key in new_dct[key].keys():
            if sub_key not in dct[key]:
                dct[key][sub_key] = new_dct[key][sub_key]
            else:
                dct[key][sub_key] += new_dct[key][sub_key]
        
def main(args):
    outdir = args.outputs
    # os.makedirs(outdir, exist_ok=True)
    if args.inputs.startswith("hdfs://"):
        fs, _ = get_hdfs_lst(args.inputs)
    else:
        import glob
        fs = glob.glob(os.path.join(args.inputs, f"*.json"))
        # fs = os.listdir(args.inputs)
        # fs = list(map(lambda x: os.path.join(args.inputs, x), fs))

    # p = Pool(50)
    # bar = tqdm(total=len(fs))
    # update = lambda *args: bar.update()

    record = {}
    for f in tqdm(fs):
        update_dict(record, gen_json(f, outdir))

    with open(args.outputs, "w", encoding="utf-8") as fo:
        longform_length = sorted(record["longform_length"].items(), key=lambda x: x[0])
        fo.write(f"########## longform_length begin ##########\n")
        total_dur = sum([x for x in record["longform_length_cnt"].values()])
        for k, v in longform_length:
            dur = record["longform_length_cnt"][k]
            avg_dur = dur / v
            fo.write(f"{k}: {v} sent, {dur/3600:.2f}h ({dur/total_dur*100:.2f}%), {avg_dur:.2f} sec/sent\n")
        fo.write(f"########## longform_length end ##########\n")

        dialog_length = sorted(record["dialog_length"].items(), key=lambda x: x[0])
        fo.write(f"########## dialog_length begin ##########\n")
        for k, v in dialog_length:
            fo.write(f"{k}: {v}\n")
        fo.write(f"########## dialog_length end ##########\n")

        turn = sorted(record["turn"].items(), key=lambda x: x[0])
        fo.write(f"########## turn begin ##########\n")
        for k, v in turn:
            fo.write(f"{k}: {v}\n")
        fo.write(f"########## turn end ##########\n")

        spk_num = sorted(record["spk_num"].items(), key=lambda x: x[0])
        fo.write(f"########## spk_num begin ##########\n")
        for k, v in spk_num:
            fo.write(f"{k}: {v}\n")
        fo.write(f"########## spk_num end ##########\n")

        dialog = sorted(record["dialog"].items(), key=lambda x: int(x[0].split("-")[0]))
        fo.write(f"########## diloag begin ##########\n")
        for k, v in dialog:
            fo.write(f"{k}: {v}\n")
        fo.write(f"########## diloag end ##########\n")

        debug = record["debug"].items()
        fo.write(f"########## diloag begin ##########\n")
        for k, v in debug:
            fo.write(f"{k}: {v}\n")
        fo.write(f"########## diloag end ##########\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
        default="",
        dest="inputs", help="")
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    parser.add_argument("--debug",
        default=False, const=True, nargs="?",
        dest="debug", help="")
    args = parser.parse_args()
    main(args)
