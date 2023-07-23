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

def quality_check(sent):
    # confidence > 0.6、rms_max > -13、snr > 6、speaker_similarity > 0.6
    try:
        rms_max = sent['rms_stats']['rms_max']
        snr = sent['snr']
        speaker_similarity_min = sent['speaker_similarity']['min']
        if rms_max > -13 and snr > 6 and speaker_similarity_min > 0.6:
            return True
        else:
            return False
    except Exception as e:
        return False

def tacolab_postprocess(meta):

    def is_phone(ph):
        return ph not in set(["sil", "sp", "pau", "<unk>"])
    def is_en(lab):
        return lab.startswith("E")

    short_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t1"
    mid_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t2"

    words = []
    for word in meta["words"]:
        if word.get("tag") == "<aed>":
            continue
        words.append(word)

    labels = meta["labels"].split("\n")
    # check if tacolab can align with words, if not, only insert short-sp.
    label_cnt = 0
    mis_align = True
    for label in labels:
        data = label.split("\t")
        if len(data) != 5:
            continue
        phone, tone, wordpost, wordcateg, prosody = data
        if is_phone(phone) and prosody != "0":
            label_cnt += 1
    if label_cnt != len(words):
        mis_align = True
        print(f"Warning: mis-align in {meta['text']}")

    idx = 0
    cur_st, pre_et = 0, 0
    flag = False
    new_lab = []
    for label in labels:
        word = words[0] if mis_align else words[idx]
        data = label.split("\t")
        if len(data) != 5:
            continue
        phone, tone, wordpost, wordcateg, prosody = data
        if flag:
            flag = False
            cur_st = word["start_time"]
            # mid sp > 50ms, short sp < 50ms
            if not mis_align and cur_st - pre_et > 0.05:
                new_lab.append(mid_sp)
            else:
                new_lab.append(short_sp)
        if is_phone(phone) and prosody != "0":
            idx += 1
            if is_en(phone) and prosody == "1":
                flag = True
                pre_et = word["end_time"]
                cur_lab = f"{phone}\t{tone}\t{wordpost}\t{wordcateg}\t0"
            else:
                cur_lab = label
            new_lab.append(cur_lab)
        else:
            new_lab.append(label)

    # safe check
    if not mis_align and idx != len(words):
        print(idx, len(words))
        print(words)
        import pdb;pdb.set_trace()
    return new_lab

def gen_json(fn, outdir, longform=False):
    wds_name = os.path.splitext(os.path.basename(fn))
    total_res = {}
    fon = os.path.join(outdir, f"{wds_name}.json")
    with open(fn, "r", encoding="utf-8") as fi:
        for key, data in tqdm(json.load(fi).items()):
            # fon = os.path.join(outdir, f"{key}.json")
            res = []
            pre_spk = None
            st, et = 0, 0
            tacolab = []
            suffix = 0
            fon = os.path.join(outdir, f"{key}.json")
            for sent in data["sent"]:
                if quality_check(sent):
                    cur_spk = sent["speaker_id"]
                    cur_st = sent["start_time"]
                    cur_et = sent["end_time"]
                    # cur_tacolab = sent["labels"].split("\n")
                    cur_tacolab = tacolab_postprocess(sent)
                    cur_lastword = sent["words"][-1]["word"]
                    # idx = f"{key}_{suffix:0>6}"
                    if longform and cur_spk == pre_spk:
                        et = cur_et
                        # TODO: tacolabel 拼接
                        # 1. 子句末标点修正 + 边界修正: 已经在 sami-tacolab 中处理
                        # 2. 句间SP插入： tacolab_postprocess
                        # 3. 词间空格插入： tacolab_postprocess
                        tacolab.extend(cur_tacolab)
                    else:
                        if len(tacolab) > 0:
                            res.append({
                                "start_time": st,
                                "end_time": et,
                                "labels": "\n".join(tacolab)
                            })
                            suffix += 1

                        tacolab = []
                        st = cur_st
                        et = cur_et
                        pre_spk = cur_spk
                        tacolab.extend(cur_tacolab)
                else:
                    if len(tacolab) > 0:
                        res.append({
                            "start_time": st,
                            "end_time": et,
                            "labels": "\n".join(tacolab)
                        })
                        suffix += 1

                    tacolab = []
                    st, et = 0, 0
                    pre_spk = None

            total_res[key] = {"sent": res}
        with open(fon, "w", encoding="utf-8") as fo:
            json.dump(total_res, fo, ensure_ascii=False, indent=2)

def main(args):
    outdir = args.outputs
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(args.inputs):
        wds = os.path.join(args.inputs, f)
        gen_json(wds, outdir, args.longform)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
        default="",
        dest="inputs", help="")
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    parser.add_argument("--longform",
        default=False, const=True, nargs="?",
        dest="longform", help="")
    args = parser.parse_args()
    main(args)
