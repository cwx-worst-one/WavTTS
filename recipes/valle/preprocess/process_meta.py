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
import logging

FORMAT = '[%(asctime)s] %(message)s'

os.environ["TRASH_DIR"] = "/mnt/bn/jeffus/data/valle/longform/tmp"

def quality_check(sent):
    # confidence > 0.6、rms_max > -13、snr > 6、speaker_similarity > 0.6
    try:
        rms_max = sent['rms_stats']['rms_max']
        snr = sent['snr']
        speaker_similarity_min = sent['speaker_similarity']['min']
        mos = sent['mos']
        # if rms_max > -13 and snr >= 6 and speaker_similarity_min > 0.6 and mos > 2.8:
        #     return True
        # else:
        #     return False
        if rms_max <= -13:
            return False, "rms_max <= -13"
        if snr < 6:
            return False, "snr < 6"
        if speaker_similarity_min <= 0.6:
            return False, "speaker_similarity_min <= 0.6"
        if mos <= 2.8:
            return False, "mos <= 2.8"

        return True, ""
    except Exception as e:
        return False, "missing some key/value"

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

    if not meta["labels"]:
        return None
    labels = meta.get("labels")
    if labels is None: return None
    labels = labels.split("\n")
    # check if tacolab can align with words, if not, only insert short-sp.
    label_cnt = 0
    mis_align = True
    for label in labels:
        try:
            data = label.split("\t")
            if len(data) != 5:
                continue
            phone, tone, wordpost, wordcateg, prosody = data
            if is_phone(phone) and prosody != "0":
                label_cnt += 1
        except Exception as e:
            continue
    if label_cnt != len(words):
        mis_align = True
        # print(f"Warning: mis-align in {meta['text']}")

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
       return None 
    return new_lab

@remote_load(0)
def gen_sub_json(fn, total_res, longform):
    try:
        ori, merge = 0, 0
        empty = 0
        error = 0
        with open(fn, "r", encoding="utf-8") as fi:
            for jdx, (key, data) in enumerate(json.load(fi).items()):
                # fon = os.path.join(outdir, f"{key}.json")
                res = []
                pre_spk = None
                st, et = 0, 0
                tacolab = []
                text = []
                suffix = 0
                for idx, sent in enumerate(data["sent"]):
                    quality, quality_r = quality_check(sent)
                    cur_st = sent.get("start_time")
                    if cur_st is None: continue
                    cur_et = sent.get("end_time")
                    if cur_et is None: continue
                    if et - cur_st > 0.5 or cur_et < cur_st or cur_et < et:
                        error += 1
                        logging.error(f"misaligned ({et}, {cur_st}, {cur_et}) in {key}")
                        quality = False
                        quality_r += "|timestamp mismatch"
                    if quality:
                        cur_text = sent.get("text")
                        if cur_text is None: continue
                        cur_spk = sent.get("speaker_id")
                        if cur_spk is None: continue

                        cur_tacolab = sent.get("labels")
                        if cur_tacolab is None: continue
                        cur_tacolab = cur_tacolab.split("\n")
                        # cur_tacolab = tacolab_postprocess(sent)
                        if cur_tacolab is None: continue

                        # idx = f"{key}_{suffix:0>6}"
                        if longform and cur_spk == pre_spk:
                            if cur_et - st > 60:
                                res.append({
                                    "start_time": st,
                                    "end_time": et,
                                    "speaker_id": pre_spk,
                                    "labels": "\n".join(tacolab),
                                    "text": "".join(text),
                                    "debug": "reach 60s"
                                })
                                suffix += 1
                                tacolab = []
                                text = []
                                st = cur_st
                                pre_spk = None
                            et = cur_et
                            # TODO: tacolabel 拼接
                            # 1. 子句末标点修正 + 边界修正: 已经在 sami-tacolab 中处理
                            # 2. 句间SP插入： tacolab_postprocess
                            # 3. 词间空格插入： tacolab_postprocess
                            tacolab.extend(cur_tacolab)
                            text.append(cur_text)
                        else:
                            if len(tacolab) > 0:
                                res.append({
                                    "start_time": st,
                                    "end_time": et,
                                    "speaker_id": pre_spk,
                                    "labels": "\n".join(tacolab),
                                    "text": "".join(text),
                                    "debug": "speaker change"
                                })
                                suffix += 1
                                tacolab = []
                                text = []

                            st = cur_st
                            et = cur_et
                            pre_spk = cur_spk
                            tacolab.extend(cur_tacolab)
                            text.append(cur_text)
                    else:
                        if len(tacolab) > 0:
                            res.append({
                                "start_time": st,
                                "end_time": et,
                                "speaker_id": pre_spk,
                                "labels": "\n".join(tacolab),
                                "text": "".join(text),
                                "debug": quality_r
                            })
                            suffix += 1
                            tacolab = []
                            text = []

                        st, et = cur_st, cur_et
                        pre_spk = None
                if len(tacolab) > 0:
                    res.append({
                        "start_time": st,
                        "end_time": et,
                        "speaker_id": pre_spk,
                        "labels": "\n".join(tacolab),
                        "text": "".join(text),
                    })
                    suffix += 1
                    tacolab = []
                    text = []
                if len(res) == 0:
                    empty += 1
                    continue
                ori += idx
                merge += len(res)
                total_res[key] = {"sent": res}
        logging.warning(f"ori: {ori}, merge: {merge}, reduced: {ori - merge}, empty: {empty}/{jdx}, misalign: {error}/{jdx}")
    except Exception as e:
        raise e
    finally:
        os.remove(fn)

def gen_json(k, fs, outdir, longform=False):
    fon = os.path.join(outdir, f"{k}.tar.json")
    if os.path.exists(fon):
        logging.warning(f"{fon} exists, skipped!")
        return
    logging.info(f"{fon}")
    total_res = {}
    for fn in fs:
        gen_sub_json(fn, total_res, longform)
    
    with open(fon, "w", encoding="utf-8") as fo:
        json.dump(total_res, fo, ensure_ascii=False, indent=2)

def main(args):
    outdir = args.outputs
    os.makedirs(outdir, exist_ok=True)
    log_fn = os.path.join(outdir, "log.txt")
    logging.basicConfig(format=FORMAT,
                        filename=log_fn,
                        filemode='a',
                        level=logging.DEBUG)
    if args.inputs.startswith("hdfs://"):
        fs, _ = get_hdfs_lst(args.inputs)
    else:
        fs = os.listdir(args.inputs)
        fs = list(map(lambda x: os.path.join(args.inputs, x), fs))

    p = Pool(50)
    bar = tqdm(total=len(fs))
    update = lambda *args: bar.update()

    res = defaultdict(list)

    for f in fs:
        basename = os.path.basename(f).split(".")[0]
        res[basename].append(f)
    for k, v in res.items():
        dirname = os.path.dirname(v[0])
        tgt = os.path.join(dirname, f"{k}.tar.json")
        if tgt in v:
            # logging.warning(f"single file in {k}")
            res[k] = [tgt]

    cnt = 0
    for k, v in tqdm(res.items()):
        if args.debug:
            gen_json(k, v, outdir, args.longform)
        else:
            p.apply_async(gen_json,
                args=(k, v, outdir, args.longform,),
                callback=update)
        if args.num > 0:
            cnt += 1
            if cnt > args.num:
                break
    p.close()
    p.join()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
        default="",
        dest="inputs", help="")
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    parser.add_argument("--num",
        default=-1, type=int,
        dest="num", help="")
    parser.add_argument("--debug",
        default=False, const=True, nargs="?",
        dest="debug", help="")
    parser.add_argument("--longform",
        default=False, const=True, nargs="?",
        dest="longform", help="")
    args = parser.parse_args()
    main(args)
