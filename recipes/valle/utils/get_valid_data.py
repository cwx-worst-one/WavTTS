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
from tqdm import tqdm
from remote_io import load_json

def quality_check(sent):
    # confidence >= 0.6、rms_max >= -13、snr >= 6、speaker_similarity >= 0.6, mos >= 2.8
    try:
        rms_max = sent['rms_stats']['rms_max']
        snr = sent['snr']
        speaker_similarity_min = sent['speaker_similarity']['min']
        mos = sent['mos']
        if rms_max >= -13 and snr >= 6 and speaker_similarity_min >= 0.6 and mos >= 2.8:
            return True
        else:
            return False
    except Exception as e:
        return False

def main(args):
    wds2meta = load_json(args.wds2meta_path)
    total_dur = 0
    valid_dur = 0
    for wds in tqdm(wds2meta.keys()):
        metas = wds2meta[wds]
        for meta in metas:
            meta_data = load_json(meta)
            for utt_10mins in meta_data.keys():
                sents = meta_data[utt_10mins]['sent']
                for sent in sents:
                    if quality_check(sent):
                        valid_dur += (sent["end_time"] - sent["start_time"])
                    total_dur += (sent["end_time"] - sent["start_time"])
        print("total_dur: ", total_dur / 60 / 60)
        print("valid_dur: ", valid_dur / 60 / 60)
    print("total_dur: ", total_dur / 60 / 60)
    print("valid_dur: ", valid_dur / 60 / 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wds2meta_path",
        default="",
        dest="wds2meta_path", help="")
    args = parser.parse_args()
    main(args)
