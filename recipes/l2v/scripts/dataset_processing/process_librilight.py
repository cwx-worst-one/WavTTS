import json
import multiprocessing
import os
import json
import numpy as np
from tqdm import tqdm
import librosa
from pathlib import Path
import shutil
import webdataset
from webdataset import WebDataset
import subprocess
import multiprocessing as mp


import sys


def create_idx(tar_json_fp):
    tar_hdfs_url, json_hdfs_url = tar_json_fp
    
    tar_name = Path(tar_hdfs_url).name
    index_output_fp = f'/mnt/bn/umm/data/librilight/{tar_name}.index'
    index_writer = open(index_output_fp, 'w')
        
    cat = subprocess.Popen(["hadoop", "fs", "-cat", json_hdfs_url], stdout=subprocess.PIPE)
    json_local = json.load(cat.stdout)

    wd = WebDataset(f'pipe: hdfs dfs -cat {tar_hdfs_url}')

    for index, item in enumerate(wd):
        file_id = item['__key__']
        if file_id in json_local:
            sents = json_local[file_id]['sent']
            meta = []
            for sent in sents:
                meta.append({'start_time': sent['start_time'], 
                             'end_time': sent['end_time'],
                             'speaker_id': sent['speaker_id'],
                             'text': sent['text'],
                             'debug': sent['debug']})
            meta_str = json.dumps(meta)
            index_writer.write(f'{file_id}\t{meta_str}\n')        
    index_writer.close()


tar_files = [f'hdfs://haruna/home/byte_speech_sv/user/dingchen.2101/AudioGPT_data/librilight/shard_{idx:05d}.tar' for idx in range(439)]
json_files = [f'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/jitong/data/tts/valle/mos3.8_sim0.0_snr7_rms-13_asr0.85/jsons/librilight_230711/shard_{idx:05d}.tar.json' for idx in range(439)]

with mp.Pool(32) as pool:
    work = pool.imap_unordered(create_idx, zip(tar_files, json_files))
    for _ in tqdm(work, total=len(tar_files)):
        pass

# write url2inx file
url2inx_fp = f' '
url2inx_writer = open(url2inx_fp, 'w')

for input_hdfs_url in tar_files:
    tar_name = Path(input_hdfs_url).name
    index_output_fp = f'/mnt/bn/umm/data/librilight/{tar_name}.index'
    url2inx_writer.write(f'{input_hdfs_url}\t{index_output_fp}\n')

url2inx_writer.close()
