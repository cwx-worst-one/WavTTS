import json
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import subprocess
import multiprocessing as mp
import pandas as pd
import sys
import langid

def isZhVocal(x):
    return "此歌曲为没有填词的纯音乐" not in x and (langid.classify(x)[0] == 'zh')

def filter_idx(csv_pair):
    csv_file, new_csv_file = csv_pair    
    df = pd.read_csv(csv_file)
    required_fields = ['lyrics', 'full_play_url']
    df_has_lyrics_url = df[df[required_fields].notnull().all(1)]
    df_has_zh_lyrics_url = df_has_lyrics_url[df_has_lyrics_url["lyrics"].apply(isZhVocal)]
    print("has zh vocal lyrics url", len(df_has_zh_lyrics_url))    
    df_has_zh_lyrics_url.to_csv(new_csv_file)
    
csv_files = [f'hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/music/qqmusic/part-{idx:05d}-fa1f06f8-7b48-4546-9781-eb4f3c1d7163-c000.csv'
             for idx in range(4046)]
new_csv_files = [f'hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/music/qqmusic_zh_vocal/part-{idx:05d}-fa1f06f8-7b48-4546-9781-eb4f3c1d7163-c000.csv'
             for idx in range(4046)]

with mp.Pool(32) as pool:
    work = pool.imap_unordered(filter_idx, zip(csv_files, new_csv_files))
    for _ in tqdm(work, total=len(csv_files)):
        pass

# csv_file = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/music/qqmusic/part-00000-fa1f06f8-7b48-4546-9781-eb4f3c1d7163-c000.csv"
# new_csv_file = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/music/qqmusic_zh_vocal/part-00000-fa1f06f8-7b48-4546-9781-eb4f3c1d7163-c000.csv"
# filter_idx((csv_file, new_csv_file))