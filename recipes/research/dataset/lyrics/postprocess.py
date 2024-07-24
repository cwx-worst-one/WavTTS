"""First, copy the lyric files to a local directory:
hdfs dfs -get hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_lyrics

"""

import json
import os
from glob import glob

import pandas as pd
from tqdm import tqdm

from samantha.data.lyrics import Lyrics

if __name__ == "__main__":
    OUT_FP = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_lyrics/billboard_hot_200-v2_lyrics.p"
    # df = pd.read_pickle(OUT_FP)

    LYRICS_FP = glob("billboard_hot_200-v2_lyrics/*.txt")

    df = []
    for fp in tqdm(LYRICS_FP):
        with open(fp) as f:
            l = json.load(f)

        key = os.path.splitext(os.path.basename(fp))[0]
        d = {"key": key, "lyrics": l}
        df.append(d)
    df = pd.DataFrame(df)
    df = df.set_index("key")
    df.to_pickle(OUT_FP)
