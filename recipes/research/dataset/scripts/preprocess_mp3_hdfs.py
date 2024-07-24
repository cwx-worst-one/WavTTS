import os
import re
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from samantha.data.audio_utils import normalize_audio
from samantha.data.av_audio import audio_read, audio_write
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.utils.hdfs_tools import hdfs_get, hdfs_mkdir, hdfs_put


def convert_file(row: pd.Series, audio_write_fn):
    global FAILED

    key = row["meta_song_id"]
    fp = row["fp"]

    suffix = os.path.splitext(fp)[1]
    tmp_path = f"/tmp/{key}{suffix}"

    if not os.path.exists(fp):
        hdfs_get(fp, tmp_path)

    try:
        audio, sr = audio_read(tmp_path)

        # VERY IMPORTANT PREPROCESSING PARAMETERS!!!!
        audio = normalize_audio(
            audio,
            normalize=True,
            strategy="loudness",
            peak_clip_headroom_db=1.0,
            rms_headroom_db=18,
            loudness_headroom_db=16,
            loudness_compressor=False,
            sample_rate=sr,
            log_clipping=True,
        )

        out_fp = fp.replace(SOURCE_PATH, OUT_FP)
        # splitting logic:
        # split_samples = split_sec * sr
        # audios = audio.split(split_samples, dim=-1)
        # for idx, audio in enumerate(audios):
        # out_fp = os.path.join(OUT_FP, f"{fp.stem}_{idx}")
        # audio_write_fn(out_fp, audio, sr)
        audio_write_fn(out_fp, audio, sr)

    except Exception as e:
        print(e)
        FAILED += 1
        print(f"Failed: {FAILED}/{len(files)}")

    os.remove(tmp_path)


def audio_write_fn(out_fp, audio, sr):
    # remove current extension
    out_fp = os.path.splitext(out_fp)[0]

    tmp_path = os.path.join("/tmp", out_fp)

    hdfs_mkdir(os.path.dirname(out_fp))

    audio_write(
        tmp_path,
        audio,
        sr,
        format="mp3",
        mp3_rate=320,
        normalize=False,
        add_suffix=True,
    )

    hdfs_put(tmp_path + ".mp3", out_fp + ".mp3")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def extract_meta_id(fp: str, parent_dir):
    meta_id = os.path.relpath(fp, parent_dir)
    return os.path.splitext(meta_id)[0].replace("/", "_")


def chunks(l, n):
    for i in range(0, n):
        yield l[i::n]


if __name__ == "__main__":
    worker_id = int(os.getenv("ARNOLD_ID", 0))
    num_workers = int(os.getenv("ARNOLD_WORKER_NUM", 1))
    print(f"ARNOLD WORKER: {worker_id + 1}/{num_workers}")

    FAILED = 0
    # SOURCE_PATH = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2"
    # SOURCE_PATH = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/jamendo"
    # SOURCE_PATH = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/musdb18hq"
    SOURCE_PATH = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/playlist_v5"

    files = hdfs_ls(os.path.join(SOURCE_PATH, "*"))
    ## hdfs dfs -ls hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/playlist_v5  | awk '{print $8}' > playlistv5.txt
    # files = open("playlistv5.txt").read().splitlines()
    files = list(chunks(files, n=num_workers))

    print(f"NUM FILE LISTS: {len(files)}")

    files = files[worker_id]
    print(f"PROCESSING {len(files)} files")

    # OUT_FP = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_normalised_-16LUFS_mp3"
    # OUT_FP = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/jamendo_normalised_-16LUFS_mp3"
    # OUT_FP = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/musdb18hq_normalised_-16LUFS_mp3"
    OUT_FP = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/playlist_v5_normalised_-16LUFS_mp3"
    hdfs_mkdir(OUT_FP)

    existing_files = hdfs_ls(os.path.join(OUT_FP, "*"))

    meta_ids = list(map(lambda f: extract_meta_id(f, SOURCE_PATH), files))
    existing_meta_ids = list(map(lambda f: extract_meta_id(f, OUT_FP), existing_files))

    remainder_meta_ids = list(set(meta_ids) - set(existing_meta_ids))

    df = pd.DataFrame(files, columns=["fp"])
    df["meta_song_id"] = df["fp"].apply(lambda f: extract_meta_id(f, SOURCE_PATH))
    df["exists"] = ~df["meta_song_id"].isin(remainder_meta_ids)

    print("Percentage done:", df["exists"].sum() / len(df) * 100)

    df = df[df["exists"] == False]

    Parallel(n_jobs=16)(
        delayed(convert_file)(row, audio_write_fn)
        for idx, row in tqdm(df.iterrows(), total=len(df))
    )
