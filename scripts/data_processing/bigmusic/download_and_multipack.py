import json
import multiprocessing as mp
import os
import re
import warnings
import pandas as pd
import requests
from utils import (
    get_audio_bytes,
    get_audio_url,
)
from tqdm import tqdm
from webdataset import TarWriter
from lightning_fabric.utilities.cloud_io import get_filesystem
q = mp.Queue(maxsize=512)
def process_line(lyrics_line, ltype):
    lyrics_dict = {}
    try:
        if not lyrics_line.startswith("["):
            return {}
        if ltype == "lrc":
            # "[00:00.00]安静地又说分开"
            # extract [00:00.00]
            lyrics_dict = {}
            timestamp = re.findall(r"\[(\d+:\d+\.\d+)\]", lyrics_line)
            if timestamp:
                timestamp = timestamp[0]
                start = int(timestamp.split(":")[0]) * 60 * 1000 + int(timestamp.split(":")[1].split(".")[0]) * 1000 + int(timestamp.split(":")[1].split(".")[1])
            else:
                print("Error:", lyrics_line)
                return {}
            lyrics_dict["start_time"] = start
            # remove all [00:00.00]
            lyrics_line = re.sub(r"\[\d+:\d+\.\d+\]", "", lyrics_line)
            if not lyrics_line:
                print("Error:", lyrics_line)
                return {}
            lyrics_dict["text"] = lyrics_line
        elif ltype == "krc":
            # "[6750,2490]<0,310,0>安<310,400,0>静<710,360,0>地<1070,350,0>又<1420,360,0>说<1780,350,0>分<2130,360,0>开"
            # extract [6750,2490]
            lyrics_dict = {}
            timestamp = re.findall(r"\[(\d+),(\d+)\]", lyrics_line)
            if timestamp:
                timestamp = timestamp[0]
                start, dur = int(timestamp[0]), int(timestamp[1])
            else:
                print("Error:", lyrics_line)
                return {}
            lyrics_dict["start_time"] = start
            lyrics_dict["end_time"] = start + dur
            # remove all [6750,2490] <0,310,0>
            lyrics_line = re.sub(r"\[\d+,\d+\]", "", lyrics_line)
            lyrics_line = re.sub(r"<\d+,\d+,\d+>", "", lyrics_line)
            if not lyrics_line:
                print("Error:", lyrics_line)
                return {}
            lyrics_dict["text"] = lyrics_line
    except:
        print("Error:", lyrics_line)
        lyrics_dict = {}
    return lyrics_dict


def process_one(chunk, output_tar, output_idx):
    fs = get_filesystem(output_tar)
    fileobj = fs.open(output_tar, mode="wb")
    tar_writer = TarWriter(fileobj)
    idx_writer = fs.open(output_idx, mode="wb")
    for row in tqdm(chunk.to_dict(orient="records")):
        # song_id,song_name,artist_names,album_name,
        # album_image_link,label_ids,full_play_url,
        # merge_genre,merge_language,duration,release_date,
        # aed_info,lyric_link,song_play_cnt_all,song_collect_cnt_all,
        # song_comment_cnt_all,song_share_cnt_all
        meta_song_id = row["song_id"]

        # 1. get audio url
        try:
            # if the input file already contains the url to download
            audio_url = row['audio_url']
        except:
            audio_url = get_audio_url(meta_song_id)
        if not audio_url:
            # don't process empty audio
            warnings.warn(f"Empty audio url for {meta_song_id}")
            continue

        # 2. download audio bytes
        audio_bytes = get_audio_bytes(audio_url)
        if not audio_bytes:
            # don't process empty audio
            warnings.warn(f"Empty audio bytes for {meta_song_id}")
            continue

        # 3. download lyrics_link and add to meta
        duration = len(audio_bytes["npy"]) / 24000
        row["duration"] = duration
        try:
            lyrics_link = row["lyric_link"]
            if lyrics_link:
                lyrics = requests.get(lyrics_link).text
                row["lyrics_from_link"] = lyrics
                lyrics_gt = []
                if lyrics_link.endswith(".lrc"):
                    lyrics_lines = lyrics.split("\n")
                    for lyrics_line in lyrics_lines:
                        lyrics_dict = process_line(lyrics_line, "lrc")
                        if lyrics_dict:
                            lyrics_gt.append(lyrics_dict)
                    # Add end_time
                    for i in range(len(lyrics_gt) - 1):
                        lyrics_gt[i]["end_time"] = lyrics_gt[i + 1]["start_time"] - 1
                    lyrics_gt[-1]["end_time"] = duration * 1000
                elif lyrics_link.endswith(".krc"):
                    lyrics_lines = lyrics.split("\n")
                    for lyrics_line in lyrics_lines:
                        lyrics_dict = process_line(lyrics_line, "krc")
                        if lyrics_dict:
                            lyrics_gt.append(lyrics_dict)
                row["lyrics_gt"] = lyrics_gt
        except:
            row["lyrics_from_link"] = None
            row["lyrics_gt"] = None
        # print(row, audio_bytes["npy"].shape, audio_bytes["npy"].dtype)
        # break
        tar_writer.write({
            "__key__": str(meta_song_id),
            "audio.npy": audio_bytes["npy"],
        })
        idx_writer.write(f"{meta_song_id}\t{json.dumps(row, ensure_ascii=False)}\n".encode())
    tar_writer.close()
    fileobj.flush()
    fileobj.close()
    idx_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio and pack.")
    parser.add_argument("--input_data", type=str, help="Input csv or excel file.")
    parser.add_argument("--output_dir", type=str, help="Output directory to save audio data.")
    args = parser.parse_args()

    input_file = args.input_data
    file_type = input_file.split("/")[-1].split(".")[-1]
    if file_type == "xlsx":
        # excel file
        df = pd.read_excel(input_file, engine='openpyxl')

        # the following are for testing Chinese music file, you should modify your file based on your own input
        # ['song_id', 'song_name', 'display_artist', 'display_album', 'vid',
        # 'clip_id', 'source', '原始音频URL']

        df['audio_url'] = df['原始音频URL']
        df = df.drop(['原始音频URL'], axis=1)

    elif file_type == "csv":
        # csv file
        df = pd.read_csv(input_file)


    def make_chunk(df, chunk_size=1000):
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]


    output = args.output_dir
    # eg: hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/sample_test_ray
    os.system(f"hdfs dfs -mkdir -p {output}")
    tasks = [
        (chunk,
         f"{output}/shard_{i:03d}.tar",
         f"{output}/shard_{i:03d}.tar.index",
        ) for i, chunk in enumerate(make_chunk(df))
    ]
    pool = mp.Pool(80)
    pool.starmap(process_one, tasks)
    pool.close()
    fs = get_filesystem(output)
    url2idx = f"{output}/url2index.txt"
    with fs.open(url2idx, mode="wb") as f:
        for task in tasks:
            _, tar, idx = task
            f.write(f"{tar}\t{idx}\n".encode())
