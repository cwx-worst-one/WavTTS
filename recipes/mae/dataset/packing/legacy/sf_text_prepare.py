import multiprocessing as mp
import os
import pickle

import pandas as pd


def load_csv(file, chunksize=50000):
    return pd.read_csv(file, chunksize=chunksize, keep_default_na=False)


def process_chunk(chunk):
    # process chunk and return result
    df_dict = {}
    for _, row in chunk.iterrows():
        # meta_song_title,meta_song_author,meta_song_album_name,genre,mood,theme,language
        df_dict[row["meta_song_id"]] = {
            "music_id": row["meta_song_id"],
            "meta_song_title": row["meta_song_title"],
            "meta_song_author": row["meta_song_author"],
            "meta_song_album_name": row["meta_song_album_name"],
            "genre": row["genre"],
            "mood": row["mood"],
            "theme": row["theme"],
            "language": row["language"],
            "data_source": "short_form",
        }
    return df_dict


def parallel_load_and_process(file):
    with mp.Pool(60) as pool:
        chunk_gen = load_csv(file)
        results = pool.map(process_chunk, chunk_gen)
    final_result = {}
    for r in results:
        final_result.update(r)
    return final_result


# Get file from hdfs
meta = (
    "hdfs://harunava/home/byte_speech_sv/mulan/"
    "short_form_example/2000w_music_info_meta.csv"
)
desc = (
    "hdfs://harunava/home/byte_speech_sv/zhongyi.huang/mulan/"
    "long_form_playlist_description.csv"
)

# Check if file exists
if not os.path.exists("2000w_music_info_meta.csv"):
    cmd = f"hdfs dfs -get {meta}"
    print(cmd)
    os.system(cmd)

if not os.path.exists("long_form_playlist_description.csv"):
    cmd = f"hdfs dfs -get {desc}"
    print(cmd)
    os.system(cmd)

result = parallel_load_and_process("2000w_music_info_meta.csv")

playlist = pd.read_csv("long_form_playlist_description.csv", keep_default_na=False)
playlist_dict = {}
for i, row in playlist.iterrows():
    # playlist_name,playlist_description
    playlist_dict[row["music_id"]] = {
        "music_id": row["music_id"],
        "playlist_name": row["playlist_name"],
        "playlist_description": row["playlist_description"],
        "data_source": "playlist",
    }

for k, v in playlist_dict.items():
    if k in result:
        result[k].update(v)
        result[k]["data_source"] = "both"
    else:
        result[k] = v

with open("short_from_and_playlist.pkl", "wb") as f:
    pickle.dump(result, f)
