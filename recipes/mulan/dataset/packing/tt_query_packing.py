import argparse

# import io
import json
import multiprocessing as mp
from time import time

import pandas as pd

# import requests
from pyarrow.fs import FileSystem
from slice import load_audio, slice_audio
from webdataset import TarWriter

# from samantha.dataio.audio_pipeline import AudioPipeline

queue = mp.Queue()

HDFS_BASE = "hdfs://harunava/home/byte_speech_sv/jingsong.gao/tt_query_700k"


def get_text_list():
    with open("tt_query_700K_audio2text.json", "r") as f:
        text_list = json.load(f)
    # convert key to int
    text_list = {int(k): v for k, v in text_list.items()}
    return text_list


def get_tos_mapping():
    df = pd.read_csv("tt_query_700K_auto2url.csv")
    # key is music_id and value is music_id_tos_url
    tos_mapping = {}
    for _, row in df.iterrows():
        tos_mapping[row["music_id"]] = row["music_id_tos_url"]
    return tos_mapping


def make_chunks(data, chunksize):
    """split data into chunks"""
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]


# def download_audio(tos_url):
#     """download audio from tos"""
#     # download audio from tos, replace https with http
#     tos_url = tos_url.replace("https", "http")
#     resp = requests.get(tos_url)
#     buf = io.BytesIO(resp.content)
#     return buf


def packer():
    while True:
        task = queue.get()
        if task is None:
            break
        i, music_ids, tos_urls, querys = task
        print(f"Processing chunk {i}")
        start = time()
        output_path = f"{HDFS_BASE}/tt_query_{i:04d}.tar"
        # parse file system
        output_fs = FileSystem.from_uri(output_path)[0]
        with output_fs.open_output_stream(output_path) as outstream:
            writer = TarWriter(outstream)
            # pipe = (
            #     AudioPipeline(data_iter=tos_urls)
            #     .apply("download", download_audio)
            #     .read_file()
            #     .resample(24000)
            #     .mono()
            #     .numpy(channel_first=True)
            #     .norm()
            #     .int16()
            # )
            total_count = 0
            invalid_count = 0
            for music_id, query, tos_url in zip(music_ids, querys, tos_urls):
                total_count += 1
                try:
                    tos_url = tos_url.replace("https", "http")
                    audio, _ = load_audio(tos_url, "channels_first", 24000, True)
                    for chunk_idx, chunk in enumerate(slice_audio(audio, 24000)):
                        sample = {
                            "__key__": f"{music_id}_{chunk_idx}",
                            "audio.npy": chunk,
                            "meta.json": {
                                "querys": query,
                                "music_id": music_id,
                                "data_source": "tt_query",
                            },
                        }
                        writer.write(sample)
                except Exception:
                    # simply skip corrupted file
                    invalid_count += 1
                    pass
            writer.close()
        print(
            f"Finished chunk {i} in {time() - start:.2f} seconds, "
            f"invalid: {invalid_count}/{total_count}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int, default=0)
    parser.add_argument("end", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=100)

    args = parser.parse_args()

    text_list = get_text_list()
    tos_mapping = get_tos_mapping()
    # filter out non string in tos_mapping
    tos_mapping = {k: v for k, v in tos_mapping.items() if isinstance(v, str)}
    # filter out the ones that are not in tos_mapping
    text_list = {k: v for k, v in text_list.items() if k in tos_mapping}

    # create a bunch of processes
    processes = []
    for i in range(args.num_workers):
        p = mp.Process(target=packer)
        p.start()
        processes.append(p)

    # put the tasks in the queue
    text_keys = list(text_list.keys())
    text_chunks = make_chunks(text_keys, args.chunk_size)
    for i, music_ids in enumerate(text_chunks):
        if i < args.start:
            continue
        if i >= args.end:
            break
        tos_urls = [tos_mapping[v] for v in music_ids]
        querys = [text_list[v] for v in music_ids]
        queue.put((i, music_ids, tos_urls, querys))

    # put the stop signals in the queue
    for p in processes:
        queue.put(None)

    # wait for all the processes to finish
    for p in processes:
        p.join()
