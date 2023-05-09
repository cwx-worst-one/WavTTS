import argparse
import json
import multiprocessing as mp
from time import time

import numpy as np
from pyarrow.fs import FileSystem
from slice import slice_audio
from webdataset import TarWriter

queue = mp.Queue()

HDFS_BASE = "hdfs://harunava/home/byte_speech_sv/jingsong.gao/mcc_n2m"


def get_noise2music_list():
    # mcc_music_to_noise2music_captions.json
    with open("mcc_music_to_noise2music_captions.json", "r") as f:
        noise2music_list = json.load(f)
    return noise2music_list


def get_bytedrive_mapping():
    # mcc_music_to_bytedrive_path.json
    with open("mcc_music_to_bytedrive_path.json", "r") as f:
        bytedrive_mapping = json.load(f)
    return bytedrive_mapping


def make_chunks(data, chunksize):
    """split data into chunks"""
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]


def load_npy_and_slice(npy_path):
    """load npy and slice it to 30 second clips"""
    audio = np.load(npy_path)
    # convert (T,) to (1, T)
    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]
    sr = 24000
    return slice_audio(audio, sr)


def packer():
    while True:
        task = queue.get()
        if task is None:
            break
        i, vals = task
        print(f"Processing chunk {i}")
        start = time()
        output_path = f"{HDFS_BASE}/mcc_n2m_{i:02d}.tar"
        # parse file system
        output_fs = FileSystem.from_uri(output_path)[0]
        with output_fs.open_output_stream(output_path) as outstream:
            writer = TarWriter(outstream)
            for meta_id, n2m in vals:
                # split meta_id into "<clip_id>_<meta_song_id>"
                clip_id, meta_song_id = meta_id.split("_")
                # get the bytedrive path
                bytedrive_path = bytedrive_mapping[meta_id]
                # load the npy
                audio = load_npy_and_slice(bytedrive_path)
                for chunk_idx, chunk in enumerate(audio):
                    # create the new dict
                    sample = {
                        "__key__": f"{meta_id}_{chunk_idx}",
                        "audio.npy": chunk,
                        "meta.json": {
                            "n2m": n2m,
                            "music_id": int(meta_song_id),
                            "clip_id": int(clip_id),
                            "data_source": "mcc_gpt_generated",
                        },
                    }
                    # write the sample to the tarfile
                    writer.write(sample)
            writer.close()
        print(f"Finished chunk {i} in {time() - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int, default=0)
    parser.add_argument("end", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=20)

    args = parser.parse_args()

    noise2music_list = get_noise2music_list()
    bytedrive_mapping = get_bytedrive_mapping()

    # create a bunch of processes
    processes = []
    for i in range(args.num_workers):
        p = mp.Process(target=packer)
        p.start()
        processes.append(p)

    # put the tasks in the queue
    noise2music_tuple = list(noise2music_list.items())
    n2m_chunks = make_chunks(noise2music_tuple, args.chunk_size)
    for i, vals in enumerate(n2m_chunks):
        if i < args.start:
            continue
        if i > args.end:
            break
        queue.put((i, vals))

    # put the stop signals in the queue
    for p in processes:
        queue.put(None)

    # wait for all the processes to finish
    for p in processes:
        p.join()
