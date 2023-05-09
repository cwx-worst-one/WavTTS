import io
import json
import multiprocessing as mp
import os
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np
from slice import STR_CH_FIRST, load_audio

packing_queue = mp.Queue()


def read_audio(audio_path):
    x, sr = load_audio(
        audio_path, ch_format=STR_CH_FIRST, sample_rate=24000, downmix_to_mono=True
    )
    if x is None:
        return None
    # padding or truncating to 10 seconds
    # x.shape (1, 240000)
    if x.shape[1] < 240000:
        x = np.pad(x, ((0, 0), (0, 240000 - x.shape[1])), "constant")
    else:
        x = x[:, :240000]
    filename = Path(audio_path).stem
    return {"__key__": filename, "chunk.npy": (x * 32768).astype(np.int16)}


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def make_chunks(data, chunksize):
    """split data into chunks"""
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]


def packing(part_id, n_worker):
    pool = mp.Pool(n_worker)
    meta_dict = pickle.load(open("audioset/audioset_meta.pkl", "rb"))
    while True:
        # Get the next task
        task = packing_queue.get()
        if task is None:
            # Poison pill means shutdown
            print("Packer: Exiting")
            break
        # Unpack the task
        files, i = task
        # Read the data
        audios = []
        print(f"Reading chunk {i:03d}...")
        t = time.time()
        audios = pool.map(read_audio, files)
        audios = [a for a in audios if a is not None]
        print(f"Read chunk {i:03d} in {time.time() - t:.2f} seconds")
        t = time.time()
        # Pack the data
        with tarfile.open(
            f"audioset/packed/audio/shards_{part_id}_{i:03d}.tar", "w"
        ) as tar:
            for _, audio in enumerate(audios):
                stream = io.BytesIO()
                np.lib.format.write_array(stream, audio["chunk.npy"])
                data = stream.getvalue()
                tarinfo = tarfile.TarInfo(audio["__key__"] + ".audio.npy")
                tarinfo.size = len(data)
                tar.addfile(tarinfo, fileobj=io.BytesIO(data))
        with tarfile.open(
            f"audioset/packed/text/shards_{part_id}_{i:03d}.tar", "w"
        ) as tar:
            for j, audio in enumerate(audios):
                # Get meta
                music_id = int(f"1{part_id}{i:03d}{j:04d}")
                meta = {
                    "tags": meta_dict[audio["__key__"][1:]],
                    "data_source": "audioset",
                    "music_id": music_id,
                }
                stream = io.BytesIO()
                stream.write(json.dumps(meta).encode())
                stream.seek(0)
                tarinfo = tarfile.TarInfo(audio["__key__"] + ".meta.json")
                tarinfo.size = len(stream.getbuffer())
                tar.addfile(tarinfo, fileobj=stream)
        print(f"Packed chunk {i:03d} in {time.time() - t:.2f} seconds")
        # Upload to HDFS
        t = time.time()
        hdfs_out_dir = "hdfs://harunava/home/byte_speech_sv/mulan/audioset_uio"
        run_cmd(
            f"hdfs dfs -put -f audioset/packed/audio/shards_{part_id}_{i:03d}.tar"
            f" {hdfs_out_dir}/audio_shards/"
        )
        run_cmd(
            f"hdfs dfs -put -f audioset/packed/text/shards_{part_id}_{i:03d}.tar"
            f" {hdfs_out_dir}/text_shards/"
        )
        os.remove(f"audioset/packed/audio/shards_{part_id}_{i:03d}.tar")
        os.remove(f"audioset/packed/text/shards_{part_id}_{i:03d}.tar")
        print(f"Uploaded chunk {i:03d} in {time.time() - t:.2f} seconds")


hdfs_base = "hdfs://harunava/home/byte_speech_sv/mulan/audioset/zip_audios/"

if __name__ == "__main__":

    os.makedirs("audioset/audio", exist_ok=True)
    os.makedirs("audioset/packed/audio", exist_ok=True)
    os.makedirs("audioset/packed/text", exist_ok=True)
    # FIXME: multiprocess will speed up downloading
    # part_id = 0
    # run_cmd(f"hdfs dfs -get {hdfs_base}balanced_train_segments.zip audioset/")
    # run_cmd(f"hdfs dfs -get {hdfs_base}eval_segments.zip audioset/")
    # run_cmd(f"unzip -jq audioset/balanced_train_segments.zip -d audioset/audio/")
    # run_cmd(f"unzip -jq audioset/eval_segments.zip -d audioset/audio/")
    # run_cmd(f"rm audioset/balanced_train_segments.zip audioset/eval_segments.zip")
    # part 1 -> range(0, 10)
    # part 2 -> range(10, 20)
    # part 3 -> range(20, 30)
    # part 4 -> range(30, 41)
    part_id = 4
    for i in range(30, 41):
        run_cmd(
            f"hdfs dfs -get {hdfs_base}unbalanced_train_segments"
            f"/unbalanced_train_segments_part{i:02d}.zip audioset/"
        )
        run_cmd(
            f"unzip -jq audioset/unbalanced_train_segments_part{i:02d}.zip"
            f" -d audioset/audio/"
        )
        run_cmd(f"rm audioset/unbalanced_train_segments_part{i:02d}.zip")

    # List all the files
    files = os.listdir("audioset/audio")
    files = ["audioset/audio/" + f for f in files if f.endswith(".wav")]

    # Make chunks
    chunk_size = 1000
    chunks = make_chunks(files, chunk_size)

    # Start the packer
    n_packer = 10
    n_worker = 10
    packers = []
    for i in range(n_packer):
        p = mp.Process(target=packing, args=(part_id, n_worker))
        p.start()
        packers.append(p)

    for i, files in enumerate(chunks):
        packing_queue.put((files, i))

    # Tell child processes to stop
    for i in range(n_packer):
        packing_queue.put(None)
