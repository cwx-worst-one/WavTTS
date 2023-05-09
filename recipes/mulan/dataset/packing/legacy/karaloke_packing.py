import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile
import time
from pathlib import Path

import numpy as np
from slice import slice_mulan_audio

task_queue = mp.Queue()


def load_dataset(dataset_path, data_type="wav"):
    path_list = sorted([path for path in Path(dataset_path).rglob(f"*.{data_type}")])

    return path_list


def traverse_path(path: str):
    """traverse the path and return all files"""
    for root, dirs, files in os.walk(path):
        for file in files:
            yield os.path.join(root, file)


def make_chunks(data, chunksize):
    """split data into chunks"""
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]


def worker(n_workers):
    pool = mp.Pool(n_workers)
    while True:
        task = task_queue.get()
        if task is None:
            break
        t = time.time()
        valid_audio_count = 0
        slice_count = 0
        tar_file, audio_files, hdfs_out_dir, i = task
        # pack audio
        with tarfile.open(tar_file, "w") as tar:
            slices = pool.map(slice_mulan_audio, audio_files)
            for chunks in slices:
                if len(chunks) > 0:
                    valid_audio_count += 1
                    slice_count += len(chunks)
                for chunk in chunks:
                    stream = io.BytesIO()
                    np.lib.format.write_array(stream, chunk["chunk.npy"])
                    data = stream.getvalue()
                    tarinfo = tarfile.TarInfo(chunk["__key__"] + ".audio.npy")
                    tarinfo.size = len(data)
                    tar.addfile(tarinfo, fileobj=io.BytesIO(data))
        # pack text
        with tarfile.open(tar_file.replace("audio", "text"), "w") as tar:
            slices = pool.map(slice_mulan_audio, audio_files)
            for chunks in slices:
                for chunk in chunks:
                    part_name = chunk["__key__"]
                    split_idx = part_name.rfind("_")
                    music_id = part_name[:split_idx]
                    meta = {}
                    meta["data_source"] = "karaoke"
                    meta["music_id"] = music_id
                    stream = io.BytesIO()
                    stream.write(json.dumps(meta).encode())
                    stream.seek(0)
                    tarinfo = tarfile.TarInfo(chunk["__key__"] + ".meta.json")
                    tarinfo.size = len(stream.getbuffer())
                    tar.addfile(tarinfo, fileobj=stream)
        if hdfs_out_dir is not None:
            # upload audio
            cmd = "hdfs dfs -put -f %s %s/ && rm %s" % (
                tar_file,
                f"{hdfs_out_dir}/audio_shards/",
                tar_file,
            )
            print(cmd)
            os.system(cmd)
            # upload text
            cmd = "hdfs dfs -put -f %s %s/ && rm %s" % (
                tar_file.replace("audio", "text"),
                f"{hdfs_out_dir}/text_shards/",
                tar_file.replace("audio", "text"),
            )
            print(cmd)
            os.system(cmd)
        print(
            f"tar file {tar_file} created in {time.time() - t:.2f}s, "
            f"valid audio count: {valid_audio_count}, "
            f"slice count: {slice_count}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunksize", type=int, default=1000, help="num samples per chunk"
    )

    parser.add_argument("--n_processes", type=int, default=10, help="num of processes")
    parser.add_argument("--n_workers", type=int, default=10, help="num of workers")
    parser.add_argument(
        "--hdfs_out_dir",
        default="/home/byte_speech_sv/mulan/karaoke_uio",
        help="hdfs output path",
    )
    parser.add_argument(
        "--shards_dir", default="karaoke/audio_shards", help="output shards dir"
    )
    parser.add_argument("--shards_list", help="output shards list file")

    args = parser.parse_args()

    chunksize = args.chunksize
    shards_dir = args.shards_dir
    n_processes = args.n_processes
    n_workers = args.n_workers

    prefix = "shards"

    data_dirpath = "../audio"
    data_paths = load_dataset(data_dirpath, data_type="mp3")

    data_chunks = make_chunks(data_paths, chunksize)
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(shards_dir.replace("audio", "text"), exist_ok=True)
    shards_list = []

    processes = []
    for i in range(n_processes):
        p = mp.Process(target=worker, args=(n_workers,))
        p.start()
        processes.append(p)
    for i, data_chunk in enumerate(data_chunks):
        tar_file = os.path.join(args.shards_dir, "{}_{:06d}.tar".format(prefix, i))
        if args.hdfs_out_dir is None:
            shards_list.append(tar_file)
        else:
            shards_list.append(
                os.path.join(args.hdfs_out_dir, "{}_{:06d}.tar".format(prefix, i))
            )
        task_queue.put((tar_file, data_chunk, args.hdfs_out_dir, i))

    for _ in range(n_processes):
        task_queue.put(None)

    for p in processes:
        p.join()

    with open(args.shards_list, "w", encoding="utf8") as fout:
        for name in shards_list:
            fout.write(name + "\n")
