import argparse
import io
import multiprocessing as mp
import os
import tarfile
import time

import numpy as np
from slice import slice_mulan_audio

task_queue = mp.Queue()


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
        tar_file, audio_files, hdfs_output_dir, i = task

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
        if hdfs_output_dir is not None:
            cmd = "hdfs dfs -put -f %s %s/ && rm %s" % (
                tar_file,
                hdfs_output_dir,
                tar_file,
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
    parser.add_argument("--drive_id", type=int, default=0, help="bytedrive id")
    parser.add_argument("--n_processes", type=int, default=10, help="num of processes")
    parser.add_argument("--n_workers", type=int, default=10, help="num of workers")
    parser.add_argument("--hdfs_out_dir", default=None, help="hdfs output path")
    parser.add_argument("--shards_dir", help="output shards dir")
    parser.add_argument("--shards_list", help="output shards list file")

    args = parser.parse_args()

    drive_id = args.drive_id
    chunksize = args.chunksize
    shards_dir = args.shards_dir
    n_processes = args.n_processes
    n_workers = args.n_workers

    prefix = f"shards_{drive_id}"

    path = f"/mnt/bd/mmdata-{drive_id}/wav"
    data = list(traverse_path(path))
    chunks = make_chunks(data, chunksize)
    os.makedirs(shards_dir, exist_ok=True)
    shards_list = []

    processes = []
    for i in range(n_processes):
        p = mp.Process(target=worker, args=(n_workers,))
        p.start()
        processes.append(p)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir, "{}_{:06d}.tar".format(prefix, i))
        if args.hdfs_out_dir is None:
            shards_list.append(tar_file)
        else:
            shards_list.append(
                os.path.join(args.hdfs_out_dir, "{}_{:06d}.tar".format(prefix, i))
            )
        task_queue.put((tar_file, chunk, args.hdfs_out_dir, i))

    for _ in range(n_processes):
        task_queue.put(None)

    for p in processes:
        p.join()

    with open(args.shards_list, "w", encoding="utf8") as fout:
        for name in shards_list:
            fout.write(name + "\n")
