import argparse
import io
import json
import multiprocessing as mp
import os
import pickle
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
        tar_file, audio_files, audio_ids, hdfs_output_dir, i, meta_dict = task
        # pack audio
        with tarfile.open(tar_file, "w") as tar:
            slices = pool.starmap(slice_mulan_audio, list(zip(audio_files, audio_ids)))
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
            slices = pool.starmap(slice_mulan_audio, list(zip(audio_files, audio_ids)))
            for chunks in slices:
                for chunk in chunks:
                    music_id = chunk["__key__"].split("_")[0]
                    meta = meta_dict[music_id]
                    meta["data_source"] = "ecals"
                    meta["music_id"] = music_id
                    stream = io.BytesIO()
                    stream.write(json.dumps(meta).encode())
                    stream.seek(0)
                    tarinfo = tarfile.TarInfo(chunk["__key__"] + ".meta.json")
                    tarinfo.size = len(stream.getbuffer())
                    tar.addfile(tarinfo, fileobj=stream)
        if hdfs_output_dir is not None:
            # upload audio
            cmd = "hdfs dfs -put -f %s %s/ && rm %s" % (
                tar_file,
                f"{hdfs_output_dir}/audio_shards/",
                tar_file,
            )
            print(cmd)
            os.system(cmd)
            # upload text
            cmd = "hdfs dfs -put -f %s %s/ && rm %s" % (
                tar_file.replace("audio", "text"),
                f"{hdfs_output_dir}/text_shards/",
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
        default="/home/byte_speech_sv/mulan/ecals_uio",
        help="hdfs output path",
    )
    parser.add_argument(
        "--shards_dir", default="ecals/audio_shards", help="output shards dir"
    )
    parser.add_argument("--shards_list", help="output shards list file")

    args = parser.parse_args()

    chunksize = args.chunksize
    shards_dir = args.shards_dir
    n_processes = args.n_processes
    n_workers = args.n_workers

    prefix = "shards"

    data_path = "../dataset"
    msd_to_id = pickle.load(
        open(os.path.join(data_path, "lastfm_annotation", "MSD_id_to_7D_id.pkl"), "rb")
    )
    id_to_path = pickle.load(
        open(os.path.join(data_path, "lastfm_annotation", "7D_id_to_path.pkl"), "rb")
    )

    annotation = json.load(
        open(os.path.join(data_path, "ecals_annotation", "annotation.json"), "r")
    )
    list_of_label = json.load(
        open(os.path.join(data_path, "ecals_annotation", "ecals_tags.json"), "r")
    )
    tag_to_idx = {i: idx for idx, i in enumerate(list_of_label)}

    track_split = json.load(
        open(os.path.join(data_path, "ecals_annotation", "ecals_track_split.json"), "r")
    )
    train_track = track_split["train_track"] + track_split["extra_track"]

    data_dicts = [annotation[i] for i in train_track]
    audio_dir = "../msd/media/bach2/dataset/MSD/songs"
    data = []
    audio_id = []
    for item in data_dicts:
        audio_path = id_to_path[msd_to_id[item["track_id"]]]
        data.append(f"{audio_dir}/{audio_path}")
        audio_id.append(item["track_id"])

    meta_dict = {}
    for item in data_dicts:
        meta_dict[item["track_id"]] = {
            "title": str(item["title"]),
            "artist_name": str(item["artist_name"]),
            "year": str(item["year"]),
            "tag": str(item["tag"]),
        }

    data_chunks = make_chunks(data, chunksize)
    audio_id_chunks = make_chunks(audio_id, chunksize)
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(shards_dir.replace("audio", "text"), exist_ok=True)
    shards_list = []

    processes = []
    for i in range(n_processes):
        p = mp.Process(target=worker, args=(n_workers,))
        p.start()
        processes.append(p)
    for i, (data_chunk, audio_id_chunk) in enumerate(zip(data_chunks, audio_id_chunks)):
        tar_file = os.path.join(args.shards_dir, "{}_{:06d}.tar".format(prefix, i))
        if args.hdfs_out_dir is None:
            shards_list.append(tar_file)
        else:
            shards_list.append(
                os.path.join(args.hdfs_out_dir, "{}_{:06d}.tar".format(prefix, i))
            )
        task_queue.put(
            (tar_file, data_chunk, audio_id_chunk, args.hdfs_out_dir, i, meta_dict)
        )

    for _ in range(n_processes):
        task_queue.put(None)

    for p in processes:
        p.join()

    with open(args.shards_list, "w", encoding="utf8") as fout:
        for name in shards_list:
            fout.write(name + "\n")
