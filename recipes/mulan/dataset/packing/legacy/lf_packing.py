import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile
from time import time

import numpy as np
import pandas as pd
from slice import slice_mulan_audio
from utils import make_chunks, run_cmd, run_cmd_async

packer_queue = mp.Queue(maxsize=100)


def get_meta_pqs():
    # Get parquet files
    cmd = (
        "hdfs dfs -ls hdfs://harunava/home/byte_speech_sv/"
        "mulan/long_form_dataset/intro_groupby_music_id/"
    )
    pq_list = os.popen(cmd).read().split("\n")[:-1]
    pq_list = [x.split(" ")[-1] for x in pq_list if x.endswith(".parquet")]
    return pq_list


def packer(n_readers):
    pool = mp.Pool(n_readers)
    while True:
        task = packer_queue.get()
        if task is None:
            break
        chunk_id, chunk = task
        print(f"packing {chunk_id:04d}...")
        t = time()

        # Construct audio file list from music_id
        audio_files = [f"/mnt/bn/mm-data/tt_video_music/wav/{c[0]}.wav" for c in chunk]
        slices = pool.map(slice_mulan_audio, audio_files)
        valid_audio_count = 0
        slice_count = 0
        print(f"read audio {chunk_id:04d} in {time() - t:.2f} seconds")

        # Create tar files
        t = time()
        tar_file = f"shards_{chunk_id:04d}.tar"
        audio_tar = tarfile.open("packed/audio/" + tar_file, "w")
        text_tar = tarfile.open("packed/text/" + tar_file, "w")

        for row, audio_chunks in zip(chunk, slices):
            if len(audio_chunks) == 0:
                continue
            valid_audio_count += 1
            slice_count += len(audio_chunks)
            music_id, item_titles, item_country_names = row
            meta = {
                "item_titles": item_titles.tolist(),
                "item_country_names": item_country_names.tolist(),
                "music_id": music_id,
                "data_source": "long_form",
            }
            for s in audio_chunks:
                key = s["__key__"]
                # Pack audio
                stream = io.BytesIO()
                np.lib.format.write_array(stream, s["chunk.npy"])
                data = stream.getvalue()
                tarinfo = tarfile.TarInfo(key + ".audio.npy")
                tarinfo.size = len(data)
                audio_tar.addfile(tarinfo, fileobj=io.BytesIO(data))
                # Pack text
                stream = io.BytesIO()
                stream.write(json.dumps(meta).encode())
                stream.seek(0)
                tarinfo = tarfile.TarInfo(key + ".meta.json")
                tarinfo.size = len(stream.getbuffer())
                text_tar.addfile(tarinfo, fileobj=stream)

        # Close tar files
        audio_tar.close()
        text_tar.close()

        print(
            f"packed {chunk_id:04d}: {time() - t:.2f}s, "
            f"{valid_audio_count} valid audios, {slice_count} slices"
        )

        # Upload tar files to hdfs
        t = time()
        hdfs_out_dir = "hdfs://harunava/home/byte_speech_sv/mulan/long_form_uio"
        run_cmd_async(
            f"hdfs dfs -put -f packed/audio/{tar_file} "
            f"{hdfs_out_dir}/audio_shards/ && rm packed/audio/{tar_file}"
        )
        run_cmd_async(
            f"hdfs dfs -put -f packed/text/{tar_file} "
            f"{hdfs_out_dir}/text_shards/ && rm packed/text/{tar_file}"
        )

        print(f"uploaded {chunk_id:04d} in {time() - t:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100000)
    parser.add_argument("--n_packers", type=int, default=10)
    parser.add_argument("--n_readers", type=int, default=10)
    args = parser.parse_args()

    pq_list = get_meta_pqs()

    os.makedirs("meta", exist_ok=True)
    os.makedirs("packed/audio", exist_ok=True)
    os.makedirs("packed/text", exist_ok=True)

    # Create packer processes
    packers = []
    for _ in range(args.n_packers):
        p = mp.Process(target=packer, args=(args.n_readers,))
        p.start()
        packers.append(p)

    chunk_id = 0
    for pq in pq_list:
        # Get csv from hdfs
        run_cmd(f"hdfs dfs -get {pq} meta/")
        # Read csv
        df = pd.read_parquet(f"meta/{pq.split('/')[-1]}").values.tolist()
        # Delete csv
        os.remove(f"meta/{pq.split('/')[-1]}")
        # Split csv into chunks
        # df header: [item_id,music_id,item_title,item_country_name]
        chunks = make_chunks(df, 1000)
        for chunk in chunks:
            if args.start <= chunk_id < args.end:
                packer_queue.put((chunk_id, chunk))
            chunk_id += 1

    # Send stop signal to packers
    for _ in range(args.n_packers):
        packer_queue.put(None)

    # Wait for packers to finish
    for p in packers:
        p.join()
