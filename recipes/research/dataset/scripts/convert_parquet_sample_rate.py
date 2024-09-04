import os
import subprocess
from argparse import ArgumentParser
from io import BytesIO
from itertools import accumulate, chain
from time import perf_counter

import pandas as pd
from joblib import Parallel, delayed
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from samantha.utils.hdfs_helper import hdfs_ls


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def wav_bytes_to_mp3_bytes(wav_bytes: bytes) -> bytes:

    mp3_buffer = BytesIO()
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=wav_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

    mp3_buffer.write(stdout)
    return mp3_buffer.getvalue()

def wav_bytes_resample(wav_bytes: bytes, target_sample_rate: int) -> bytes:
    audio_buffer = BytesIO()
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-f",
            "wav",
            "-ar",
            str(target_sample_rate),
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=wav_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

    audio_buffer.write(stdout)
    return audio_buffer.getvalue()


def process_row_group(row_group_no):

    ## re-open the Parquet file, as it cannot be pickled across processes
    fs = get_filesystem(fp)
    stream = fs.open(fp, skip_instance_cache=True)
    parquet_file = ParquetFile(stream)

    columns = None
    group_data = parquet_file.read_row_group(row_group_no, columns=columns)
    group_datas = group_data.to_pandas()
    items = []
    for row in group_datas.iterrows():
        index, values = row
        uttid = values.uttid
        audio = values.audio
        # audio_bytes = wav_bytes_to_mp3_bytes(audio)

        audio_bytes = wav_bytes_resample(audio, TARGET_SAMPLE_RATE)

        items.append(
            {
                "row_group_no": row_group_no,
                "group_index": index,
                "uttid": uttid,
                "audio": audio_bytes,
            }
        )
    return items


if __name__ == "__main__":
    n_workers = 32  # NOTE: size of parquet file * n_workers much fit into RAM!
    TARGET_SAMPLE_RATE = 24000

    # src_dir = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_everynoise_N937k_Lmix/data"
    # target_dir = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_everynoise_N937k_Lmix_24k/data"

    parser = ArgumentParser()
    parser.add_argument("--rank", required=True, type=str)
    parser.add_argument("--world_size", required=True, type=str)
    parser.add_argument('--src_dir', required=True, type=str)
    parser.add_argument('--target_dir', required=True, type=str)

    args = parser.parse_args()

    rank = int(args.rank)
    world_size = int(args.world_size)
    src_dir = args.src_dir
    target_dir = args.target_dir

    ## pip3 install fastparquet

    ## obtain shard lists to process per rank:
    fps = hdfs_ls(src_dir)
    existing_fps = [os.path.basename(fp) for fp in hdfs_ls(target_dir)]
    fps = [fp for fp in fps if os.path.basename(fp) not in existing_fps]
    fps = list(chunks(fps, world_size))
    rank_fps = fps[rank]

    ## process each shard
    for fp in tqdm(rank_fps, desc=f"Rank: {rank} / {world_size}"):
        tik = perf_counter()

        ## open source Parquet file
        fs = get_filesystem(fp)
        stream = fs.open(fp, skip_instance_cache=True)
        parquet_file = ParquetFile(stream)

        items = Parallel(n_jobs=n_workers, prefer="processes")(
            delayed(process_row_group)(row_group)
            for row_group in tqdm(range(parquet_file.num_row_groups))
        )

        items = list(chain(*items))

        df = pd.DataFrame(items)

        ## first sort by row_group, then by group_index, to preserve original order
        df = df.sort_values(by=["row_group_no", "group_index"], ascending=True)

        ## this gets a list of how many items are in each `row_group_no`, which are used as offsets
        n_rows_per_group = df.groupby("row_group_no")["group_index"].count().tolist()
        assert len(n_rows_per_group) == parquet_file.num_row_groups

        ## start positions = [0, cumsum()]
        row_group_offsets = [0, *list(accumulate(n_rows_per_group))[:-1]]

        df = df.drop(["row_group_no", "group_index"], axis=1)

        ## write target Parquet file

        shard_name = os.path.basename(fp)
        target_fp = f"{target_dir}/{shard_name}"
        print(f"Writing to: {target_fp}")
        df.to_parquet(
            target_fp,
            row_group_offsets=row_group_offsets,
            engine="fastparquet",
            compression=None,
            index=None,
        )  # compression=snappy?

        tok = perf_counter()

        print(f"Finished in {tok - tik} seconds")
