import argparse
from io import BytesIO
import json
import multiprocessing as mp
import os

from scipy.io.wavfile import write
from tqdm import tqdm

from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.parquet.writer import IndexShardWriter


def count_one(idx):
    if idx.startswith("hdfs://"):
        result = os.popen(f"hdfs dfs -cat {idx}").read().split("\n")
    else:
        result = open(idx, "r").read().split("\n")
    return len(result)


def try_get_field_and_rename(meta, new_key, possible_keys):
    for key in possible_keys:
        if key in meta:
            meta[new_key] = meta[key]
            return
    # if not found, set to None
    meta[new_key] = None


def worker(args, part, idxes):
    sr = args.sr
    feats = ["mss"] if args.mss else None
    writer = IndexShardWriter(
        args.output, partitions=[f"part={part:05d}"], row_group_size=2, feats=feats
    )
    t = tqdm()
    for idx in idxes:
        tar = idx2tar[idx]
        dataset = IndexedWebDataset({tar: idx}).decode()
        for sample in dataset:
            uttid = sample["__key__"]
            meta = sample["__index_data__"]
            data_dict = {"uttid": uttid}
            item_dict = {"uttid": uttid, "text": ""}
            # if metadata in meta, lift it
            if "metadata" in meta:
                metadata = meta["metadata"]
                for key, value in metadata.items():
                    if key in meta:
                        continue
                    meta[key] = value
                del meta["metadata"]
            if args.mss:
                # try get full from full.npy or audio.npy
                full = sample["full.npy"] if "full.npy" in sample else sample["audio.npy"]
                if len(full.shape) == 2:
                    full = full[0]
                io = BytesIO()
                write(io, sr, full)
                wav = io.getvalue()
                data_dict["audio"] = wav
                meta["duration"] = len(full) / sr # seconds

                # add vocal/acc to mss feat
                mss_dict = {"uttid": uttid}
                for key in ["vocal", "acc"]:
                    npy = sample[f"{key}.npy"]
                    # convert to wav
                    io = BytesIO()
                    write(io, sr, npy)
                    wav = io.getvalue()
                    mss_dict[key] = wav
                feat_dict = {"mss": mss_dict}
            else:
                npy = sample["audio.npy"]
                if len(npy.shape) == 2:
                    npy = npy[0]
                io = BytesIO()
                write(io, sr, npy)
                wav = io.getvalue()
                data_dict["audio"] = wav
                meta["duration"] = len(npy) / sr # seconds
                feat_dict = None
            # try get genre, theme, mood, language
            try_get_field_and_rename(meta, "genre", ["genre", "first_genre", "merge_genre", "genres"])
            try_get_field_and_rename(meta, "theme", ["theme", "merge_theme"])
            try_get_field_and_rename(meta, "mood", ["mood", "merge_mood"])
            try_get_field_and_rename(meta, "language", ["language", "merge_language", "final_language", "meta_song_language"])
            item_dict["meta"] = json.dumps(meta, ensure_ascii=False)
            writer.write(data_dict, item_dict, feat_dict)
            t.update()
    writer.close()
    print(f"Worker {part} finished, total {t.n} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url2idx", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_partitions", type=int, default=80)
    parser.add_argument("--mss", action="store_true")
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--add_number_suffix", action="store_true")
    parser.add_argument("--remove_old", action="store_true")

    args = parser.parse_args()

    url2idx = args.url2idx
    if url2idx.startswith("hdfs://"):
        if url2idx.endswith("/"):
            result = []
            for p in os.popen(f"hdfs dfs -ls {url2idx}"):
                if p and "hdfs://" in p:
                    p = p.split()[-1]
                    result += os.popen(f"hdfs dfs -cat {p}").readlines()
        else:
            result = os.popen(f"hdfs dfs -cat {url2idx}").readlines()
    else:
        result = open(url2idx, "r").readlines()
    idxes = []
    idx2count = {}
    idx2tar = {}
    for line in result:
        tar, idx = line.strip().split("\t")
        idxes.append(idx)
        idx2tar[idx] = tar

    pool = mp.Pool(80)
    for idx, count in zip(idxes, pool.map(count_one, idxes)):
        idx2count[idx] = count
    pool.close()
    
    total_count = sum(idx2count.values())
    sample_per_partition = total_count // args.num_partitions
    print(f"Total {total_count} samples, ~ {sample_per_partition} samples per partition")

    # convert total count to _N<num_samples> suffix
    if args.add_number_suffix:
        # check if output endswith _N
        last_suffix = args.output.split("_")[-1]
        if last_suffix.startswith("N"):
            args.output = args.output[: -len(last_suffix) - 1]
        if total_count >= 2000000:
            total_count = f"{total_count // 1000000}m"
        elif total_count >= 2000:
            total_count = f"{total_count // 1000}k"
        else:
            total_count = str(total_count)
        args.output = args.output + f"_N{total_count}"
    print(f"Output to {args.output}")

    if args.remove_old:
        os.system(f"hdfs dfs -rm -r {args.output}")

    # split idxes into partitions
    idxes_per_partition = []
    cur_idxes = []
    cur_count = 0
    for idx in idxes:
        cur_idxes.append(idx)
        cur_count += idx2count[idx]
        if cur_count >= sample_per_partition:
            idxes_per_partition.append(cur_idxes)
            cur_idxes = []
            cur_count = 0
    if cur_idxes:
        idxes_per_partition.append(cur_idxes)
    
    # start workers
    procs = []
    for i, idxes in enumerate(idxes_per_partition):
        p = mp.Process(target=worker, args=(args, i, idxes))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
