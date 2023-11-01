import argparse
import json
import multiprocessing as mp
import os
from io import BytesIO

from scipy.io.wavfile import write
from tqdm import tqdm

from samantha.dataio.parquet.writer import IndexShardWriter
from samantha.dataio.webdataset.extension import IndexedWebDataset


def worker(q, args, part):
    writer = IndexShardWriter(args.output, partitions=[f"part={part:05d}"])
    t = tqdm()
    while True:
        task = q.get()
        if task is None:
            break
        tar, idx = task
        dataset = IndexedWebDataset({tar: idx}).decode()
        for sample in dataset:
            npy = sample["audio.npy"]
            uttid = sample["__key__"]
            meta = sample["__index_data__"]
            duration = len(npy) / 24000  # seconds
            meta["duration"] = duration
            # convert to wav
            io = BytesIO()
            write(io, 24000, npy)
            wav = io.getvalue()
            data_dict = {"uttid": uttid, "audio": wav}
            item_dict = {
                "uttid": uttid,
                "meta": json.dumps(meta, ensure_ascii=False),
                "text": "",
            }
            writer.write(data_dict, item_dict)
            t.update()
    writer.close()
    print(f"Worker {part} finished, total {t.n} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url2idx", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_partitions", type=int, default=80)

    args = parser.parse_args()

    procs = []
    q = mp.Queue()
    for i in range(args.num_partitions):
        p = mp.Process(target=worker, args=(q, args, i))
        p.start()
        procs.append(p)
    result = os.popen(f"hdfs dfs -cat {args.url2idx}").readlines()
    for line in result:
        ary = line.strip().split("\t")
        q.put((ary[0], ary[1]))

    for i in range(args.num_partitions):
        q.put(None)

    for p in procs:
        p.join()
