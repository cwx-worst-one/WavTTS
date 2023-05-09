import multiprocessing as mp
from symbol import testlist_star_expr
import time, argparse, json
from typing import List
import random, json

import numpy as np
import webdataset as wds
from pyarrow.fs import FileSystem
from utils.hdfs_tools import hdfs_mkdir

queue = mp.Queue()


def make_chunk(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def worker():
    while True:
        x = queue.get()
        if x is None:
            break
        i, chunk, args = x
        print(f"Task {i} started, packing {len(chunk)} samples")
        t = time.time()
        output_file = f"{args.hdfs_out_dir}/data_{i:03d}.tar"  # noqa
        fs, _ = FileSystem.from_uri(output_file)
        with fs.open_output_stream(output_file) as fstream:
            writer = wds.TarWriter(fstream)
            for j, sample in enumerate(chunk):
                if args.data_type == 'json':
                    sample_data = json.loads(sample)
                elif args.data_type == 'text':
                    sample_data = {'targets': sample.lower().strip()}
                else:
                    raise NotImplementedError
                writer.write(
                    {
                        "__key__": f"{i * args.sample_per_tar + j:08d}",
                        "sample.pyd": sample_data,
                    }
                )
            writer.close()
        print(f"Task {i} finished in {time.time() - t:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", nargs="+", type=str, required=True, help="input text/json data"
    )
    parser.add_argument(
        "--data_type", type=str, required=True, help="input_data_format, can be json or pure text"
    )
    parser.add_argument(
        "--hdfs_out_dir", type=str, required=True, help="output local/hdfs path pattern"
    )
    parser.add_argument(
        "--sample_per_tar", type=int, default=50000, help="number of samples per tar"
    )
    parser.add_argument(
        "--num_worker", type=int, default=32, help="number of workers for multiprocessing"
    )
    args = parser.parse_args()

    text_samples = []
    for item in args.input_file:
        text_samples.extend(open(item).readlines())

    random.shuffle(text_samples)
    hdfs_mkdir(args.hdfs_out_dir)

    workers = [mp.Process(target=worker) for _ in range(args.num_worker)]
    for w in workers:
        w.start()
    for i, x in enumerate(make_chunk(iter(text_samples), args.sample_per_tar)):
        queue.put((i, x, args))

    for w in workers:
        queue.put(None)

    for w in workers:
        w.join()
