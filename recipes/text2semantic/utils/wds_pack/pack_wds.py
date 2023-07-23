import argparse
import functools
import logging
import os
from multiprocessing import Manager, Pool, Process

import numpy as np
import pickle
from pydub import AudioSegment

from samantha.dataio.webdataset import ShardWriter

logger = logging.getLogger(__name__)


def writer(queue, output, maxcount, maxsize, resume_shard):
    writer = ShardWriter(
        pattern=output, maxcount=maxcount, maxsize=maxsize, start_shard=resume_shard
    )
    count = resume_shard * maxcount
    logger.info(f"resume from shard {resume_shard}, estimate count is {count}")
    while True:
        try:
            item = queue.get()
            if item is None:
                break
            writer.write(item)
        except Exception as e:
            logger.error(f"write failed with msg {e}")
    writer.close()
    logger.info(f"Finished packing of output pattern {output}")


def read(line):
    x = line.strip().split("|")
    bn_path, text_id_path = x[0], x[1]
    uttid = os.path.splitext(os.path.basename(text_id_path))[0]
    bn = np.load(bn_path)
    text_id = np.load(text_id_path)
    item = {
        "__key__": uttid.replace(".", "-"),
        "uttid": uttid,
        "meta": pickle.dumps({"length": int(x[-1]), "bn": bn, "text_id": text_id}),
    }
    return item


def pool_reader(queue, input, num_reader, start, poolsize):
    lines = open(input, "r", encoding="utf8").readlines()

    if start >= len(lines):
        raise ValueError(
            f"Expecting start less than total lines, but got "
            f"start={start}, total={len(lines)}"
        )

    lines = lines[start:]
    pool = Pool(num_reader)
    for i in range(0, len(lines), poolsize):
        cur_line = lines[i : i + poolsize]
        for item in pool.map(read, cur_line):
            queue.put(item)


def worker_group(input, output, args):

    queue = Manager().Queue(512)
    r = Process(
        target=pool_reader,
        args=(
            queue,
            input,
            args.num_reader,
            args.resume_shard * args.maxcount,
            args.poolsize,
        ),
    )
    r.start()

    w = Process(
        target=writer,
        args=(queue, output, args.maxcount, args.maxsize, args.resume_shard),
    )
    w.start()

    r.join()
    queue.put(None)
    w.join()


def packer(args):
    input_paths, output_paths = args.input, args.output
    if len(output_paths) != len(input_paths):
        raise ValueError("length of output should be same as input.")

    procs = []
    for input, output in zip(args.input, args.output):
        p = Process(target=worker_group, args=(input, output, args))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", nargs="+", type=str, required=True, help="input meta path"
    )
    parser.add_argument(
        "--output", nargs="+", type=str, required=True, help="output pattern"
    )
    parser.add_argument(
        "--maxcount", type=int, default=2048, help="maximum number of records per shard"
    )
    parser.add_argument(
        "--maxsize", type=float, default=1e10, help="maximum size per shard"
    )
    parser.add_argument(
        "--poolsize", type=int, default=1024, help="chunksize of pool map"
    )
    parser.add_argument("--num-reader", type=int, default=4, help="num reader")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="sample rate if data is numpy array",
    )
    parser.add_argument(
        "--resume-shard", type=int, default=0, help="resume sharding dataset"
    )
    parser.add_argument(
        "--split-chunk",
        type=int,
        default=-1,
        help="split the entire dataset with split-chunk per part",
    )
    args = parser.parse_args()
    if args.split_chunk != -1:
        if len(args.input) != len(args.output):
            raise ValueError("length of output should be same as input.")
        new_inputs, new_outputs = [], []
        split_cache = ".split/"
        os.makedirs(split_cache, exist_ok=True)
        for input, output in zip(args.input, args.output):
            lines = open(input, "r", encoding="utf8").readlines()
            for idx, i in enumerate(range(0, len(lines), args.split_chunk)):
                bn = os.path.basename(input)
                new_inputs.append(os.path.join(split_cache, f"{idx}_{bn}"))
                out_list = output.split("/")
                new_outputs.append(
                    "/".join(out_list[:-1] + [f"part_{idx:03d}"] + out_list[-1:])
                )
                with open(new_inputs[-1], "w") as f:
                    for line in lines[i : i + args.split_chunk]:
                        f.write(line)
        args.input = new_inputs
        args.output = new_outputs

    print(args)
    packer(args)
