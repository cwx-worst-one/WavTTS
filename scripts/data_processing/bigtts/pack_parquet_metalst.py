import io
import json
import logging
import multiprocessing.util
import os
import sys
import time
import uuid
from multiprocessing import Manager
from multiprocessing.pool import Pool

import ffmpeg
import numpy as np
from scipy.io.wavfile import write

from samantha.dataio.parquet.writer import ShardWriter
from samantha.utils.watch import elapsed_time

# for randomizing manager server port
multiprocessing.util.abstract_sockets_supported = False
# for multiple filesystem instance
multiprocessing.set_start_method("spawn", force=True)


logger = logging.getLogger(__name__)


def ffmpeg_read_audio(audio_bin, sample_rate=24000):
    seg_bin, _ = (
        ffmpeg.input("pipe:")
        .output("pipe:", loglevel="error", format="s16le", ar=sample_rate)
        .run(input=audio_bin, quiet=True)
    )
    return np.frombuffer(seg_bin, dtype="int16")


def process_one(q1, q2, min_dur, max_dur, sr):
    while True:
        item_t = q1.get()
        if item_t is None:
            break
        data = item_t
        audio = data["bin"]
        key = f"{data['speaker_id']}-{data['uttid']}"
        try:
            wav = ffmpeg_read_audio(audio, sr)
            if wav.size == 0:
                continue
        except Exception as e:
            logger.warning(f"ffmpeg error with msg {e}")
            continue

        text = data["text"]
        meta = {}
        for k in ["debug", "mos", "snr"]:
            if k in data:
                meta[k] = data[k]

        for k in ["speaker_id", "labels"]:
            if k not in data:
                logger.warning(f"no {k} found in {key}")
                break
            meta[k] = data[k]
        else:
            # write to pq
            meta["duration"] = wav.shape[0] / sr
            bytes_io = io.BytesIO()
            write(bytes_io, sr, wav)
            bytes_io.seek(0)
            oitem = {
                "uttid": key,
                "audio": bytes_io.read(),
                "text": text,
                "meta": json.dumps(meta, ensure_ascii=False),
            }
            q2.put((meta["duration"], oitem))


def r(q1, key, text, wav_path, lab_path):
    speaker_id, uttid = key.split("/")
    bin = open(wav_path, "rb").read()
    labels = open(lab_path, "r", encoding="utf8").read()
    item = {
        "mos": "5",
        "snr": "10",
        "debug": "null",
        "speaker_id": speaker_id,
        "labels": labels,
        "bin": bin,
        "text": text,
        "uttid": uttid,
    }
    q1.put(item)


PLACEHOLDER = "__placeholder__"


class Consumer:
    def __init__(
        self, output_pattern, min_dur, max_dur, verbose=False, row_group_size=64
    ):
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.verbose = verbose
        self._prefix = output_pattern.split(PLACEHOLDER)[0]
        self.data_writer = ShardWriter(
            output_pattern=output_pattern.replace(PLACEHOLDER, "data"),
            maxcount=2048,
            row_group_size=row_group_size,
        )
        self.meta_writer = ShardWriter(
            output_pattern=output_pattern.replace(PLACEHOLDER, "index_1"),
            maxcount=2048,
            row_group_size=row_group_size,
            need_row_group_no=True,
        )

    def write(self, item):
        duration, data = item
        if self.min_dur < duration <= self.max_dur:
            total = self.data_writer.write(data)
            meta = {
                "uttid": data["uttid"],
                "text": data["text"],
                "meta": data["meta"],
                "data_file": self.data_writer.filename.replace(self._prefix, "../../"),
            }
            self.meta_writer.write(meta)
            if self.verbose and total is not None:
                logger.info(f"{self} writed {total=}")

    def close(self):
        self.data_writer.close()
        self.meta_writer.close()

    def __repr__(self):
        return f"Consumer(min_dur={self.min_dur}, max_dur={self.max_dur})"


@elapsed_time
def main(args, M):

    part = args.base_part + int(os.getenv("ARNOLD_ID", 0))
    for idx, meta_file in enumerate(args.meta_files):
        part_name = f"{part+idx:05d}"
        if args.part_names is not None:
            assert len(args.part_names) == len(args.meta_files)
            part_name = args.part_names[idx]
        minumal_dur, maximal_dur = sys.maxsize, -1
        consumers = []
        for basename in args.basenames:
            duration = None
            for item in basename.split("_"):
                if item.startswith("D"):
                    duration = item[1:]
            if duration is None:
                min_dur, max_dur = -1, 10000000
            else:
                assert "-" in duration
                min_dur, max_dur = [int(e) for e in duration.split("_")]
            minumal_dur = min(min_dur, minumal_dur)
            maximal_dur = max(max_dur, maximal_dur)
            output_pattern = f"{args.basedir}/{basename}/{PLACEHOLDER}/part={part_name}/shard-%05d.parquet"  # noqa
            consumers.append(
                Consumer(
                    output_pattern=output_pattern,
                    min_dur=min_dur,
                    max_dur=max_dur,
                    row_group_size=args.row_group_size,
                )
            )

        q1 = M.Queue(5120)
        q2 = M.Queue(5120)

        r_pool = Pool(args.num_reader)

        with open(meta_file, "r", encoding="utf-8") as fi:
            for line in fi:
                key, text, wav_path, lab_path = line.strip().split("|")

                r_pool.apply_async(func=r, args=(q1, key, text, wav_path, lab_path))

        num_processor = args.num_processor
        process_pool = Pool(num_processor)
        for _ in range(num_processor):
            process_pool.apply_async(
                func=process_one, args=(q1, q2, minumal_dur, maximal_dur, 24_000)
            )

        while True:
            try:
                item = q2.get(timeout=20)
                if item is None:
                    continue
                for consumer in consumers:
                    consumer.write(item)
            except Exception as e:
                logger.warning(f"finished with exception [{e}]")
                break

        for _ in range(num_processor):
            q1.put(None)

        process_pool.close()
        r_pool.close()
        r_pool.join()
        process_pool.join()

        for consumer in consumers:
            consumer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basedir",
        default="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/data/bigtts",  # noqa
        type=str,
        help="output base directory",
    )
    parser.add_argument(
        "--basenames",
        default=uuid.uuid4().hex,
        type=str,
        nargs="+",
        help="basename: tts_L语言_S来源_Ffilter版本_D时长_P批次",
    )
    parser.add_argument("--part_names", type=str, nargs="+", default=None)

    parser.add_argument("--base_part", default=0, type=int, help="base part")
    parser.add_argument("--meta_files", default=None, type=str, nargs="+")
    parser.add_argument(
        "--num_reader",
        default=10,
        type=int,
        help="number of reader to read meta simultaneous",
    )
    parser.add_argument(
        "--num_processor",
        default=20,
        type=int,
        help="number of process to processing samples",
    )
    parser.add_argument("--row_group_size", default=64, type=int)
    args = parser.parse_args()
    with Manager() as M:
        main(args, M)
