import copy
import io
import json
import logging
import multiprocessing.util
from multiprocessing.pool import Pool

import librosa
import numpy as np
from lightning_fabric.utilities.cloud_io import get_filesystem
from tqdm import tqdm

from samantha.dataio.parquet.writer import ShardWriter
from samantha.dataio.utils import parquet_reader
from samantha.utils.watch import elapsed_time

# for randomizing manager server port
multiprocessing.util.abstract_sockets_supported = False
# for multiple filesystem instance
multiprocessing.set_start_method("spawn", force=True)


logger = logging.getLogger(__name__)


def read_url(data_url, fs, index_url=None):
    if index_url is not None:
        meta_items = {
            item["uttid"]: item
            for item in parquet_reader(index_url, fs=fs, need_group_no=False)
        }
        for data_item in parquet_reader(data_url, fs=fs, need_group_no=False):
            uttid = data_item["uttid"]
            if uttid not in meta_items:
                logger.warning(f"skip cause {index_url=} does not have {uttid=}")
                continue
            meta_item = meta_items[uttid]
            meta = json.loads(meta_item["meta"])
            meta["duration"] = librosa.get_duration(
                filename=io.BytesIO(data_item["audio"])
            )
            meta = json.dumps(meta, ensure_ascii=False)
            meta_item["meta"] = meta
            data_item["meta"] = meta
            yield data_item, meta_item
    else:
        for x in parquet_reader(data_url, fs=fs, need_group_no=False):
            item = copy.deepcopy(x)
            meta = json.loads(item["meta"])
            meta["duration"] = librosa.get_duration(filename=io.BytesIO(item["audio"]))
            item["meta"] = json.dumps(meta, ensure_ascii=False)
            yield item,


PLACEHOLDER = "__placeholder__"


class Consumer:
    def __init__(self, output_pattern):
        self._prefix = output_pattern.split(PLACEHOLDER)[0]
        self.output_pattern = output_pattern
        self.data_writer = ShardWriter(
            output_pattern=output_pattern.replace(PLACEHOLDER, "data"), maxcount=2048
        )
        self.meta_writer = ShardWriter(
            output_pattern=output_pattern.replace(PLACEHOLDER, "index_1"),
            maxcount=2048,
            need_row_group_no=True,
        )

    def write(self, data_item, meta_item=None):
        self.data_writer.write(data_item)
        meta_json_str = data_item["meta"]
        if meta_item is not None:
            meta_json_str = meta_item["meta"]
        meta = {
            "uttid": data_item["uttid"],
            "text": data_item["text"],
            "meta": meta_json_str,
            "data_file": self.data_writer.filename.replace(self._prefix, "../../"),
        }
        self.meta_writer.write(meta)

    def close(self):
        self.data_writer.close()
        self.meta_writer.close()


def run(url_pairs, output_pattern, fs, part):
    logger.info(f"{len(url_pairs)=}, {output_pattern=}")
    consumer = Consumer(output_pattern)
    for data_url, index_url in tqdm(url_pairs, desc=f"{part}"):
        if not (fs.exists(data_url) and fs.exists(index_url)):
            continue
        try:
            for item in read_url(data_url=data_url, fs=fs, index_url=index_url):
                consumer.write(*item)
        except Exception as exn:
            logger.warning(f"{data_url=} error", exc_info=exn)
    consumer.close()


def get_work_urls(data_urls, index_version, num_parts):
    index = f"index_{index_version}"
    index_urls = [
        url.replace("/data/part=", f"/{index}/part=").replace(
            ".parquet", f".{index}.parquet"
        )
        for url in data_urls
    ]
    urls_per_part = np.array_split(list(zip(data_urls, index_urls)), num_parts)
    yield from urls_per_part


@elapsed_time
def main(args):
    filesystem = get_filesystem(args.input_pattern)
    output_pattern = f"{args.output_root}/{PLACEHOLDER}/part=%05d/shard-%%05d.parquet"
    if args.repart_num is not None:
        data_urls = filesystem.glob(args.input_pattern)
        pool = Pool(min(args.num_processor, args.repart_num))
        for part, url_pairs in enumerate(
            get_work_urls(data_urls, args.index_version, args.repart_num)
        ):
            cur_output_pattern = output_pattern % part
            pool.apply_async(
                func=run,
                args=(url_pairs, cur_output_pattern, filesystem, part),
                error_callback=lambda x: print("err ------------>", x),
            )
        pool.close()
        pool.join()
    else:
        pool = Pool(args.num_processor)
        for part in range(50, 100):
            input_pattern = args.input_pattern % part
            data_urls = filesystem.glob(input_pattern)
            cur_output_pattern = output_pattern % part
            for url_pairs in get_work_urls(data_urls, args.index_version, 1):
                pool.apply_async(
                    func=run,
                    args=(url_pairs, cur_output_pattern, filesystem, part),
                    error_callback=lambda x: print("err ------------>", x),
                )
        pool.close()
        pool.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pattern",
        type=str,
        default=None,
        help=r"""input data parquet pattern. if repart_num is None,
        input_pattern should be like hdfs://xxx/part=%%05d/*/*.parquet,
        if not None, it shoule be like hdfs://xxx/part=*/*/*.parquet
        """,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="hdfs://xxx/BigTTS/tts_Len_Sresso-podcast_F2.0_D5-10_P1/",
        help="output base directory",
    )
    parser.add_argument(
        "--num_processor", default=10, type=int, help="number processor to repack"
    )
    parser.add_argument("--repart_num", type=int, default=None)
    parser.add_argument("--index_version", type=int, default=1)

    args = parser.parse_args()

    main(args)
