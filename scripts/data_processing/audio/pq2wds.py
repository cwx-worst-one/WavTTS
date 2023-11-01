import argparse
import io
import json
import logging
import multiprocessing
import os
import time
from multiprocessing.pool import Pool

import librosa
import numpy as np
import torch
import webdataset as wds
from bytedance.easycycle import get_dataset_detail, get_dataset_info
from lightning_fabric.utilities.cloud_io import get_filesystem
from scipy.io.wavfile import write
from torchaudio.transforms import Resample

from samantha.dataio.utils import parquet_reader
from samantha.utils.watch import elapsed_time
from scripts.utils.bigspeech import get_partition

logger = logging.getLogger(__name__)
# for multiple filesystem instance
multiprocessing.set_start_method("spawn", force=True)


def resample(audio, target_sample_rate, resampler):
    audio, _ = librosa.load(io.BytesIO(audio), sr=None, mono=True)
    audio = resampler(torch.from_numpy(audio).float().cuda()).cpu().numpy()
    audio /= max(1e-5, np.max(np.abs(audio)))
    byteio = io.BytesIO()
    write(byteio, target_sample_rate, (audio * 32767).astype(np.int16))
    byteio.seek(0)
    return byteio.read()


def read_pq(data_url, index_url, dataset_name, target_sample_rate):
    src_sample_rate = target_sample_rate
    for sample in parquet_reader(data_url, need_group_no=False):
        audio = sample["audio"]
        src_sample_rate = librosa.get_samplerate(io.BytesIO(audio))
        break
    logger.info(f"{src_sample_rate=}")
    resampler = None
    if target_sample_rate != src_sample_rate:
        resampler = Resample(
            orig_freq=src_sample_rate, new_freq=target_sample_rate
        ).cuda()
    meta_items = {
        item["uttid"]: item for item in parquet_reader(index_url, need_group_no=False)
    }
    for sample in parquet_reader(data_url, need_group_no=False):
        uttid = sample["uttid"]
        if uttid not in meta_items:
            logger.warning(f"skip cause {index_url=} does not have {uttid=}")
            continue
        meta_item = meta_items[uttid]
        audio = sample["audio"]
        if target_sample_rate != src_sample_rate:
            audio = resample(audio, target_sample_rate, resampler)
        item = {}
        meta_obj = json.loads(meta_item["meta"])
        item["__key__"] = sample["uttid"]
        item["wav"] = audio
        item["text"] = str(meta_item["text"])
        item["labels"] = str(meta_obj.get("labels", ""))
        item["dataset_name"] = str(dataset_name)
        item["speaker_name"] = str(meta_obj.get("speaker_id", ""))
        item["snr"] = str(meta_obj.get("snr", "10.0"))
        item["mos"] = str(meta_obj.get("mos", "5.0"))
        item["rms_stats_rms_max"] = "-1"
        item["speaker_similarity_min"] = "1.0"
        yield item


def convert(
    data_url,
    index_url,
    output_url,
    dataset_name,
    ckpt_file,
    target_sample_rate,
    filesystem=None,
):
    if filesystem is None:
        filesystem = get_filesystem(data_url)
    tarstream, stream = None, None
    try:
        stream = filesystem.open(output_url, mode="wb")
        tarstream = wds.TarWriter(stream)
        for item in read_pq(data_url, index_url, dataset_name, target_sample_rate):
            tarstream.write(item)
        tarstream.close()
        stream.close()
        logger.info(f"done {output_url=}")
        filesystem.touch(ckpt_file)
    except Exception as e:
        logger.warning(f"{data_url=} error with msg {e}")
        if tarstream is not None:
            tarstream.close()
        if stream is not None:
            stream.close()
        if filesystem.exists(output_url):
            filesystem.rm_file(output_url)


@elapsed_time
def main(args):
    for dataset_id in args.dataset_ids:
        ROOT_PATH = get_dataset_info(dataset_id)
        dataset_name = get_dataset_detail(dataset_id)["dataset"]["name"]
        filesystem = get_filesystem(ROOT_PATH)
        partitions, suffix = get_partition(ROOT_PATH, filesystem)
        url_pattern = f"{ROOT_PATH}/data/{'/'.join(['*'] * len(partitions))}/*.{suffix}"
        wav = f"wav_{args.version}_web_dataset_{args.index_version}"
        urls = filesystem.glob(url_pattern)
        index = f"index_{args.index_version}"
        num_process = min(len(urls), 10)
        rpool = Pool(num_process)
        for url in urls:
            url = "hdfs://haruna" + url
            index_url = url.replace(
                f"{ROOT_PATH}/data/", f"{ROOT_PATH}/{index}/"
            ).replace(".parquet", f".{index}.parquet")
            output_url = url.replace(
                f"{ROOT_PATH}/data/", f"{ROOT_PATH}/package/{wav}/data/"
            ).replace(".parquet", ".tar")

            ckpt_file = os.path.join(
                os.path.dirname(output_url), f".{os.path.basename(output_url)}.SUCCESS"
            )
            if (not args.force and filesystem.exists(ckpt_file)) or (
                not filesystem.exists(index_url)
            ):
                logger.info(f"skipping file {output_url=}")
                continue
            rpool.apply_async(
                func=convert,
                args=(
                    url,
                    index_url,
                    output_url,
                    dataset_name,
                    ckpt_file,
                    args.target_sample_rate,
                    filesystem,
                ),
                error_callback=lambda x: print("rpool:", x),
            )
        rpool.close()
        rpool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", nargs="+", type=str, required=True)
    parser.add_argument("--index_version", type=int, default=1)
    parser.add_argument("--version", type=str, default="1.0")
    parser.add_argument("--target_sample_rate", type=int, default=24000)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
