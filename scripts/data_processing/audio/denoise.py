"""thanks to @liuyang.1314"""
import argparse
import io
import json
import logging
import multiprocessing
import os
import warnings
from multiprocessing.pool import Pool

import librosa
import numpy as np
import torch
import torch.cuda
import tqdm
from lightning_fabric.utilities.cloud_io import get_filesystem
from scipy.io.wavfile import write

from samantha.dataio.parquet.writer import Writer
from samantha.dataio.utils import parquet_reader
from scripts.data_processing.audio.stft import ISTFT, STFT
from scripts.data_processing.bigtts.gen_duration import get_partition

warnings.simplefilter(action="ignore", category=FutureWarning)


logger = logging.getLogger(__name__)
multiprocessing.set_start_method("spawn", force=True)


class Worker:
    def __init__(self, rank, max_duration) -> None:
        self.device = f"cuda:{rank}"
        self.rank = rank
        self.max_duration = max_duration
        self.ori_data_files = []
        self.ori_index_files = []
        self.new_data_files = []
        self.ckpt_files = []

    def en_task(self, ori_data_file, ori_index_file, new_data_file, ckpt_file):
        self.ori_data_files.append(ori_data_file)
        self.ori_index_files.append(ori_index_file)
        self.new_data_files.append(new_data_file)
        self.ckpt_files.append(ckpt_file)


class Consumer:
    def __init__(self, data_file, filesystem):
        self.data_writer = Writer(data_file, filesystem=filesystem)
        self.meta_writer = Writer(
            data_file.replace("/data/part=", "/index_1/part=").replace(
                ".parquet", ".index_1.parquet"
            ),
            need_row_group_no=True,
            filesystem=filesystem,
        )
        self._prefix = data_file.split("data/part=")[0]

    def write(self, data_item, meta_item):
        self.data_writer.write(data_item)
        assert data_item["uttid"] == meta_item["uttid"]
        meta = {
            "uttid": meta_item["uttid"],
            "text": meta_item["text"],
            "meta": meta_item["meta"],
            "data_file": self.data_writer.filename.replace(self._prefix, "../../"),
        }
        self.meta_writer.write(meta)

    def close(self):
        self.data_writer.close()
        self.meta_writer.close()


def process_batch(
    model, batch_items, sr, device, max_duration, stft, inverse_stft, thresh
):

    batch_wavs = [
        librosa.load(io.BytesIO(item["audio"]), sr=None, mono=True)[0]
        for item in batch_items
    ]
    length = [wav.shape[0] for wav in batch_wavs]
    max_length = max(max_duration * sr, max(length))
    batch_wavs = [
        np.pad(wav, pad_width=(0, max_length - length[i])).reshape((1, -1))
        for i, wav in enumerate(batch_wavs)
    ]

    batch_wavs = np.vstack(batch_wavs)
    batch_wavs = torch.from_numpy(batch_wavs).to(device)
    max_value = torch.max(batch_wavs, dim=1)[0][:, None]
    min_value = torch.min(batch_wavs, dim=1)[0][:, None]
    threhold = torch.where(
        torch.logical_or(max_value > thresh, min_value < -thresh),
        torch.ones_like(max_value) * 0.5,
        torch.ones_like(max_value),
    )
    low_waveform = batch_wavs * threhold

    audio_length = low_waveform.shape[-1]
    (low_real, low_imag) = stft(low_waveform)
    result = model(low_real, low_imag)
    audio_data = inverse_stft(result[0], result[1], audio_length).cpu().numpy()
    return [audio[:ilen] for ilen, audio in zip(length, audio_data)]


def load_model(device):
    model = torch.jit.load("model2_best.pt").eval()
    model = model.to(device)
    stft = (
        STFT(
            n_fft=512,
            hop_length=256,
            win_length=512,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        .to(device)
        .eval()
    )
    inverse_stft = (
        ISTFT(
            n_fft=512,
            hop_length=256,
            win_length=512,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        .to(device)
        .eval()
    )
    return model, stft, inverse_stft


@torch.no_grad()
def deal_wav(
    ori_data_files,
    ori_index_files,
    new_data_files,
    ckpt_files,
    sr,
    device,
    max_duration,
    fs,
    batch_size=128,
    thresh=0.85,
    denoise=True,
):
    if denoise:
        model, stft, inverse_stft = load_model(device)

    logger.info(f"{len(ori_data_files)=}, {len(new_data_files)=}")
    for ori_data_file, ori_index_file, new_data_file, ckpt_file in zip(
        ori_data_files, ori_index_files, new_data_files, ckpt_files
    ):

        consumer = Consumer(new_data_file, filesystem=fs)
        batch_items, meta_items = [], []
        if denoise:
            for data_item, meta_item in tqdm.tqdm(
                zip(
                    parquet_reader(ori_data_file, fs, need_group_no=False),
                    parquet_reader(ori_index_file, fs, need_group_no=False),
                )
            ):
                meta = json.loads(meta_item["meta"])
                if meta["mosnet_202309"] < 3.8:
                    continue
                batch_items.append(data_item)
                meta_items.append(meta_item)
                if len(batch_items) == batch_size:
                    for old_item, old_meta_item, cur_wav in zip(
                        batch_items,
                        meta_items,
                        process_batch(
                            model,
                            batch_items,
                            sr,
                            device,
                            max_duration,
                            stft,
                            inverse_stft,
                            thresh,
                        ),
                    ):
                        bytes_io = io.BytesIO()
                        write(bytes_io, sr, (cur_wav * 32767).astype(np.int16))
                        bytes_io.seek(0)
                        old_item["audio"] = bytes_io.read()
                        consumer.write(data_item=old_item, meta_item=old_meta_item)
                    batch_items, meta_items = [], []

            if batch_items:
                for old_item, old_meta_item, cur_wav in zip(
                    batch_items,
                    meta_items,
                    process_batch(
                        model,
                        batch_items,
                        sr,
                        device,
                        max_duration,
                        stft,
                        inverse_stft,
                        thresh,
                    ),
                ):
                    bytes_io = io.BytesIO()
                    write(bytes_io, sr, (cur_wav * 32767).astype(np.int16))
                    bytes_io.seek(0)
                    old_item["audio"] = bytes_io.read()
                    consumer.write(data_item=old_item, meta_item=old_meta_item)
                batch_items, meta_items = [], []
        else:
            for data_item, meta_item in tqdm.tqdm(
                zip(
                    parquet_reader(ori_data_file, fs, need_group_no=False),
                    parquet_reader(ori_index_file, fs, need_group_no=False),
                )
            ):
                meta = json.loads(meta_item["meta"])
                key = "mosnet_202309"
                if key not in meta:
                    logger.info(f"file={ori_index_file} does not has {key=}")
                    break
                if meta[key] < 3.8:
                    continue
                consumer.write(data_item=data_item, meta_item=meta_item)
        consumer.close()
        fs.touch(ckpt_file)


def main(args):
    new_root_path = args.new_root_path
    ROOT_PATH = args.ori_root_path
    base_part = args.base_part + int(os.getenv("ARNOLD_ID", 0)) * args.part_interval
    parts = [
        f"{args.part_prefix}={part:05d}"
        for part in range(base_part, base_part + args.part_interval)
    ]
    filesystem = get_filesystem(ROOT_PATH)
    partitions, suffix = get_partition(ROOT_PATH, filesystem)
    url_pattern = f"{ROOT_PATH}/data/{'/'.join(['*'] * len(partitions))}/*.{suffix}"
    urls = filesystem.glob(url_pattern)
    n_worker = args.num_worker
    index = f"index_{args.index_version}"
    workers = []
    for i in range(n_worker):
        workers.append(Worker(i, args.max_duration))

    idx = 0
    for url in urls:
        for part in parts:
            if part in url:
                url = "hdfs://haruna" + url
                index_url = url.replace(
                    f"{ROOT_PATH}/data/", f"{ROOT_PATH}/{index}/"
                ).replace(".parquet", f".{index}.parquet")
                new_url = url.replace(ROOT_PATH, new_root_path)
                ckpt_file = os.path.join(
                    os.path.dirname(new_url), f".{os.path.basename(new_url)}.SUCCESS"
                )
                if filesystem.exists(ckpt_file):
                    break
                logger.info(f"{url=}")
                logger.info(f"{index_url=}")
                logger.info(f"{new_url=}")
                workers[idx % n_worker].en_task(url, index_url, new_url, ckpt_file)
                idx += 1
                break
    pool = Pool(n_worker)
    for worker in workers:
        pool.apply_async(
            func=deal_wav,
            args=(
                worker.ori_data_files,
                worker.ori_index_files,
                worker.new_data_files,
                worker.ckpt_files,
                24000,
                worker.device,
                worker.max_duration,
                filesystem,
                args.batch_size,
                args.thresh,
                args.denoise,
            ),
            error_callback=lambda x: print("err:", x),
        )

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_root_path", type=str, required=True)
    parser.add_argument("--max_duration", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--new_root_path", type=str)
    parser.add_argument("--model", type=str, default="model2_best.pt")
    parser.add_argument("--base_part", type=int, default=0)
    parser.add_argument("--part_interval", type=int, default=1)
    parser.add_argument("--part_prefix", type=str, default="part")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--thresh", type=float, default=0.85)
    parser.add_argument("--index_version", type=int, default=1)
    parser.add_argument("--denoise", action="store_true", default=False)
    args = parser.parse_args()
    logger.info(f"launch with {args=}")
    main(args)
