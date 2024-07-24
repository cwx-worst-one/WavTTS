import os
import pickle
from argparse import ArgumentParser
from io import BytesIO
from itertools import accumulate, chain
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetFile
from torchaudio.transforms import AmplitudeToDB
from tqdm import tqdm

from samantha.data.audio_utils import convert_audio
from samantha.data.av_audio import audio_info, audio_read
from samantha.transforms.audio import MelSpectrogram
from samantha.utils.hdfs_helper import hdfs_ls


class MelSpectrogramFeatures(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        f_max: float,
        clamp_min: float = 1e-9,
    ):
        super().__init__()
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            center=True,
            power=2.0,
            f_max=f_max,
        )
        self.clamp_min = clamp_min
        self.amplitude_to_db = AmplitudeToDB(stype="power")  # power

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.float()
        with torch.cuda.amp.autocast(enabled=False):
            # audio = audio - torch.mean(audio, dim=2, keepdim=True)  # TODO: remove DC offset?
            mel, _ = self.mel_spec(audio)

            mel = mel.clamp(min=self.clamp_min)

            mel = self.amplitude_to_db(mel)
            mel = mel[..., :-1]
        return mel


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def wav_bytes_to_audio_tensor(
    data: bytes, sample_rate: int, channels: int, format: str
):
    data = BytesIO(data)
    # info = audio_info(data, format=format)
    out, sr = audio_read(data, pad=False, format=format)
    out = convert_audio(out, sr, sample_rate, channels)
    return out, sample_rate


def process_row_group(
    row_group_no, sample_rate: int, channels: int, dtype: torch.dtype
):

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
        data = values.audio

        audio, sr = wav_bytes_to_audio_tensor(
            data, sample_rate=sample_rate, channels=channels, format="wav"
        )

        mel = feature_extractor.forward(audio)
        mel = mel.numpy()

        buffer = BytesIO()
        np.save(buffer, mel, allow_pickle=True)

        items.append(
            {
                "row_group_no": row_group_no,
                "group_index": index,
                "uttid": uttid,
                "mel": buffer.getvalue(),
            }
        )
    return items


if __name__ == "__main__":
    n_workers = 32  # NOTE: size of parquet file * n_workers much fit into RAM!
    src_dir = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_everynoise_N937k_Lmix/data"
    target_dir = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_everynoise_N937k_Lmix_mel/data"

    parser = ArgumentParser()
    parser.add_argument("--rank", required=True, type=str)
    parser.add_argument("--world_size", required=True, type=str)
    args = parser.parse_args()

    rank = int(args.rank)
    world_size = int(args.world_size)

    ## pip3 install fastparquet

    ## obtain shard lists to process per rank:
    fps = hdfs_ls(src_dir)
    existing_fps = [os.path.basename(fp) for fp in hdfs_ls(target_dir)]
    fps = [fp for fp in fps if os.path.basename(fp) not in existing_fps]
    fps = list(chunks(fps, world_size))
    rank_fps = fps[rank]

    sample_rate = 44100
    channels = 1
    dtype = torch.bfloat16
    feature_extractor = MelSpectrogramFeatures(
        sample_rate=44100,
        n_mels=160,
        n_fft=8192,
        hop_length=441,
        f_max=20000,
        clamp_min=1e-9,
    )

    ## process each shard
    for fp in tqdm(rank_fps, desc=f"Rank: {rank} / {world_size}"):
        tik = perf_counter()

        ## open source Parquet file
        fs = get_filesystem(fp)
        stream = fs.open(fp, skip_instance_cache=True)
        parquet_file = ParquetFile(stream)

        ## process Parquet file
        items = Parallel(n_jobs=n_workers, prefer="processes")(
            delayed(process_row_group)(
                row_group, sample_rate=sample_rate, channels=channels, dtype=dtype
            )
            for row_group in tqdm(range(parquet_file.num_row_groups))
        )

        ## write to new parquet:
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
        )

        tok = perf_counter()

        print(f"Finished in {tok - tik} seconds")
