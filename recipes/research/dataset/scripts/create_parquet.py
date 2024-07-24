import json
import os
import subprocess
from glob import glob
from io import BytesIO
from typing import Generator, Iterable, List

from lightning_fabric.utilities.cloud_io import get_filesystem
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from torch.utils.data import DataLoader
from tqdm import tqdm
from webdataset import SimpleShardList
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline
from webdataset.shardlists import split_by_node, split_by_worker
from webdataset.utils import pytorch_worker_info

from samantha.dataio.parquet.writer import IndexShardWriter
from samantha.utils.hdfs_helper import fast_glob_files
from samantha.utils.logger import RankedLogger
from samantha.data.av_audio import audio_info

logger = RankedLogger()


class HDFSDataset(DataPipeline, FluidInterface):
    def __init__(self, data_urls: List[str], nodesplitter=split_by_node):
        super().__init__()
        self.data_urls = data_urls

        self.append(SimpleShardList(self.data_urls))
        self.append(nodesplitter)
        self.append(split_by_worker)
        self = self.compose(self.transform)

    def transform(self, items: Iterable) -> Generator:
        for item in items:
            fs = get_filesystem(item["url"])

            fp, ext = os.path.splitext(item["url"])
            lyrics_lrc_fp = fp + ".lrc"

            if os.path.exists(lyrics_lrc_fp):
                with open(lyrics_lrc_fp) as f:
                    lyrics_lrc = f.read()
            else:
                lyrics_lrc = ""

            meta_song_id = os.path.basename(fp)

            with fs.open(item["url"], skip_instance_cache=True) as f:
                yield {
                    "meta_song_id": meta_song_id,
                    "audio" + ext: f.read(),
                    "lyrics.lrc": lyrics_lrc,
                }


def get_mp3_metadata(mp3_bytes: bytes) -> dict:
    id3 = EasyID3(BytesIO(mp3_bytes))
    return dict(id3)


def wav_bytes_to_mp3_bytes(wav_bytes: bytes) -> bytes:

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

    mp3_buffer = BytesIO()
    mp3_buffer.write(stdout)
    return mp3_buffer.getvalue()


def resample_audio_bytes(wav_bytes: bytes, target_sample_rate: int) -> bytes:

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-f",
            "flac",
            "-ar",
            str(target_sample_rate),
            "-f",
            "wav",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=wav_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

    audio_buffer = BytesIO()
    audio_buffer.write(stdout)
    return audio_buffer.getvalue()


class ParallelParquetWriter(HDFSDataset):
    def __init__(
        self,
        data_urls: List[str],
        target_dir: str,
        sample_rate: int,
        n_songs_per_shard: int,
        n_songs_per_group: int,
        idx_version: int = 1,
    ):
        super().__init__(data_urls=data_urls, nodesplitter=split_by_node)
        self.target_dir = target_dir
        self.sample_rate = sample_rate
        self.n_songs_per_shard = n_songs_per_shard
        self.n_songs_per_group = n_songs_per_group
        self.idx_version = idx_version
        self = self.compose(self.write)

    def write(self, items: Iterable) -> Generator:
        rank, world_size, worker, num_workers = pytorch_worker_info()

        partition = f"part={((rank + 1) * (worker + 1) - 1):05d}"

        logger.info(
            f"Opened partition {partition} ({rank=}, {world_size=}, {worker=}, {num_workers=})"
        )
        writer = IndexShardWriter(
            self.target_dir,
            idx_version=self.idx_version,
            partitions=[partition],
            maxcount=self.n_songs_per_shard,
            row_group_size=self.n_songs_per_group,
        )
        try:
            for item in items:
                audio = item["audio.mp3"]
                # audio = wav_bytes_to_mp3_bytes(audio)

                metadata = get_mp3_metadata(audio)

                audio = resample_audio_bytes(audio, self.sample_rate)

                info = audio_info(BytesIO(audio), format="wav")
                data = {"uttid": item["meta_song_id"], "audio": audio}

                meta = {
                    "meta_song_id": item["meta_song_id"],
                    "duration": info.duration,
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "lyrics.lrc": item["lyrics.lrc"],
                    **metadata,
                }

                writer.write(item["meta_song_id"], data=data, meta=meta)
                yield item
        finally:
            logger.info(f"Closed partition {partition}")
            writer.close()


if __name__ == "__main__":
    # pip3 install mutagen
    rank, world_size, worker, num_workers = pytorch_worker_info()

    print(f"{rank=}/{world_size=}")

    # data_urls = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/playlist_v5/*.flac"
    # data_urls = sorted(fast_glob_files(data_urls))

    data_urls = sorted(glob("artist_sft/frank_sinatra/*.mp3"))
    # data_urls = sorted(glob("artist_sft/radiohead/*.mp3"))
    # data_urls = sorted(fast_glob_files(data_urls))

    target_dir = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_artist_sft_frank_sinatra"

    sample_rate = 44100
    parallel = ParallelParquetWriter(
        data_urls,
        target_dir,
        sample_rate=sample_rate,
        n_songs_per_shard=27,
        n_songs_per_group=3,
    )

    items_per_rank = len(data_urls) // world_size
    for idx, item in enumerate(tqdm(parallel, total=items_per_rank)):
        pass
        # if idx == 1000:
        #     break
