import os
from recipes.datasets.billboard_hot200.billboard_hot200 import BillboardHot200Dataset, BillboardDataModule
from recipes.datasets.base import parallel_resample
from samantha.utils.hdfs_tools import hdfs_exists


if __name__ == "__main__":
    version = "hot200"
    split = "train"
    sample_rate = 44100
    mono = False
    lyrics_only = False

    audio_dir = "/mnt/bn/janne-research-xl/data/mcc/billboard_hot_200-v2/"
    metadata_fp = (
        f"./spotify-scraper/datasets/billboard/billboard_tables/{version}/final.pickle"
    )

    dataset_name = (
        f"billboard_{version}_v2_lyrics" if lyrics_only else f"billboard_{version}_v2"
    )
    pattern = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/{dataset_name}/{sample_rate}hz/{split}/%05d.tar"

    if hdfs_exists(os.path.dirname(pattern)):
        raise FileExistsError(f"{pattern} already exists on HDFS")

    dataset = BillboardHot200Dataset(
        metadata_fp=metadata_fp,
        audio_dir=audio_dir,
        split=split,
        verify_dataset=False,
        ext_audio=".flac",
        lyrics_only=lyrics_only,
    )
    dataset.random_shuffle()

    parallel_resample(
        dataset.audio_filepaths,
        sample_rate=sample_rate,
        mono=mono,
        n_jobs=32,
        overwrite=False,
    )

    maxsize = 1 << 33  # 8GiB, maximum size of each shard
    start_shard_idx = 0
    print(f"Shard size: {maxsize}")
    print(pattern)
    BillboardDataModule.create_webdataset(
        dataset,
        sample_rate=sample_rate,
        mono=mono,
        pattern=pattern,
        maxsize=maxsize,
        start_shard_idx=start_shard_idx,
    )
