import os
from recipes.datasets.billboard_hot200.billboard_hot200 import (
    BillboardDataModule,
    BillboardASRDataModule,
)
from samantha.utils.hdfs_tools import hdfs_exists


if __name__ == "__main__":
    version = "hot200"
    split = "test"
    sample_rate = 24000
    mono = True

    url2index = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_{version}_v2/{sample_rate}hz/{split}/url2index.txt"

    datamodule = BillboardDataModule(
        url2index=url2index,
        sample_rate=sample_rate,
        batch_size=8,
        shuffle_buffer_size=50,
    )

    pattern = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_{version}_lyrics_asr_v2/{sample_rate}hz/{split}/%05d.tar"
    if hdfs_exists(os.path.dirname(pattern)):
        raise FileExistsError(f"{pattern} already exists on HDFS")

    maxsize = 1 << 33  # 8GiB, maximum size of each shard
    start_shard_idx = 0
    print(f"Shard size: {maxsize}")
    print(pattern)
    BillboardASRDataModule.create_webdataset(
        datamodule,
        data_sample_rate=sample_rate,
        pattern=pattern,
        maxsize=maxsize,
        start_shard_idx=start_shard_idx,
    )
