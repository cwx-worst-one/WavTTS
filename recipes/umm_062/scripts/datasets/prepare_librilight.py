import os
import random
from pathlib import Path

import torchaudio

from recipes.datasets.librilight import (
    SAMPLE_RATE,
    LibriLightDataset,
    LibriLightWebDataModule,
    preprocess_librilight,
)
from recipes.umm_062.transforms.speech import SpeechTransform
from samantha.utils.hdfs_helper import hdfs_getsize, hdfs_ls
from samantha.utils.hdfs_tools import hdfs_loadtxt

if __name__ == "__main__":
    preproc_dir = "/mnt/bn/janne-research-xl/data/librilight/preprocessed"
    split = "large"
    # preprocess_librilight(
    #     out_dir=preproc_dir,
    #     root="/mnt/bn/janne-research-xl/data/librilight/",
    #     split=split,
    #     max_len_sec=60
    # )

    ## Prepare webdataset
    dataset = LibriLightDataset(
        root=preproc_dir,
        split=split,
    )
    ## NOTE: VERY IMPORTANT TO GET DIVERSE SHARDS:
    random.seed(42) # make deterministic
    random.shuffle(dataset._walker)
    pattern = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/speech/librilight/{split}/%05d.tar"
    maxsize = (1 << 32) * 2 # 8GiB, maximum size of each shard

    print(dataset._walker[:10])
    # dataset_start_idx = 0
    # start_shard_idx = 0
    # print(f"Resuming at: {dataset_start_idx/dataset.total*100:.1f}%")
    # dataset._walker = dataset._walker[dataset_start_idx:]
    # dataset.total = len(dataset._walker)

    LibriLightWebDataModule.create_webdataset(
        dataset,
        str(pattern),
        maxsize,
        start_shard_idx=start_shard_idx
    )
