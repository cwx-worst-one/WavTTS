import random

import torchaudio

from recipes.datasets.libritts import (
    SAMPLE_RATE,
    LibriTTSDataset,
    LibriTTSWebDataModule,
)
from recipes.umm_062.datamodules.base import speech_collate_fn
from recipes.umm_062.transforms.speech import SpeechTransform

if __name__ == "__main__":

    split = "test-clean" # train-clean-360
    target_sample_rate = 24000 # 16000
    dataset = LibriTTSDataset(
        root="/mnt/bn/janne-research-xl/data/",
        split=split
    )

    pattern = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/speech/libritts/{target_sample_rate}hz/{split}/%05d.tar"
    
    maxsize = (1 << 32) * 2 # 8GiB, maximum size of each shard

    pl_datamodule = LibriTTSWebDataModule(
        sample_rate=target_sample_rate,
        batch_size=8,
        shuffle_buffer_size=128,
        collate_fn=speech_collate_fn,
    )

    # VERY IMPORTANT TO GET DIVERSE SHARDS:
    if "train" in split:
        dataset.random_shuffle()

    pl_datamodule.create_webdataset(
        dataset,
        pattern,
        maxsize
    )

    train_loader = pl_datamodule.train_dataloader()
    for batch in train_loader:
        audio = batch["audio"]
        torchaudio.save("test.mp3", audio[0], SAMPLE_RATE)
        break

