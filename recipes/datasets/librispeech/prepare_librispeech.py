import torchaudio

from recipes.datasets.librispeech.librispeech import LibriSpeechDataset, LibriSpeechWebDataModule, SAMPLE_RATE

if __name__ == "__main__":

    split = "test-clean"
    dataset = LibriSpeechDataset(root="/mnt/bn/janne-research-xl/data/librispeech", split=split)
    item = next(iter(dataset))

    pattern = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/speech/librispeech/{split}/%05d.tar"

    maxsize = (1 << 32) * 2  # 8GiB, maximum size of each shard

    pl_datamodule = LibriSpeechWebDataModule(
        sample_rate=SAMPLE_RATE,
        batch_size=8,
        shuffle_buffer_size=100,
        duration=5.0,
        buckets_sec=None,
        use_bucket_batcher=False,
    )
    pl_datamodule.create_webdataset(dataset, pattern, maxsize)

    # train_loader = pl_datamodule.train_dataloader()
    # for batch in train_loader:
    #     audio = batch.audio[0]
    #     torchaudio.save("test.mp3", audio, SAMPLE_RATE)
    #     break
