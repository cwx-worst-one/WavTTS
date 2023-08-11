import torchaudio

from recipes.datasets.librispeech import LibriSpeechDataset, LibriSpeechWebDataModule, SAMPLE_RATE

if __name__ == "__main__":

    split = "train-clean-360"
    dataset = LibriSpeechDataset(root="/mnt/bn/janne-research-xl/data", split=split)

    pattern = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/speech/libritts/{split}/%05d.tar"

    maxsize = (1 << 32) * 2  # 8GiB, maximum size of each shard

    pl_datamodule = LibriSpeechWebDataModule(
        duration=5.0, batch_size=8, shuffle_buffer_size=100
    )
    pl_datamodule.create_webdataset(dataset, pattern, maxsize)

    train_loader = pl_datamodule.train_dataloader()
    for batch in train_loader:
        audio = batch[0]
        torchaudio.save("test.mp3", audio[0], SAMPLE_RATE)
        break
