import os
from uuid import uuid4
from tqdm import tqdm
from recipes.datasets.mir.music_sft import MusicSFTDataModule, MusicSFTVocalDataModuleEN, MCC30KDataModuleZH
from recipes.datasets.mir.taxonomies.music_sft_en import MusicSFTTokenizerGenresEN, MusicSFTTokenizerVoiceGenderEN
from recipes.datasets.mir.taxonomies.music_sft_zh import MusicSFTTokenizerGenresZH, MusicSFTTokenizerMoodsZH, MusicSFTTokenizerScenesZH
from recipes.mi1.models.tokenizers import UMMTokenizer
from samantha.dataio.webdataset.writer import IndexShardWriter
from recipes.datasets.mir.shutterstock import ShutterStockSFTDataModule

if __name__ == "__main__":
    total_datapoints = 58089
    duration = 30
    batch_size = 64
    shuffle_buffer_size = 0
    num_workers = 8

    tag_key = "scene"
    # tokenizer = MusicSFTTokenizerVoiceGenderEN()
    # tokenizer = MusicSFTTokenizerGenresZH()
    # tokenizer = MusicSFTTokenizerMoodsZH()
    # tokenizer = MusicSFTTokenizerScenesZH()
    
    # datamodule = MusicSFTDataModule(
    # datamodule = MusicSFTVocalDataModuleEN(
    # datamodule = MCC30KDataModuleZH(
    datamodule = ShutterStockSFTDataModule(
        tag_key=tag_key,
        sample_rate=24000,
        duration=duration,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        resampled=False,
        shardshuffle=False,
        num_workers=num_workers
    )
    train_loader = datamodule.train_dataloader()


    # load audio tokenizer
    # audio_tokenizer = UMMTokenizer().to("cuda")

    writers = {}

    for v in datamodule.tokenizer.vocab:
        dir = os.path.join("/mnt/bn/janne-research-xl/data/music_sft/english/shutterstock", tag_key, v)
        os.makedirs(dir, exist_ok=True)
        writers[v] = IndexShardWriter(
            os.path.join(dir, "%05d.tar"),
            maxcount=2000,
        )
    
    for batch in tqdm(train_loader, total=total_datapoints // batch_size):
        audio = batch.audio.to("cuda")
        # hidden_states = audio_tokenizer(audio).hidden_states.cpu()

        for idx in range(batch_size):
            obj = {}
            index = {}

            obj["__key__"] = str(uuid4())
            obj["audio.npy"] = batch.audio[idx].numpy()
            # obj["hidden_states.npy"] = hidden_states[idx].numpy()

            tag_name = batch.tag_names[idx]

            index["tag_ids"] = batch.tag_ids[idx]
            index["tag_names"] = batch.tag_names[idx]
            index["song_id"] = batch.song_id[idx]
            index["metadata"] = batch.metadata[idx]
            index["duration"] = batch.duration[idx]
            index["song_id"] = batch.song_id[idx]
            index["dataset_name"] = batch.dataset_name[idx]

            try:
                writers[tag_name].write(obj, index)
            except Exception as e:
                print(e)

    for k in writers:
        writers[k].close()