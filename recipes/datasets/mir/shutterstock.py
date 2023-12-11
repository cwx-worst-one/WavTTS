import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, List, Optional

import torch
import torchaudio
from webdataset import shardlists
from webdataset.pipeline import DataPipeline
from glob import glob

from recipes.datasets.base import WebDataModuleBase
from recipes.datasets.mir.taxonomies.music_sft_en import MusicSFTTokenizerGenresEN, MusicSFTTokenizerMoodsEN, MusicSFTTokenizerScenesEN
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.transforms.audio import Pad, RandomResizedCrop
from samantha.utils.webdataset import return_self

TOKENIZERS = {
    "genre_1": MusicSFTTokenizerGenresEN,
    "mood": MusicSFTTokenizerMoodsEN,
    "scene": MusicSFTTokenizerScenesEN,
}

def get_tokenizer_from_key(tag_key: str):
    return TOKENIZERS[tag_key]()

@dataclass
class ShutterStockSFTDataResult:
    audio: torch.Tensor
    tag_ids: torch.Tensor
    tag_names: str
    duration: float
    song_id: str
    dataset_name: str
    metadata: dict


class ShutterStockSFTDataModule(WebDataModuleBase):
    torchaudio.set_audio_backend("soundfile")
    _data_id: int = 119

    def __init__(
        self,
        tag_key: str, # genre_1, mood, scene, theme
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        self.tag_key = tag_key
        self.tokenizer = TOKENIZERS[tag_key]()
        self._sample_rate = sample_rate
        self._duration = duration

        self.random_crop = RandomResizedCrop(self.duration_audio_samples)
        self.pad = Pad(self.duration_audio_samples)

        train_dataset = ParquetDataset(
            data_id=self._data_id,
            resampled=resampled,
            shardshuffle=shardshuffle,
            nodesplitter=shardlists.single_node_only,
        )

        train_dataset = train_dataset.compose(self.transform)

        validation_dataset = ParquetDataset(
            data_id=self._data_id,
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self,
        )
        validation_dataset = validation_dataset.compose(self.transform)
        super().__init__(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=validation_dataset,
            predict_dataset=validation_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    @property
    def duration_audio_samples(self) -> int:
        return self._sample_rate * self._duration

    def extract_metadata(self, metadata: dict) -> dict:

        if "xbs_label" in metadata:
            # music_Ssstk-pond5_Mnonvocal_T44k_N1608k
            xbs_label = metadata["xbs_label"]["inspection"]
            genre_1 = xbs_label["genre"]
            mood = xbs_label["mood"]
            scene = xbs_label["scene"]
            instruments = xbs_label["instrument"]
            duration = metadata["duration"]
            song_id = metadata["asset_id"]
            genre_2 = None
            theme = None
            voice_gender = None
            voice_char = None
        else:
            # music_sft-instrumental_Smcc_Mnonvocal_N5.6k
            mood = list(set([m for m in metadata["mood"].split(",")]))
            scene = list(set([s for s in metadata["scene"].split(",")]))
            theme = list(set([t for t in metadata["theme"].split(",")]))
            instruments = list(set([i for i in metadata["instrument"].split(",")]))
            genre_1 = list(set([g for g in metadata["genre_1"].split(",")]))
            genre_2 = list(set([g for g in metadata["genre_2"].split(",")]))
            voice_gender = metadata["voice_gender"]
            voice_char = metadata["voice_char"]
            duration = metadata["duration"]
            song_id = metadata["song_id"]
        return dict(
            mood=mood,
            scene=scene,
            theme=theme,
            instruments=instruments,
            genre_1=genre_1,
            genre_2=genre_2,
            voice_gender=voice_gender,
            voice_char=voice_char,
            duration=duration,
            song_id=song_id,
        )

    def transform(self, items: Iterable[Any]):
        for item in items:
            audio_bytes = BytesIO(item["wav"])
            audio, _ = torchaudio.load(audio_bytes)

            audio = audio.mean(dim=0, keepdim=True)

            metadata = json.loads(item["meta"])
            metadata = self.extract_metadata(metadata)

            primary_tag_names = metadata[self.tag_key][0] # primary

            if primary_tag_names not in self.tokenizer.translation_keys:
                continue

            tag_ids = self.tokenizer(primary_tag_names, device="cpu")
            tag_names = self.tokenizer.decode(tag_ids)

            for a in audio.split(self.duration_audio_samples, dim=-1)[:-1]:
                audio = self.pad(audio)
                audio = self.random_crop(audio)
                yield ShutterStockSFTDataResult(
                    audio=a,
                    tag_ids=tag_ids,
                    tag_names=tag_names,
                    song_id=metadata["song_id"],
                    duration=metadata["duration"],
                    dataset_name="sstk_24khz_labeled_train",
                    metadata=metadata,
                )


class ShutterStockPreprocessedDataModule(WebDataModuleBase):

    def __init__(
        self,
        tag_key: str,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        self.tag_key = tag_key
        self.tokenizer = TOKENIZERS[tag_key]()
        self._sample_rate = sample_rate
        self._duration = duration

        print("initializing train datasets...")
        train_datasets = []
        for url2index in glob(f"/mnt/bn/janne-research-xl/data/music_sft/english/shutterstock/{self.tag_key}/*/url2index.txt"):
            print(f"adding dataset: {url2index}")

            dataset = IndexedWebDataset(url2index, resampled=resampled, shardshuffle=shardshuffle).decode().compose(self.transform)
            train_datasets.append(dataset)

        train_dataset = DataPipeline(MultiIterableDataset(train_datasets))

        # TODO: these are the same at the moment :(
        print("initializing validation datasets...")
        validation_datasets = []
        for url2index in glob(f"/mnt/bn/janne-research-xl/data/music_sft/english/shutterstock/{self.tag_key}/*/url2index.txt"):
            dataset = IndexedWebDataset(url2index, resampled=False, shardshuffle=False, nodesplitter=return_self).decode().compose(self.transform)
            validation_datasets.append(dataset)

        validation_dataset = DataPipeline(MultiIterableDataset(validation_datasets))

        super().__init__(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=validation_dataset,
            predict_dataset=validation_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    def transform(self, items: Iterable[Any]):
        for item in items:
            index = item["__index_data__"]

            audio = torch.tensor(item["audio.npy"])
            tag_names = index["tag_names"]
            tag_ids = self.tokenizer(tag_names, device="cpu")
            yield ShutterStockSFTDataResult(
                audio=audio,
                tag_ids=tag_ids,
                tag_names=tag_names,
                song_id=index["song_id"],
                metadata=index["metadata"],
                duration=index["duration"],
                dataset_name=index["dataset_name"],
            )


