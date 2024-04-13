import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, List, Optional

import torch
import torchaudio
from webdataset import shardlists
from webdataset.pipeline import DataPipeline

from apps.bigmusic.umm.ar.datasets.mir.taxonomies.music_sft_en import MusicSFTTokenizerEN
from apps.bigmusic.umm.ar.datasets.mir.taxonomies.music_sft_zh import (
    MusicSFTTokenizerGenresZH,
    MusicSFTTokenizerMoodsZH,
    MusicSFTTokenizerScenesZH,
)
from apps.bigmusic.umm.ar.datasets.base import WebDataModuleBase
from samantha.dataio.parquet import ParquetDataset
from samantha.transforms.audio import Pad, RandomResizedCrop
from samantha.utils.webdataset import return_self


@dataclass
class MusicSFTDataResult:
    audio: torch.Tensor
    tag_ids: List[int]
    tag_names: List[str]
    song_id: str
    metadata: dict
    duration: float
    uttid: str
    dataset_name: str
    hidden_states: Optional[torch.Tensor] = None


class MusicSFTDataModule(WebDataModuleBase):
    torchaudio.set_audio_backend("soundfile")

    # pre-clipped: 963
    # raw: 881
    # _data_id: int = 963 # CN region
    _data_id: int = 112

    def __init__(
        self,
        tokenizer: MusicSFTTokenizerEN,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        self.tokenizer = tokenizer
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
        metadata = metadata["ori_meta"]
        mood = list(set([m for m in metadata["mood"].split(",")]))
        scene = list(set([s for s in metadata["scene"].split(",")]))
        theme = list(set([t for t in metadata["theme"].split(",")]))
        instruments = list(set([i for i in metadata["instrument"].split(",")]))
        genre_1 = list(set([g for g in metadata["genre_1"].split(",")]))
        genre_2 = list(set([g for g in metadata["genre_2"].split(",")]))
        return dict(
            mood=mood,
            scene=scene,
            theme=theme,
            instruments=instruments,
            genre_1=genre_1,
            genre_2=genre_2,
            voice_gender=metadata["voice_gender"],
            voice_char=metadata["voice_char"],
            duration=metadata["duration"],
            song_id=metadata["song_id"],
        )

    def transform(self, items: Iterable[Any]):
        for item in items:
            audio_bytes = BytesIO(item["wav"])
            audio, _ = torchaudio.load(audio_bytes)

            audio = audio.mean(dim=0, keepdim=True)


            for a in audio.split(self.duration_audio_samples, dim=-1)[:-1]:
                # audio = self.pad(audio)
                # audio = self.random_crop(audio)

                metadata = json.loads(item["meta"])
                metadata = self.extract_metadata(metadata)

                primary_genre = metadata["genre_1"][0]
                primary_mood = metadata["mood"][0]
                primary_scene = metadata["scene"][0]

                # if primary_genre not in self.tokenizer.vocab:
                #     continue

                # if primary_mood not in self.tokenizer.translation_keys:
                #     continue

                if primary_scene not in self.tokenizer.translation_keys:
                    continue

                # tags = set(
                #     metadata["genre_1"]
                #     + metadata["genre_2"]
                #     + metadata["mood"]
                #     + metadata["scene"]
                #     + metadata["theme"]
                # )
                # tags = list(filter(None, tags))

                tag_ids = self.tokenizer(primary_scene, device="cpu")
                tag_names = self.tokenizer.decode(tag_ids)

                yield MusicSFTDataResult(
                    audio=a,
                    tag_ids=tag_ids,
                    tag_names=tag_names,
                    song_id=metadata["song_id"],
                    metadata=metadata,
                    duration=metadata["duration"],
                    uttid=item["uttid"],
                    dataset_name=item["__dataset_name__"],
                )



class MusicSFTDataModuleZH(MusicSFTDataModule):
    # pre-clipped: -
    # raw: 1049
    
    # _data_id: int = 1049 # CN
    _data_id: int = 111 # US

    def __init__(
        self,
        tokenizer: MusicSFTTokenizerEN,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            duration=duration,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            resampled=resampled,
            shardshuffle=shardshuffle,
            num_workers=num_workers,
        )

    def extract_metadata(self, metadata: dict) -> dict:
        mood = metadata["mood"].split(",")
        scene = metadata["scene"].split(",")
        theme = metadata["theme"].split(",")
        instruments = metadata["instrument"].split(",")
        genre_1 = metadata["genre_1"].split(",")
        genre_2 = metadata["genre_2"].split(",")
        return dict(
            mood=mood,
            scene=scene,
            theme=theme,
            instruments=instruments,
            genre_1=genre_1,
            genre_2=genre_2,
            voice_gender=metadata["voice_gender"],
            voice_char=metadata["voice_char"],
            duration=metadata["duration"],
            song_id=metadata["song_id"],
        )


from apps.bigmusic.umm.ar.datasets.mir.taxonomies.music_sft_zh import MusicSFTTokenizerGenresAllZH


class MCC30KDataModuleZH(MusicSFTDataModule):
    _data_id: int = 116 # US

    def __init__(
        self,
        tokenizer: MusicSFTTokenizerGenresZH,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            duration=duration,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            resampled=resampled,
            shardshuffle=shardshuffle,
            num_workers=num_workers,
        )

        # required to map
        self.translator = MusicSFTTokenizerGenresAllZH()

    def extract_metadata(self, metadata: dict) -> dict:
        metadata["final_scene"] = None
        metadata["final_mood"] = None
        metadata["final_genre"] = None
        
        # get scene/mood/genre
        if metadata["playlist_extra"]["label1"] == "中文场景":
            metadata["final_scene"] = metadata["playlist_extra"]["label2"]
        elif metadata["playlist_extra"]["label1"] == "中文心情":
            metadata["final_mood"] = metadata["playlist_extra"]["label2"]
        else:       
            metadata["final_genre_1"] = self.translator.translate(metadata["playlist_extra"]["label1"])

            if metadata["playlist_extra"]["label2"] is not None:
                metadata["final_genre_2"] = self.translator.translate(metadata["playlist_extra"]["label2"])
                metadata["final_genre"] = metadata["final_genre_1"] + ", " + metadata["final_genre_2"]
            else:
                metadata["final_genre"] = metadata["final_genre_1"]

        return dict(
            mood=[metadata["final_mood"]],
            scene=[metadata["final_scene"]],
            genre_1=[metadata["final_genre"]],
            duration=[metadata["duration"]],
            song_id=[metadata["playlist_id"]],
        )

class MusicSFTVocalDataModuleEN(MusicSFTDataModule):

    def __init__(
        self,
        tokenizer: MusicSFTTokenizerEN,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            duration=duration,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            resampled=resampled,
            shardshuffle=shardshuffle,
            num_workers=num_workers,
        )

    def transform(self, items: Iterable[Any]):
        for item in items:
            audio_bytes = BytesIO(item["wav"])
            audio, _ = torchaudio.load(audio_bytes)

            audio = audio.mean(dim=0, keepdim=True)

            audio = self.pad(audio)
            audio = self.random_crop(audio)

            metadata = json.loads(item["meta"])
            metadata = self.extract_metadata(metadata)

            if metadata["voice_gender"] not in self.tokenizer.translation_keys:
                continue

            tag_ids = self.tokenizer(metadata["voice_gender"], device="cpu")
            tag_names = self.tokenizer.decode(tag_ids)

            yield MusicSFTDataResult(
                audio=audio,
                tag_ids=tag_ids,
                tag_names=tag_names,
                song_id=metadata["song_id"],
                metadata=metadata,
                duration=metadata["duration"],
                uttid=item["uttid"],
                dataset_name=item["__dataset_name__"],
            )


from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset


class MusicSFTPreprocessedDataModule(WebDataModuleBase):

    def __init__(
        self,
        tokenizer: MusicSFTTokenizerEN,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        self.tokenizer = tokenizer
        self._sample_rate = sample_rate
        self._duration = duration

        print("initializing train datasets...")
        train_datasets = []
        for tag in tokenizer.vocab:
            url2index = f"/mnt/bn/janne-research-xl/data/music_sft/english/primary_genre/{tag}/url2index.txt"
            print(f"adding dataset: {url2index}")

            dataset = IndexedWebDataset(url2index, resampled=resampled, shardshuffle=shardshuffle).decode().compose(self.transform)
            train_datasets.append(dataset)

        train_dataset = DataPipeline(MultiIterableDataset(train_datasets))


        # TODO: these are the same at the moment :(
        print("initializing validation datasets...")
        validation_datasets = []
        for tag in tokenizer.vocab:
            url2index = f"/mnt/bn/janne-research-xl/data/music_sft/english/primary_genre/{tag}/url2index.txt"
            print(f"adding dataset: {url2index}")

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

    @property
    def duration_audio_samples(self) -> int:
        return self._sample_rate * self._duration

    def transform(self, items: Iterable[Any]):
        for item in items:
            index = item["__index_data__"]

            audio = torch.tensor(item["audio.npy"])
            hidden_states = torch.tensor(item["hidden_states.npy"])

            tag_names = index["tag_names"]

            tag_ids = self.tokenizer(tag_names, device="cpu")

            yield MusicSFTDataResult(
                audio=audio,
                hidden_states=hidden_states,
                tag_ids=tag_ids,
                tag_names=tag_names,
                song_id=index["song_id"],
                metadata=index["metadata"],
                duration=index["duration"],
                uttid=index["uttid"],
                dataset_name=index["dataset_name"],
            )


class MusicSFTVocalPreprocessedDataModule(WebDataModuleBase):

    def __init__(
        self,
        tokenizer: MusicSFTTokenizerEN,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        self.tokenizer = tokenizer
        self._sample_rate = sample_rate
        self._duration = duration

        print("initializing train datasets...")
        train_datasets = []
        for tag in tokenizer.vocab:
            url2index = f"/mnt/bn/janne-research-xl/data/music_sft/english/vocal_gender/{tag}/url2index.txt"
            print(f"adding dataset: {url2index}")

            dataset = IndexedWebDataset(url2index, resampled=resampled, shardshuffle=shardshuffle).decode().compose(self.transform)
            train_datasets.append(dataset)

        train_dataset = DataPipeline(MultiIterableDataset(train_datasets))


        # TODO: these are the same at the moment :(
        print("initializing validation datasets...")
        validation_datasets = []
        for tag in tokenizer.vocab:
            url2index = f"/mnt/bn/janne-research-xl/data/music_sft/english/vocal_gender/{tag}/url2index.txt"
            print(f"adding dataset: {url2index}")

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

    @property
    def duration_audio_samples(self) -> int:
        return self._sample_rate * self._duration

    def transform(self, items: Iterable[Any]):
        for item in items:
            index = item["__index_data__"]

            audio = torch.tensor(item["audio.npy"])
            hidden_states = torch.tensor(item["hidden_states.npy"])

            tag_names = index["tag_names"]

            tag_ids = self.tokenizer(tag_names, device="cpu")

            yield MusicSFTDataResult(
                audio=audio,
                hidden_states=hidden_states,
                tag_ids=tag_ids,
                tag_names=tag_names,
                song_id=index["song_id"],
                metadata=index["metadata"],
                duration=index["duration"],
                uttid=index["uttid"],
                dataset_name=index["dataset_name"],
            )

class MusicSFTMCC30KPreprocessedDataModule(WebDataModuleBase):

    def __init__(
        self,
        tokenizer: MusicSFTTokenizerScenesZH,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int,
    ):
        self.tokenizer = tokenizer
        self._sample_rate = sample_rate
        self._duration = duration

        print("initializing train datasets...")
        train_datasets = []
        for tag in tokenizer.vocab:
            url2index = f"/mnt/bn/janne-research-xl/data/music_sft/chinese/genre_1/{tag}/url2index.txt"
            print(f"adding dataset: {url2index}")

            dataset = IndexedWebDataset(url2index, resampled=resampled, shardshuffle=shardshuffle).decode().compose(self.transform)
            train_datasets.append(dataset)

        train_dataset = DataPipeline(MultiIterableDataset(train_datasets))


        # TODO: these are the same at the moment :(
        print("initializing validation datasets...")
        validation_datasets = []
        for tag in tokenizer.vocab:
            url2index = f"/mnt/bn/janne-research-xl/data/music_sft/chinese/genre_1/{tag}/url2index.txt"
            print(f"adding dataset: {url2index}")

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

    @property
    def duration_audio_samples(self) -> int:
        return self._sample_rate * self._duration

    def transform(self, items: Iterable[Any]):
        for item in items:
            index = item["__index_data__"]

            audio = torch.tensor(item["audio.npy"])

            tag_names = index["tag_names"]

            tag_ids = self.tokenizer(tag_names, device="cpu")

            yield MusicSFTDataResult(
                audio=audio,
                tag_ids=tag_ids,
                tag_names=tag_names,
                song_id=index["song_id"],
                metadata=index["metadata"],
                duration=index["duration"],
                uttid=index["uttid"],
                dataset_name=index["dataset_name"],
            )


