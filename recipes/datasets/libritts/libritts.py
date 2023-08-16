import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import Dataset
from torchaudio_augmentations import Compose
from tqdm import tqdm
from webdataset import WebDataset

from recipes.datasets.base import BaseDataModule, _load_waveform
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset import ShardWriter
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    fp32_to_int16,
)
from samantha.transforms.utils import get_transform_version
from samantha.utils.webdataset import return_self

SAMPLE_RATE = 24000
FOLDER_IN_ARCHIVE = "LibriTTS"


def load_libritts_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
    ext_normalized_txt: str,
):
    speaker_id, chapter_id, segment_id, utterance_id = fileid.split("_")
    utterance_id = fileid

    normalized_text = utterance_id + ext_normalized_txt
    normalized_text = os.path.join(path, speaker_id, chapter_id, normalized_text)

    original_text = utterance_id + ext_original_txt
    original_text = os.path.join(path, speaker_id, chapter_id, original_text)

    file_audio = utterance_id + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = _load_waveform(file_audio, SAMPLE_RATE)

    # Load original text
    with open(original_text) as ft:
        original_text = ft.readline()

    # Load normalized text
    with open(normalized_text, "r") as ft:
        normalized_text = ft.readline()

    return {
        "audio": waveform,
        "sample_rate": sample_rate,
        "original_text": original_text,
        "normalized_text": normalized_text,
        "speaker_id": speaker_id,
        "chapter_id": chapter_id,
        "utterance_id": utterance_id,
    }


class LibriTTSDataset(Dataset):
    """LibriTTS dataset.
    Args:
    root (str or Path): Path to the directory where the dataset is found or downloaded.
    split (str, optional): The split to use,
        or the type of the dataset to dowload.
        Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
        ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
        ``"train-other-500"``. (default: ``"train-clean-100"``)
    folder_in_archive (str, optional):
        The top-level directory of the dataset. (default: ``"LibriTTS"``)
    """

    _ext_original_txt = ".original.txt"
    _ext_normalized_txt = ".normalized.txt"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
        split: str = "train-clean-360",
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
    ) -> None:
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)
        self._path = os.path.join(root, folder_in_archive, split)
        self._split = split

        if not os.path.isdir(self._path):
            raise RuntimeError(f"Dataset not found at {self._path}.")

        self._walker = sorted(
            str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio)
        )
        self.total = len(self._walker)

    def random_shuffle(self, seed: int = 42):
        random.seed(seed)
        random.shuffle(self._walker)

    def __len__(self):
        return self.total

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        fileid = self._walker[n]
        data = load_libritts_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_original_txt,
            self._ext_normalized_txt,
        )
        return {
            "audio": data["audio"],
            "original_text": data["original_text"],
            "normalized_text": data["normalized_text"],
            "speaker_id": data["speaker_id"],
            "chapter_id": data["chapter_id"],
            "utterance_id": data["utterance_id"],
        }


def librispeech_collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)

    audio = []
    original_text = []
    normalized_text = []
    speaker_id = []
    utterance_id = []
    chapter_id = []
    shard = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        original_text.append(batch[idx]["original_text"])
        normalized_text.append(batch[idx]["normalized_text"])
        speaker_id.append(batch[idx]["speaker_id"])
        chapter_id.append(batch[idx]["chapter_id"])
        utterance_id.append(batch[idx]["utterance_id"])
        shard.append(batch[idx]["shard"])

    return {
        "audio": torch.stack(audio),
        "original_text": original_text,
        "normalized_text": normalized_text,
        "speaker_id": speaker_id,
        "chapter_id": chapter_id,
        "utterance_id": utterance_id,
        "shard": shard,
    }


class LibriTTSWebDataModule(BaseDataModule):
    data_sample_rate = SAMPLE_RATE

    def __init__(
        self,
        sample_rate: int,
        batch_size: int,
        shuffle_buffer_size: int,
        buckets_sec: List[int] = [
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
        ],
        train_shards: str = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar",
        valid_shards: str = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar",
        use_bucket_batcher: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        shardshuffle: bool = True,
        duration: Optional[float] = None,
        collate_fn: Optional[Callable] = librispeech_collate_fn,
    ):
        batcher = None
        self.use_bucket_batcher = use_bucket_batcher
        if use_bucket_batcher:
            if duration is not None:
                raise Exception(
                    "duration must be set to `None` when using BucketBatcher"
                )

            buckets_samples = list(map(lambda i: i * sample_rate, buckets_sec))
            batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:
            if duration is None:
                raise Exception("duration must be set when not using BucketBatcher")

        self.duration = duration
        train_dataset = WebDataset(
            urls=train_shards, resampled=resampled, shardshuffle=shardshuffle
        )
        validation_dataset = WebDataset(urls=valid_shards, nodesplitter=return_self)

        pipeline = []
        pipeline.append("decode")
        pipeline.append({"map": [self.wds_transform]})

        if use_bucket_batcher:
            pipeline.append({"compose": [self.bucketize]})

        train_dataset = WebPipeline(train_dataset, pipeline)
        predict_dataset = train_dataset  # TODO
        validation_dataset = WebPipeline(validation_dataset, pipeline)

        super().__init__(
            sample_rate=sample_rate,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=predict_dataset,
            batcher=batcher,
            collate_fn=collate_fn,
        )
        self.base_transform = Compose(
            [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        )

        # if we have a BucketBatcher, don't do random crop/padding
        # the bucket will take care of this.
        if use_bucket_batcher:
            self.max_audio_samples = buckets_samples[-1]
        else:
            self.random_pad = RandomPad(self.n_audio_samples)
            self.random_crop = RandomResizedCrop(self.n_audio_samples)

    def create_webdataset(self, dataset: LibriTTSDataset, pattern: str, maxsize: int):
        writer = ShardWriter(pattern=pattern, maxsize=maxsize)

        for idx, item in enumerate(tqdm(dataset)):
            id = f"{item['speaker_id']}-{item['chapter_id']}-{item['utterance_id']}"

            item["audio"] = self.resample(item["audio"])

            # item["audio"] = fp32_to_int16(item["audio"])
            obj = {
                "__key__": id,
                "audio.npy": item["audio"].numpy(),
                "original_text.txt": item["original_text"],
                "normalized_text.txt": item["normalized_text"],
                "metadata.json": {
                    "speaker_id": item["speaker_id"],
                    "utterance_id": item["utterance_id"],
                    "chapter_id": item["chapter_id"],
                },
            }
            writer.write(obj)
        writer.close()

    @property
    def n_audio_samples(self):
        return int(self.duration * self.sample_rate)

    def wds_transform(self, item):
        audio = item["audio.npy"]
        audio = self.base_transform(audio)

        if self.use_bucket_batcher:
            if audio.shape[1] > self.max_audio_samples:
                audio = audio[
                    :, : self.max_audio_samples
                ]  # TODO: Revise trimming for long samples
        else:
            audio = self.random_pad(audio)
            audio = self.random_crop(audio)

        shard = os.path.basename(item["__url__"])
        return {
            "audio": audio,
            "original_text": item["original_text.txt"],
            "normalized_text": item["normalized_text.txt"],
            "speaker_id": int(item["metadata.json"]["speaker_id"]),
            "utterance_id": int(item["metadata.json"]["utterance_id"]),
            "chapter_id": int(item["metadata.json"]["chapter_id"]),
            "shard": shard,
        }

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch
