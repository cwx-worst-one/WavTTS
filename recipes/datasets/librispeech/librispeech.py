import os
import random
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import torch
from julius.resample import ResampleFrac
from torch.utils.data import Dataset
from tqdm import tqdm
from webdataset import WebDataset

from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo, ShardInfo
from samantha.data.base import BaseAudioTransform, WebDataModuleBase, _load_waveform
from samantha.data.lyrics.lyrics import LyricChunk, Lyrics
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset import ShardWriter
from samantha.transforms.audio import Pad, RandomResizedCrop
from samantha.utils.webdataset import return_self

SAMPLE_RATE = 16000
FOLDER_IN_ARCHIVE = "LibriSpeech"


def load_librispeech_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
):
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    # Get audio path and sample rate
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
    file_audio = os.path.join(path, speaker_id, chapter_id, f"{fileid_audio}{ext_audio}")

    # Load audio
    waveform, sample_rate = _load_waveform(file_audio, SAMPLE_RATE)
    
    # Load text
    file_text = f"{speaker_id}-{chapter_id}{ext_original_txt}"
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)
    with open(file_text) as ft:
        for line in ft:
            fileid_text, original_text = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {fileid_audio}")

    return {
        "audio": waveform,
        "sample_rate": sample_rate,
        "original_text": original_text,
        "speaker_id": speaker_id,
        "chapter_id": chapter_id,
        "utterance_id": utterance_id,
    }


class LibriSpeechDataset(Dataset):
    """LibriSpeechDataset dataset.
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

    _ext_original_txt = ".trans.txt"
    _ext_audio = ".flac"

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
        data = load_librispeech_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_original_txt,
        )
        return {
            "audio": data["audio"],
            "original_text": data["original_text"],
            "speaker_id": data["speaker_id"],
            "chapter_id": data["chapter_id"],
            "utterance_id": data["utterance_id"],
        }


class LibriSpeechWebDataModule(WebDataModuleBase):
    data_sample_rate = SAMPLE_RATE

    def __init__(
        self,
        sample_rate: int,
        batch_size: int,
        shuffle_buffer_size: int,
        buckets_sec: List[int],
        use_bucket_batcher: bool,
        train_shards: str = "pipe: hdfs dfs -cat hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/speech/librispeech/train-clean-360/{00000..00009}.tar",
        valid_shards: str = "pipe: hdfs dfs -cat hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/speech/librispeech/test-clean/00000.tar",
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        shardshuffle: bool = True,
        duration: Optional[float] = None,
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

        self.sample_rate = sample_rate
        self.batcher = batcher
        self.duration = duration
        self.base_transform = BaseAudioTransform()

        if self.data_sample_rate != self.sample_rate:
            self.resample = ResampleFrac(self.data_sample_rate, self.sample_rate)

        # if we have a BucketBatcher, don't do random crop/padding
        # the bucket will take care of this.
        if use_bucket_batcher:
            self.min_audio_samples = buckets_samples[0]
            self.max_audio_samples = buckets_samples[-1]
        else:
            self.random_pad = Pad(self.n_audio_samples, value=0.0)
            self.random_crop = RandomResizedCrop(self.n_audio_samples)
        
        train_dataset = WebDataset(
            urls=train_shards, resampled=resampled, shardshuffle=shardshuffle
        )
        validation_dataset = WebDataset(urls=valid_shards, nodesplitter=return_self)

        train_dataset = train_dataset.decode().compose(self.transform)
        validation_dataset = validation_dataset.decode().compose(self.transform)
        predict_dataset = train_dataset  # TODO
        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=validation_dataset,
            predict_dataset=predict_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batcher=batcher,
        )

    def collate_fn(self, batch):
        batch_size = len(batch)

        collated = AudioDataResult(
            audio=[],
            segment_info=[],
            index=[],
            shard=[],
            key=[],
            shard_info=[],
        )

        input_length = []
        for res in batch:
            input_length.append(res.audio.shape[-1])
            collated.audio.append(res.audio)
            collated.segment_info.append(res.segment_info)
            collated.index.append(res.index)
            collated.shard.append(res.shard)
            collated.key.append(res.key)
            collated.shard_info.append(res.shard_info)

        max_length = max(input_length)
        pad = Pad(max_length, value=0.0)
        for i in range(batch_size):
            audio = collated.audio[i]
            collated.audio[i] = pad(audio)

        collated.audio = torch.stack(collated.audio, dim=0)
        return collated

    def create_webdataset(self, dataset: LibriSpeechDataset, pattern: str, maxsize: int):
        writer = ShardWriter(pattern=pattern, maxsize=maxsize)

        for idx, item in enumerate(tqdm(dataset)):
            id = f"{item['speaker_id']}-{item['chapter_id']}-{item['utterance_id']}"

            # item["audio"] = fp32_to_int16(item["audio"])
            obj = {
                "__key__": id,
                "audio.npy": item["audio"].numpy(),
                "original_text.txt": item["original_text"],
                "metadata.json": {
                    "speaker_id": item["speaker_id"],
                    "chapter_id": item["chapter_id"],
                    "utterance_id": item["utterance_id"],
                },
            }
            writer.write(obj)
        writer.close()

    @property
    def n_audio_samples(self):
        return int(self.duration * self.sample_rate)

    def transform(self, items: Iterable) -> Generator:
        for item in items:
            audio = item["audio.npy"]
            audio = self.base_transform(audio)

            if self.data_sample_rate != self.sample_rate:
                audio = self.resample(audio)

            n_frames = audio.shape[1]
            duration = n_frames / self.sample_rate

            if self.use_bucket_batcher:
                if audio.shape[1] < self.min_audio_samples or audio.shape[1] > self.max_audio_samples:
                    continue
            else:
                audio = self.random_pad(audio)
                audio = self.random_crop(audio)
            

            key = item["__key__"]

            lyrics = Lyrics()
            lyrics.add_chunk(LyricChunk(start_time=0, end_time=duration, text=item["original_text.txt"], language="en", confidence=1.0))
            
            segment_info = SegmentInfo(
                data_type="speech",
                meta=AudioMeta(path=key, duration=duration, sample_rate=self.sample_rate),
                seek_time=0,
                n_frames=n_frames,
                total_frames=audio.shape[-1],
                sample_rate=self.sample_rate,
                channels=audio.shape[0],
                lyrics=lyrics,
            )

            index = {
                "original_text": item["original_text.txt"],
                "speaker_id": int(item["metadata.json"]["speaker_id"]),
                "utterance_id": int(item["metadata.json"]["utterance_id"]),
                "chapter_id": int(item["metadata.json"]["chapter_id"]),
            }

            shard = os.path.basename(item["__url__"])
            yield AudioDataResult(
                shard=shard,
                key=key,
                audio=audio,
                segment_info=segment_info,
                index=index,
                shard_info=ShardInfo(url=shard),
            )

if __name__ == "__main__":

    datamodule = LibriSpeechWebDataModule(
        sample_rate=16000,
        batch_size=8,
        shuffle_buffer_size=0,
        buckets_sec=[2, 4, 6, 8, 10],
        use_bucket_batcher=True,
    )

    dataloader = iter(datamodule.train_dataloader())

    print(next(dataloader))
