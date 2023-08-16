import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import soundfile as sf
import torch
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
from samantha.utils.hdfs_tools import hdfs_loadtxt, hdfs_open
from samantha.utils.webdataset import return_self

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


def load_librilight_metadata(filepath: Path, ext_metadata: str):
    file_metadata = filepath.parent / (filepath.stem + ext_metadata)
    with open(file_metadata) as f:
        metadata = json.load(f)
    speaker_id = metadata["speaker"]
    book_id = metadata["book_meta"]["id"]
    vad = metadata["voice_activity"]
    return {"speaker_id": speaker_id, "book_id": book_id, "vad": vad}


def load_librilight_item(filepath: Path, ext_audio: str, ext_metadata: str):
    file_audio = filepath.parent / (filepath.stem + ext_audio)

    waveform, _ = _load_waveform(file_audio, SAMPLE_RATE)
    metadata = load_librilight_metadata(filepath, ext_metadata)
    return {"audio": waveform, **metadata}


def load_preprocessed_librilight_item(filepath: Path, ext_audio: str):
    file_audio = filepath.parent / (filepath.stem + ext_audio)

    filename_split = filepath.stem.split("_")

    speaker_id = filepath.parent.parent.stem
    book_id = filepath.parent.stem
    chapter_id = "_".join(filename_split[0:-2])
    utterance_id = filename_split[-2]
    utterance_sub_id = filename_split[-1]

    waveform, _ = _load_waveform(file_audio, SAMPLE_RATE)

    return {
        "audio": waveform,
        "speaker_id": speaker_id,
        "book_id": book_id,
        "chapter_id": chapter_id,
        "utterance_id": utterance_id,
        "utterance_sub_id": utterance_sub_id,
    }


def cut_sequence(
    path: str, vad, path_out: Path, max_seq_len: int, out_extension: str
) -> None:
    """Copied from original repository:
    https://github.com/facebookresearch/libri-light/blob/main/data_preparation/cut_by_vad.py
    """

    def save(seq, fname, index, extension):
        ## NOTE: Original code saves the list of slices into a single file
        ## This will prevent us to bucket, so instead we'll save all  the
        ## separte utterances (separated by VAD) in seperate files.

        ## ORIGINAL
        # output = np.hstack(seq)
        # file_name = fname.parent / (fname.stem + f"_{index:04}{extension}")
        # fname.parent.mkdir(exist_ok=True, parents=True)
        # sf.write(str(file_name), output, samplerate=16000)
        ##

        for utt_sub_idx, output in enumerate(seq):
            file_name = fname.parent / (
                fname.stem + f"_{index:04}_{utt_sub_idx:04}{extension}"
            )
            fname.parent.mkdir(exist_ok=True, parents=True)
            sf.write(str(file_name), output, samplerate=16000)

    data, samplerate = sf.read(path)
    assert len(data.shape) == 1
    assert samplerate == SAMPLE_RATE

    to_stitch = []
    length_accumulated = 0.0

    i = 0
    for start, end in vad:
        start_index = int(start * samplerate)
        end_index = int(end * samplerate)
        slice = data[start_index:end_index]

        # if a slice is longer than target_len_sec, we put it entirely in it's own piece
        if length_accumulated + (end - start) > max_seq_len and length_accumulated > 0:
            save(to_stitch, path_out, i, out_extension)
            to_stitch = []
            i += 1
            length_accumulated = 0

        to_stitch.append(slice)
        length_accumulated += end - start

    if to_stitch:
        save(to_stitch, path_out, i, out_extension)


def preprocess_librilight(
    out_dir: str,
    root: str,
    split: str,
    max_len_sec: int,
    out_extension: str = ".flac",
    n_processes: int = 16,
):
    root = Path(root)
    dataset = LibriLightRawDataset(root=root, split=split)

    args = []
    for fp in tqdm(dataset._walker, desc="Preparing arguments for multi-processing"):
        metadata = load_librilight_metadata(fp, ".json")
        path_out = Path(
            os.path.join(
                out_dir, split, metadata["speaker_id"], metadata["book_id"], fp.stem
            )
        )
        args.append((str(fp), metadata["vad"], path_out, max_len_sec, out_extension))

    with Pool(processes=n_processes) as pool:
        pool.starmap(cut_sequence, tqdm(args, total=len(dataset._walker)))


class LibriLightRawDataset(Dataset):
    """LibriLight raw dataset.
    Args:
    root (str or Path): Path to the directory where the dataset is found or downloaded.
    split (str, optional): The split to use (small, medium, large)
    """

    _ext_audio = ".flac"
    _ext_metadata = ".json"

    def __init__(self, root: str, split: str) -> None:
        root = os.fspath(root)
        self._archive = root
        self._path = os.path.join(root, split)
        self._split = split

        if not os.path.isdir(self._path):
            raise RuntimeError(f"Dataset not found at {self._path}.")

        self._walker = [p for p in Path(self._path).glob("*/*/*" + self._ext_audio)]
        self.total = len(self._walker)

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
                Speaker ID
            int:
                Book ID
            Tensor:
                Voice activity detection (VAD)
        """
        filepath = self._walker[n]
        return load_librilight_item(filepath, self._ext_audio, self._ext_metadata)


class LibriLightDataset(Dataset):
    """LibriLight dataset.
    Args:
    root (str or Path): Path to the directory where the dataset is found or downloaded.
    split (str, optional): The split to use (small, medium, large)
    """

    _ext_audio = ".flac"
    _ext_metadata = ".json"

    def __init__(self, root: str, split: str) -> None:
        root = os.fspath(root)
        self._archive = root
        self._path = os.path.join(root, split)
        self._split = split

        if not os.path.isdir(self._path):
            raise RuntimeError(f"Dataset not found at {self._path}.")

        self._walker = [p for p in Path(self._path).glob("*/*/*" + self._ext_audio)]
        self.total = len(self._walker)

    def __len__(self):
        return self.total

    def __getitem__(self, n: int) -> Dict[str, Any]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Speaker ID
            int:
                Book ID
            Tensor:
                Voice activity detection (VAD)
        """
        filepath = self._walker[n]
        return load_preprocessed_librilight_item(filepath, self._ext_audio)


def librilight_collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)

    audio = []
    speaker_id = []
    book_id = []
    chapter_id = []
    utterance_id = []
    utterance_sub_id = []
    shard = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        speaker_id.append(batch[idx]["speaker_id"])
        book_id.append(batch[idx]["book_id"])
        chapter_id.append(batch[idx]["chapter_id"])
        utterance_id.append(batch[idx]["utterance_id"])
        utterance_sub_id.append(batch[idx]["utterance_sub_id"])
        shard.append(batch[idx]["shard"])

    return {
        "audio": torch.stack(audio),
        "speaker_id": speaker_id,
        "chapter_id": chapter_id,
        "utterance_id": utterance_id,
        "utterance_sub_id": utterance_sub_id,
        "shard": shard,
    }


def write_index(hdfs_fp: str, index: List[str]):
    with hdfs_open(hdfs_fp, "w") as f:
        f.write("\n".join(index))


class LibriLightWebDataModule(BaseDataModule):
    data_sample_rate = SAMPLE_RATE

    def __init__(
        self,
        sample_rate: int,
        split: str,
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
        use_bucket_batcher: Optional[bool] = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        shardshuffle: bool = True,
        duration: Optional[float] = None,
        collate_fn: Optional[Callable] = librilight_collate_fn,
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

        self.split = split
        self.duration = duration

        train_shards, valid_shards = self.get_hdfs_shard_uri(split)
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

    def load_transcriptions(self):
        uri = self.get_hdfs_transcription_uri()
        logger.warn(f"Loading transcription file from HDFS: {uri}")
        transcriptions = hdfs_open(uri, "r").read()
        return transcriptions

    @staticmethod
    def get_hdfs_transcription_uri() -> str:
        return "hdfs://haruna/home/byte_speech_sv/data/speech/librilight/librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85.txt"

    @staticmethod
    def get_hdfs_shard_uri(split: str) -> Tuple[str, str]:
        if split == "small":
            train_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/small/{00000..00013}.tar"
            valid_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/small/00014.tar"
        elif split == "medium":
            train_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/medium/{00000..00126}.tar"
            valid_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/medium/00127.tar"
        elif split == "large":
            train_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/large/{00000..00932}.tar"
            valid_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/large/00933.tar"
        elif split == "large2":
            train_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/large2/{00000..01650}.tar"
            valid_shards = "pipe: hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/speech/librilight/large2/01651.tar"
        else:
            raise NotImplementedError("Choose between `small, medium`")
        return train_shards, valid_shards

    @staticmethod
    def create_webdataset(
        dataset: LibriLightDataset, pattern: str, maxsize: int, start_shard_idx: int
    ):
        writer = ShardWriter(
            pattern=pattern, maxsize=maxsize, start_shard=start_shard_idx
        )
        index = []
        current_shard = writer.shard
        for idx, item in enumerate(tqdm(dataset)):
            if current_shard != writer.shard:
                index_fp = os.path.join(
                    os.path.dirname(writer.fname), f"{current_shard-1:05d}.tar.index"
                )
                print(f"Writing index: {index_fp}")
                write_index(index_fp, index)
                current_shard = writer.shard
                index = []

            id = f"{idx}-{item['speaker_id']}-{item['book_id']}-{item['chapter_id']}-{item['utterance_id']}-{item['utterance_sub_id']}"

            # item["audio"] = fp32_to_int16(item["audio"])
            obj = {
                "__key__": id,
                "audio.npy": item["audio"].numpy(),
                "metadata.json": {
                    "speaker_id": item["speaker_id"],
                    "book_id": item["book_id"],
                    "chapter_id": item["chapter_id"],
                    "utterance_id": item["utterance_id"],
                    "utterance_sub_id": item["utterance_sub_id"],
                },
            }
            writer.write(obj)
            index.append(id)

        index_fp = os.path.join(
            os.path.dirname(writer.fname), f"{current_shard-1:05d}.tar.index"
        )
        print(f"Writing index: {index_fp}")
        write_index(index_fp, index)
        writer.close()

    @property
    def n_audio_samples(self):
        return int(self.duration * self.sample_rate)

    def wds_transform(self, item) -> Dict[str, Any]:
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
            "speaker_id": int(item["metadata.json"]["speaker_id"]),
            "book_id": int(item["metadata.json"]["book_id"]),
            "chapter_id": item["metadata.json"]["chapter_id"],
            "utterance_id": item["metadata.json"]["utterance_id"],
            "utterance_sub_id": item["metadata.json"]["utterance_sub_id"],
            "shard": shard,
        }

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


# TODO: Parallel sharding:
# @staticmethod
# def create_webdataset(dataset: LibriLightDataset, pattern: str, max_count: int):

#     ranges = list(range(0, len(dataset), max_count))
#     if ranges[-1] != len(dataset):
#         ranges.append(len(dataset))

#     min_idxs = ranges[::2]
#     max_idxs = ranges[1::2]
#     assert len(min_idxs) == len(max_idxs)

#     def upload_dataset_to_shard(dataset, pattern, shard_idx: int, min_idx: int, max_idx: int) -> None:

#         def limit_dataset(d: LibriLightDataset, min_idx: int, max_idx: int):
#             d = deepcopy(d)
#             d._walker = d._walker[min_idx:max_idx]
#             d.total = len(d._walker)
#             return d

#         writer = ShardWriter(
#             pattern=pattern,
#             maxsize=9e99, # inf, this is essentially a tarwriter
#             maxcount=9e99, # inf, this is essentially a tarwriter
#             start_shard=shard_idx,
#         )

#         dataset_range = limit_dataset(dataset, min_idx, max_idx)
#         for idx, item in enumerate(tqdm(dataset_range)):
#             id = f"{item['speaker_id']}-{item['book_id']}-{item['chapter_id']}-{item['utterance_id']}-{item['utterance_sub_id']}"

#             # item["audio"] = fp32_to_int16(item["audio"])
#             obj = {
#                 "__key__": id,
#                 "audio.npy": item["audio"].numpy(),
#                 "metadata.json": {
#                     "speaker_id": item["speaker_id"],
#                     "book_id": item["book_id"],
#                     "chapter_id": item["chapter_id"],
#                     "utterance_id": item["utterance_id"],
#                     "utterance_sub_id": item["utterance_sub_id"],
#                 }
#             }
#             writer.write(obj)
#         writer.close()

#     Parallel(n_jobs=10)(delayed(upload_dataset_to_shard)(dataset, pattern, shard_idx, min_idx, max_idx) for shard_idx, (min_idx, max_idx) in enumerate(tqdm(zip(min_idxs, max_idxs), total=len(min_idxs))))
#     # for shard_idx, (min_idx, max_idx) in enumerate(zip(min_idxs, max_idxs)):
#     #     upload_dataset_to_shard(dataset, pattern, shard_idx, min_idx, max_idx)
