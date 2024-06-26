# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AudioDataset support. In order to handle a larger number of files
without having to scan again the folders, we precompute some metadata
(filename, sample rate, duration), and use that to efficiently sample audio segments.
"""
import argparse
import copy
import gzip
import json
import logging
import os
import random
import sys
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ChainDataset, ConcatDataset, IterableDataset
from torchaudio_augmentations import Compose
from webdataset.shardlists import split_by_node

from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo, ShardInfo
from samantha.data.audio.webdataset import FeatureWebDataset
from samantha.data.audio_utils import convert_audio
from samantha.data.av_audio import audio_info, audio_read
from samantha.data.base import LightningDataModuleBase
from samantha.dataio.dataset import MultiIterableDataset
from samantha.transforms.audio import Mono, Pad, RescaleAudio
from samantha.utils.logger import RankedLogger

logger = RankedLogger(rank_zero_only=True)


DEFAULT_EXTS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

logger = logging.getLogger(__name__)


def _get_audio_meta(file_path: str, minimal: bool = True) -> AudioMeta:
    """AudioMeta from a path to an audio file.

    Args:
        file_path (str): Resolved path of valid audio file.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
    Returns:
        AudioMeta: Audio file path and its metadata.
    """
    info = audio_info(file_path)
    amplitude: Optional[float] = None
    if not minimal:
        wav, sr = audio_read(file_path)
        amplitude = wav.abs().max().item()
    return AudioMeta(file_path, info.duration, info.sample_rate, amplitude)


def find_audio_files(
    path: Union[Path, str],
    exts: List[str] = DEFAULT_EXTS,
    minimal: bool = True,
    progress: bool = False,
    workers: int = 0,
    n_samples: Optional[int] = None,
) -> List[AudioMeta]:
    """Build a list of AudioMeta from a given path,
    collecting relevant audio files and fetching meta info.

    Args:
        path (str or Path): Path to folder containing audio files.
        exts (list of str): List of file extensions to consider for audio files.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
        progress (bool): Whether to log progress on audio files collection.
        workers (int): number of parallel workers, if 0, use only the current thread.
    Returns:
        list of AudioMeta: List of audio file path and its metadata.
    """
    audio_files = []
    futures: List[Future] = []
    pool: Optional[ThreadPoolExecutor] = None
    with ExitStack() as stack:
        if workers > 0:
            pool = ThreadPoolExecutor(workers)
            stack.enter_context(pool)

        if progress:
            print("Finding audio files...")
        for root, folders, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = Path(root) / file
                if full_path.suffix.lower() in exts:
                    audio_files.append(full_path)
                    if pool is not None:
                        futures.append(
                            pool.submit(_get_audio_meta, str(audio_files[-1]), minimal)
                        )
                    if progress:
                        print(
                            format(len(audio_files), " 8d"), end="\r", file=sys.stderr
                        )

        if n_samples is not None:
            audio_files = audio_files[:n_samples]

        if progress:
            print("Getting audio metadata...", len(audio_files))

        meta: List[AudioMeta] = []
        for idx, file_path in enumerate(audio_files):
            try:
                if pool is None:
                    m = _get_audio_meta(str(file_path), minimal)
                else:
                    m = futures[idx].result()
            except Exception as err:
                print("Error with", str(file_path), err, file=sys.stderr)
                continue
            meta.append(m)
            if progress:
                print(
                    format((1 + idx) / len(audio_files), " 3.1%"),
                    end="\r",
                    file=sys.stderr,
                )
    meta.sort()
    return meta


def load_audio_meta(path: Union[str, Path], fast: bool = True) -> List[AudioMeta]:
    """Load list of AudioMeta from an optionally compressed json file.

    Args:
        path (str or Path): Path to JSON file.
        fast (bool): activates some tricks to make things faster.
    Returns:
        list of AudioMeta: List of audio file path and its total duration.
    """
    open_fn = gzip.open if str(path).lower().endswith(".gz") else open
    with open_fn(path, "rb") as fp:  # type: ignore
        lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        m = AudioMeta.from_dict(d)
        meta.append(m)
    return meta


def save_audio_meta(path: Union[str, Path], meta: List[AudioMeta]):
    """Save the audio metadata to the file pointer as json.

    Args:
        path (str or Path): Path to JSON file.
        metadata (list of BaseAudioMeta): List of audio meta to save.
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    open_fn = gzip.open if str(path).lower().endswith(".gz") else open
    with open_fn(path, "wb") as fp:  # type: ignore
        for m in meta:
            json_str = json.dumps(m.to_dict()) + "\n"
            json_bytes = json_str.encode("utf-8")
            fp.write(json_bytes)


class AudioDataset:
    """Base audio dataset.

    The dataset takes a list of AudioMeta and create a dataset composed of segments of audio
    and potentially additional information, by creating random segments from the list of audio
    files referenced in the metadata and applying minimal data pre-processing such as resampling,
    mixing of channels, padding, etc.

    If no segment_duration value is provided, the AudioDataset will return the full wav for each
    audio file. Otherwise, it will randomly sample audio files and create a segment of the specified
    duration, applying padding if required.

    By default, only the torch Tensor corresponding to the waveform is returned. Setting return_info=True
    allows to return a tuple containing the torch Tensor and additional metadata on the segment and the
    original audio meta.

    Note that you can call `start_epoch(epoch)` in order to get
    a deterministic "randomization" for `shuffle=True`.
    For a given epoch and dataset index, this will always return the same extract.
    You can get back some diversity by setting the `shuffle_seed` param.

    Args:
        meta (list of AudioMeta): List of audio files metadata.
        segment_duration (float, optional): Optional segment duration of audio to load.
            If not specified, the dataset will load the full audio segment from the file.
        shuffle (bool): Set to `True` to have the data reshuffled at every epoch.
        sample_rate (int): Target sample rate of the loaded audio samples.
        channels (int): Target number of channels of the loaded audio samples.
        sample_on_duration (bool): Set to `True` to sample segments with probability
            dependent on audio file duration. This is only used if `segment_duration` is provided.
        sample_on_weight (bool): Set to `True` to sample segments using the `weight` entry of
            `AudioMeta`. If `sample_on_duration` is also True, the actual weight will be the product
            of the file duration and file weight. This is only used if `segment_duration` is provided.
        min_segment_ratio (float): Minimum segment ratio to use when the audio file
            is shorter than the desired segment.
        max_read_retry (int): Maximum number of retries to sample an audio segment from the dataset.
        return_info (bool): Whether to return the wav only or return wav along with segment info and metadata.
        min_audio_duration (float, optional): Minimum audio file duration, in seconds, if provided
            audio shorter than this will be filtered out.
        max_audio_duration (float, optional): Maximal audio file duration in seconds, if provided
            audio longer than this will be filtered out.
        shuffle_seed (int): can be used to further randomize
        load_wav (bool): if False, skip loading the wav but returns a tensor of 0
            with the expected segment_duration (which must be provided if load_wav is False).
        permutation_on_files (bool): only if `sample_on_weight` and `sample_on_duration`
            are False. Will ensure a permutation on files when going through the dataset.
            In that case the epoch number must be provided in order for the model
            to continue the permutation across epochs. In that case, it is assumed
            that `nitems = total_batch_size * num_updates_per_epoch`, with
            `total_batch_size` the overall batch size accounting for all gpus.
    """

    def __init__(
        self,
        meta: List[AudioMeta],
        data_type: str,
        segment_duration: Optional[float] = None,
        shuffle: bool = True,
        sample_rate: int = 48_000,
        channels: int = 2,
        pad: bool = True,
        sample_on_duration: bool = True,
        sample_on_weight: bool = True,
        min_segment_ratio: float = 0.5,
        max_read_retry: int = 10,
        return_info: bool = True,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
        shuffle_seed: int = 0,
        load_wav: bool = True,
        permutation_on_files: bool = False,
        audio_transform: Optional[Compose] = None,
    ):
        assert (
            len(meta) > 0
        ), "No audio meta provided to AudioDataset. Please check loading of audio meta."
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration

        # split meta across ranks:
        logger.info(f"Splitting files (before) {len(meta)}")
        meta = list(split_by_node(meta))
        logger.info(f"Splitting files (after) {len(meta)}")

        self.meta: List[AudioMeta] = self._filter_duration(meta)
        assert len(self.meta)  # Fail fast if all data has been filtered.
        self.total_duration = sum(d.duration for d in self.meta)

        if segment_duration is None:
            self.nitems = len(self.meta)
        else:
            # self.nitems = nitems # NOTE: I'm overriding this using total_duration
            self.nitems = int(self.total_duration / segment_duration)

        self.data_type = data_type
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.channels = channels
        self.pad = pad
        self.sample_on_weight = sample_on_weight
        self.sample_on_duration = sample_on_duration
        self.sampling_probabilities = self._get_sampling_probabilities()
        self.max_read_retry = max_read_retry
        self.return_info = return_info
        self.shuffle_seed = shuffle_seed
        self.current_epoch: Optional[int] = None
        self.load_wav = load_wav
        self.urls = [m.path for m in self.meta]
        if not load_wav:
            assert segment_duration is not None
        self.permutation_on_files = permutation_on_files
        if permutation_on_files:
            assert not self.sample_on_duration
            assert not self.sample_on_weight
            assert self.shuffle
        self.audio_transform = audio_transform

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self):
        return self.nitems

    def _get_sampling_probabilities(self, normalized: bool = True):
        """Return the sampling probabilities for each file inside `self.meta`."""
        scores: List[float] = []
        for file_meta in self.meta:
            score = 1.0
            if self.sample_on_weight and file_meta.weight is not None:
                score *= file_meta.weight
            if self.sample_on_duration:
                score *= file_meta.duration
            scores.append(score)
        probabilities = torch.tensor(scores)
        if normalized:
            probabilities /= probabilities.sum()
        return probabilities

    @staticmethod
    @lru_cache(16)
    def _get_file_permutation(num_files: int, permutation_index: int, base_seed: int):
        # Used to keep the most recent files permutation in memory implicitely.
        # will work unless someone is using a lot of Datasets in parallel.
        rng = torch.Generator()
        rng.manual_seed(base_seed + permutation_index)
        return torch.randperm(num_files, generator=rng)

    def sample_file(self, index: int, rng: torch.Generator) -> AudioMeta:
        """Sample a given file from `self.meta`. Can be overridden in subclasses.
        This is only called if `segment_duration` is not None.

        You must use the provided random number generator `rng` for reproducibility.
        You can further make use of the index accessed.
        """
        if self.permutation_on_files:
            assert self.current_epoch is not None
            total_index = self.current_epoch * len(self) + index
            permutation_index = total_index // len(self.meta)
            relative_index = total_index % len(self.meta)
            permutation = AudioDataset._get_file_permutation(
                len(self.meta), permutation_index, self.shuffle_seed
            )
            file_index = permutation[relative_index]
            return self.meta[file_index]

        if not self.sample_on_weight and not self.sample_on_duration:
            file_index = int(
                torch.randint(
                    len(self.sampling_probabilities), (1,), generator=rng
                ).item()
            )
        else:
            file_index = int(
                torch.multinomial(self.sampling_probabilities, 1, generator=rng).item()
            )

        return self.meta[file_index]

    def _audio_read(self, path: str, seek_time: float = 0, duration: float = -1):
        # Override this method in subclass if needed.
        if self.load_wav:
            return audio_read(path, seek_time, duration, pad=False)
        else:
            assert self.segment_duration is not None
            n_frames = int(self.sample_rate * self.segment_duration)
            return torch.zeros(self.channels, n_frames), self.sample_rate

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, SegmentInfo]]:
        if self.segment_duration is None:
            file_meta = self.meta[index]
            out, sr = audio_read(file_meta.path)
            out = convert_audio(out, sr, self.sample_rate, self.channels)
            n_frames = out.shape[-1]
            segment_info = SegmentInfo(
                file_meta,
                seek_time=0.0,
                n_frames=n_frames,
                total_frames=n_frames,
                sample_rate=self.sample_rate,
                channels=out.shape[0],
                data_type=self.data_type,
            )
        else:
            rng = torch.Generator()
            if self.shuffle:
                # We use index, plus extra randomness, either totally random if we don't know the epoch.
                # otherwise we make use of the epoch number and optional shuffle_seed.
                if self.current_epoch is None:
                    rng.manual_seed(index + self.nitems * random.randint(0, 2**24))
                else:
                    rng.manual_seed(
                        index + self.nitems * (self.current_epoch + self.shuffle_seed)
                    )
            else:
                # We only use index
                rng.manual_seed(index)

            for retry in range(self.max_read_retry):
                file_meta = self.sample_file(index, rng)
                # We add some variance in the file position even if audio file is smaller than segment
                # without ending up with empty segments
                max_seek = max(
                    0,
                    file_meta.duration - self.segment_duration * self.min_segment_ratio,
                )
                seek_time = torch.rand(1, generator=rng).item() * max_seek
                try:
                    out, sr = audio_read(
                        file_meta.path, seek_time, self.segment_duration, pad=False
                    )
                    out = convert_audio(out, sr, self.sample_rate, self.channels)
                    n_frames = out.shape[-1]
                    target_frames = int(self.segment_duration * self.sample_rate)
                    if self.pad:
                        out = F.pad(out, (0, target_frames - n_frames))
                    segment_info = SegmentInfo(
                        file_meta,
                        seek_time,
                        n_frames=n_frames,
                        total_frames=target_frames,
                        sample_rate=self.sample_rate,
                        channels=out.shape[0],
                        data_type=self.data_type,
                    )
                except Exception as exc:
                    logger.warning("Error opening file %s: %r", file_meta.path, exc)
                    if retry == self.max_read_retry - 1:
                        raise
                else:
                    break

        if self.audio_transform is not None:
            out = self.audio_transform(out)

        shard_info = ShardInfo(url=file_meta.path, uttid=file_meta.path)
        return AudioDataResult(
            audio=out,
            segment_info=segment_info,
            index=[],
            key=index,
            shard=None,
            shard_info=shard_info,
        )

    def collater(self, samples):
        """The collater function has to be provided to the dataloader
        if AudioDataset has return_info=True in order to properly collate
        the samples of a batch.
        """
        if self.segment_duration is None and len(samples) > 1:
            assert (
                self.pad
            ), "Must allow padding when batching examples of different durations."

        # In this case the audio reaching the collater is of variable length as segment_duration=None.
        to_pad = self.segment_duration is None and self.pad
        if to_pad:
            max_len = max([wav.shape[-1] for wav, _ in samples])

            def _pad_wav(wav):
                return F.pad(wav, (0, max_len - wav.shape[-1]))

        if self.return_info:
            if len(samples) > 0:
                assert len(samples[0]) == 2
                assert isinstance(samples[0][0], torch.Tensor)
                assert isinstance(samples[0][1], SegmentInfo)

            wavs = [wav for wav, _ in samples]
            segment_infos = [copy.deepcopy(info) for _, info in samples]

            if to_pad:
                # Each wav could be of a different duration as they are not segmented.
                for i in range(len(samples)):
                    # Determines the total length of the signal with padding, so we update here as we pad.
                    segment_infos[i].total_frames = max_len
                    wavs[i] = _pad_wav(wavs[i])

            wav = torch.stack(wavs)
            return wav, segment_infos
        else:
            assert isinstance(samples[0], torch.Tensor)
            if to_pad:
                samples = [_pad_wav(s) for s in samples]
            return torch.stack(samples)

    def _filter_duration(self, meta: List[AudioMeta]) -> List[AudioMeta]:
        """Filters out audio files with audio durations that will not allow to sample examples from them."""
        orig_len = len(meta)

        # Filter data that is too short.
        if self.min_audio_duration is not None:
            meta = [m for m in meta if m.duration >= self.min_audio_duration]

        # Filter data that is too long.
        if self.max_audio_duration is not None:
            meta = [m for m in meta if m.duration <= self.max_audio_duration]

        filtered_len = len(meta)
        removed_percentage = 100 * (1 - float(filtered_len) / orig_len)
        msg = (
            "Removed %.2f percent of the data because it was too short or too long."
            % removed_percentage
        )
        if removed_percentage < 10:
            logging.debug(msg)
        else:
            logging.warning(msg)
        return meta

    @classmethod
    def from_meta(cls, root: Union[str, Path], **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_dir():
            if (root / "data.jsonl").exists():
                root = root / "data.jsonl"
            elif (root / "data.jsonl.gz").exists():
                root = root / "data.jsonl.gz"
            else:
                raise ValueError(
                    "Don't know where to read metadata from in the dir. "
                    "Expecting either a data.jsonl or data.jsonl.gz file but none found."
                )
        meta = load_audio_meta(root)
        return cls(meta, **kwargs)

    @classmethod
    def from_path(
        cls,
        root: Union[str, Path],
        minimal_meta: bool = True,
        exts: List[str] = DEFAULT_EXTS,
        **kwargs,
    ):
        """Instantiate AudioDataset from a path containing (possibly nested) audio files.

        Args:
            root (str or Path): Path to root folder containing audio files.
            minimal_meta (bool): Whether to only load minimal metadata or not.
            exts (list of str): Extensions for audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_file():
            meta = load_audio_meta(root)
        else:
            meta = find_audio_files(root, exts, minimal=minimal_meta)
        return cls(meta, **kwargs)


class AudioFolderDataset(AudioDataset):
    def __init__(
        self,
        root: str,
        data_type: str,
        segment_duration: float,
        sample_rate: int,
        channels: int,
        shuffle: bool,
        num_workers: int,
        pad: bool = False,
        n_samples: Optional[int] = None,
        min_segment_ratio: float = 1.0,
    ):
        self.root = root
        meta = find_audio_files(
            root,
            DEFAULT_EXTS,
            progress=True,
            minimal=True,
            workers=num_workers,
            n_samples=n_samples,
        )
        self.audio_transform = Compose(
            [
                # RandomApply([Silence()], p=0.05),
                RescaleAudio()
            ]
        )
        super().__init__(
            meta,
            data_type=data_type,
            segment_duration=segment_duration,
            shuffle=shuffle,
            sample_rate=sample_rate,
            channels=channels,
            pad=pad,
            sample_on_duration=True,
            sample_on_weight=True,
            audio_transform=self.audio_transform,
            min_segment_ratio=min_segment_ratio,
        )

    def __repr__(self):
        return f"{self.__class__.__name__} - root: {self.root} - nitems: {len(self)}"


def merge_datasets(datasets, weights: Optional[List[float]] = None):
    if len(datasets) == 1:
        return datasets[0]
    assert len(datasets) == len(weights)
    np.testing.assert_almost_equal(sum(weights), 1, decimal=5)
    return MultiIterableDataset(
        datasets, weights=weights, num_samples=10_000_000
    )  # TODO: rm num_samples!


class AudioFolderDataModule(LightningDataModuleBase):
    def __init__(
        self,
        train_datasets: List[AudioFolderDataset],
        validation_datasets: List[AudioFolderDataset],
        test_datasets: List[AudioFolderDataset],
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        weights: Optional[List[float]] = None,
        pin_memory: bool = True,
        validation_batch_size: Optional[int] = None,
        prefetch_factor: int = 2,
        batch_drop_duplicates: bool = True,
    ):
        self.weights = weights
        logger.info("Training datasets:")
        logger.info(train_datasets)

        all_nitems = [t.nitems for t in train_datasets]
        self._nitems = sum(all_nitems)
        if len(train_datasets) > 1 and weights is None:
            logger.info("Inferring weights from dataset sizes:")

            self.weights = [n / self._nitems for n in all_nitems]

            for t, n, w in zip(train_datasets, all_nitems, self.weights):
                logger.info(f"{t.__class__.__name__}: {n} ({w * 100}%)")

        train_dataset = merge_datasets(train_datasets, self.weights)

        if len(validation_datasets):
            validation_dataset = merge_datasets(validation_datasets)
        else:
            validation_dataset = []

        if len(test_datasets):
            test_dataset = merge_datasets(test_datasets)
        else:
            test_dataset = []

        self.batch_drop_duplicates = batch_drop_duplicates

        super().__init__(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            validation_batch_size=validation_batch_size,
            prefetch_factor=prefetch_factor,
        )

        logger.info(f"Total samples: {self.nitems}")

    @property
    def nitems(self) -> int:
        return self._nitems

    def collate_fn(self, batch) -> AudioDataResult:
        batch_keys = set()

        collated = AudioDataResult(
            audio=[], segment_info=[], index=[], shard=[], key=[], shard_info=[]
        )

        input_length = []
        for res in batch:

            # if the key exists in the batch, add it to the queue and refill it
            if self.batch_drop_duplicates and res.key in batch_keys:
                continue

            batch_keys.add(res.key)
            collated.audio.append(res.audio)
            collated.segment_info.append(res.segment_info)
            collated.index.append(res.index)
            collated.shard.append(res.shard)
            collated.key.append(res.key)
            collated.shard_info.append(res.shard_info)

            if res.audio is not None:
                input_length.append(res.audio.shape[-1])

        if len(input_length):
            max_length = max(input_length)
            pad = Pad(max_length, value=0.0)
            for i in range(len(collated.audio)):
                audio = collated.audio[i]
                collated.audio[i] = pad(audio)

            collated.audio = torch.stack(collated.audio, dim=0)
        return collated


class FeatureDataModule(LightningDataModuleBase):
    def __init__(
        self,
        train_datasets: List[FeatureWebDataset],
        validation_datasets: List[FeatureWebDataset],
        test_datasets: List[FeatureWebDataset],
        weights: List[float],
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        validation_batch_size: Optional[int] = None,
        prefetch_factor: int = 2,
        batch_drop_duplicates: bool = True,
    ):
        logger.info("Training datasets:")
        logger.info(train_datasets)

        train_dataset = merge_datasets(train_datasets, weights)

        if len(validation_datasets):
            validation_dataset = merge_datasets(validation_datasets)
        else:
            validation_dataset = None

        if len(test_datasets):
            test_dataset = merge_datasets(test_datasets)
        else:
            test_dataset = None

        self.batch_drop_duplicates = batch_drop_duplicates

        super().__init__(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            validation_batch_size=validation_batch_size,
            prefetch_factor=prefetch_factor,
        )

    def collate_fn(self, batch) -> AudioDataResult:
        batch_keys = set()

        collated = AudioDataResult(
            audio=[], segment_info=[], index=[], shard=[], key=[]
        )

        input_length = []
        for res in batch:

            # if the key exists in the batch, add it to the queue and refill it
            if self.batch_drop_duplicates and res.key in batch_keys:
                continue

            batch_keys.add(res.key)
            input_length.append(res.audio.shape[-1])
            collated.audio.append(res.audio)
            collated.segment_info.append(res.segment_info)
            collated.index.append(res.index)
            collated.shard.append(res.shard)
            collated.key.append(res.key)

        collated.audio = torch.stack(collated.audio, dim=0)
        return collated


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="audio_dataset", description="Generate .jsonl files by scanning a folder."
    )
    parser.add_argument("root", help="Root folder with all the audio files")
    parser.add_argument("output_meta_file", help="Output file to store the metadata, ")
    parser.add_argument(
        "--complete",
        action="store_false",
        dest="minimal",
        default=True,
        help="Retrieve all metadata, even the one that are expensive "
        "to compute (e.g. normalization).",
    )
    parser.add_argument("--workers", default=10, type=int, help="Number of workers.")
    args = parser.parse_args()
    meta = find_audio_files(
        args.root,
        DEFAULT_EXTS,
        progress=True,
        minimal=args.minimal,
        workers=args.workers,
    )
    save_audio_meta(args.output_meta_file, meta)


if __name__ == "__main__":
    main()
