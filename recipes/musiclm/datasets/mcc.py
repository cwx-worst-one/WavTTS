from typing import Callable, List, Optional, Union
import torch
import os
from functools import partial
import librosa
import numpy as np
from pydub import AudioSegment
import webdataset as wds
from torch.utils.data import Dataset
from recipes.musiclm.transforms.musiclm import MCCTransforms, PGCTransforms, ARFiltering
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.utils.datastructures import select_keys
from recipes.musiclm.transforms.audio import LoudnessCheck

from samantha.dataio.dataset import MultiIterableDataset


def get_active_frames(audio, threshold=0.05, sample_rate=24000):
    window_size = int(sample_rate * 0.1)

    frames = librosa.util.frame(
        x=audio, frame_length=window_size, hop_length=window_size
    ).T
    energy = np.max(np.abs(frames), axis=-1)  # shape: (frames_num,)
    rate = np.sum(energy > threshold) / energy.shape[0]

    if rate < 1 / 10:
        return False
    return True


class MCC7MDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        sample_rate=24000,
        sample_duration=10,
        use_cache=False,
        cache_path=None,
        min_volume_threshold=0.05,
        loudness_ratio_threshold=0.1,
    ):
        self.sample_rate = sample_rate
        self.durations = sample_duration
        self.use_cache = use_cache
        self.segment_size = sample_duration * self.sample_rate
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.audio_paths = []

        for meta_path in meta_paths:
            self.audio_paths += self.get_meta_data(meta_path)
        if use_cache:
            with open(cache_path, "r") as f:
                lines = [line.strip() for line in f]
            self.cache_dict = {"w2v": {}, "mulan": {}}
            for line in lines:
                line = line.split("|")
                self.cache_dict["w2v"].update({line[0]: line[1]})
        self.is_loud = LoudnessCheck(
            self.sample_rate,
            self.min_volume_threshold,
            self.loudness_ratio_threshold
        )

    def get_meta_data(self, meta_path):
        with open(meta_path, "r") as f:
            lines = [line.strip() for line in f]
        paths = []
        for line in lines:
            l_res = line.split("|")
            path, length = l_res[0:2]
            length = float(length)
            if (length >= self.segment_size - 0.05 * self.sample_rate) and (
                length <= self.sample_rate * 600
            ):
                paths.append(path)
        paths = [
            os.path.abspath(os.path.join(os.path.dirname(meta_path, p)))
            if p[0] != "/"
            else p
            for p in paths
        ]
        return paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        while True:
            """
            file_size = os.path.getsize(path)
            if file_size / 1024 / 1024 > 60: # skip > 60M audio
                path = np.random.choice(self.audio_paths)
                continue
            """
            try:
                if path.endswith(".npy"):
                    wav = np.load(path)
                else:
                    audio = AudioSegment.from_file(path)
                    audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
                    wav = np.asarray(audio.get_array_of_samples())
                if wav.dtype == np.int16:
                    wav = wav / 32768.0
                elif wav.dtype == np.int32:
                    wav = wav / 2_147_483_648.0
                if len(wav.shape) >= 2:
                    wav = wav[0]
                scale = np.max(np.abs(wav))
                wav = wav.astype(np.float32)
                # random slice
                wav_len = wav.shape[0]
                # prevent data not long enough
                if wav_len < self.segment_size - 0.05 * self.sample_rate:
                    raise Exception
                if wav_len < self.segment_size:
                    wav = np.pad(wav, (0, self.segment_size - wav_len))
                    rand_slice = wav
                else:
                    beg = np.random.randint(low=0, high=wav_len - self.segment_size + 1)
                    rand_slice = wav[beg : beg + self.segment_size]
                # prevent silence
                if np.sqrt(np.mean(rand_slice**2)) > 1e-2 and self.is_loud(torch.from_numpy(rand_slice[None, :])):
                    rand_slice = rand_slice / scale * 0.95
                    break
                else:
                    path = np.random.choice(self.audio_paths)
            except Exception:
                print("File {} cant load, choose another one".format(path))
                path = np.random.choice(self.audio_paths)
        return rand_slice


class MCC40MDataset(WebPipeline):

    def __init__(
        self,
        url2index: Union[str, int],
        sample_rate: int,
        duration: Union[float, List[float]],
        audio_key: str = "mp3",
        min_length_ratio: float = 0.8,
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = True,
        sstk_filtered: Optional[str] = None,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        avoid_vocal: bool = True,
        max_vocal_threshold: float = 0.5,
        overlap_vocal_threshold: float = 0.1,
        audio_metrics_filtered: bool = True,
        ar_filtering: Optional[ARFiltering] = None,
        text_type: Optional[str] = None,
        max_num_crops: Optional[Union[int, List[int]]] = None,
        crop_step_size: Optional[Union[int, List[int]]] = None,
        max_duration: Optional[int] = None,
        additional_transforms: Optional[List] = None,
        handler: Callable = wds.warn_and_continue,
        **kwargs,
    ):
        is_parquet = isinstance(url2index, int)
        if is_parquet:
            dataset = ParquetDataset(data_id=url2index, **kwargs)
        else:
            dataset = IndexedWebDataset(url2index=url2index, handler=handler, **kwargs)

        if isinstance(duration, (list, tuple)):
            n_samples = [int(d * sample_rate) for d in duration]
        else:
            n_samples = int(duration * sample_rate)

        audio_transforms = MCCTransforms(
            n_samples=n_samples,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_length_ratio=min_length_ratio,
            normalize_audio=normalize_audio,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            aed_filtered=aed_filtered,
            sstk_filtered=sstk_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            overlap_vocal_threshold=overlap_vocal_threshold,
            audio_metrics_filtered=audio_metrics_filtered,
            ar_filtering=ar_filtering,
            text_type=text_type,
            max_num_crops=max_num_crops,
            crop_step_size=crop_step_size,
            max_samples=None if max_duration is None else max_duration * sample_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        additional_transforms = [wds.map(t) for t in additional_transforms] if additional_transforms else []
        pipeline = []
        if not is_parquet:
            pipeline.append("decode")
        pipeline.append(
            {"compose": [preprocessor.train_buffer_preprocessor, *additional_transforms]}
        )
        super().__init__(dataset, pipeline)


class WrappedMCC40MDataset(MultiIterableDataset):

    def __init__(
        self,
        url2index_list: list,
        sample_rate: int,
        duration: Union[float, List[float]],
        audio_key: str = "mp3",
        min_length_ratio: float = 0.8,
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = True,
        sstk_filtered: Optional[str] = None,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        avoid_vocal: bool = True,
        max_vocal_threshold: float = 0.5,
        overlap_vocal_threshold: float = 0.1,
        audio_metrics_filtered: bool = True,
        ar_filtering: Optional[ARFiltering] = None,
        text_type: Optional[str] = None,
        max_num_crops: Optional[Union[int, List[int]]] = None,
        crop_step_size: Optional[Union[int, List[int]]] = None,
        additional_transforms: Optional[List] = None,
        handler: Callable = wds.warn_and_continue,
        num_samples: int = -1,
        seed: int = 2023,
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        datasets = [
            MCC40MDataset(
                url2index=url2index,
                sample_rate=sample_rate,
                duration=duration,
                audio_key=audio_key,
                min_length_ratio=min_length_ratio,
                normalize_audio=normalize_audio,
                min_volume_threshold=min_volume_threshold,
                loudness_ratio_threshold=loudness_ratio_threshold,
                aed_filtered=aed_filtered,
                sstk_filtered=sstk_filtered,
                avoid_sound_effect=avoid_sound_effect,
                exclude_licenses=exclude_licenses,
                avoid_vocal=avoid_vocal,
                max_vocal_threshold=max_vocal_threshold,
                overlap_vocal_threshold=overlap_vocal_threshold,
                audio_metrics_filtered=audio_metrics_filtered,
                ar_filtering=ar_filtering,
                text_type=text_type,
                max_num_crops=max_num_crops,
                crop_step_size=crop_step_size,
                additional_transforms=additional_transforms,
                handler=handler,
                **kwargs,
            ) for url2index in url2index_list
        ]

        if weights is None:
            weights=[1.0 for _ in range(len(datasets))]
        assert len(weights) == len(datasets)
        super().__init__(
            datasets=datasets,
            num_samples=num_samples,
            weights=weights,
            seed=seed,
        )


class PGCDataset(WebPipeline):

    def __init__(
        self,
        duration: float,
        sample_rate: int = 24000,
        url2index: str = "/mnt/bn/audio-diffusion/data/mcc_pgc_600k/pgc_url2idx.txt",
        audio_key: str = "audio.npy",
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        max_num_crops: Optional[int] = None,
        crop_step_size: Optional[int] = None,
        handler: Callable = wds.warn_and_continue,
        use_pipe: bool = True,
        **kwargs,
    ):
        dataset = IndexedWebDataset(
            url2index=url2index,
            use_pipe=use_pipe,
            handler=handler,
            **kwargs,
        )

        audio_transforms = PGCTransforms(
            n_samples=int(duration * sample_rate),
            sample_rate=sample_rate,
            audio_key=audio_key,
            normalize_audio=normalize_audio,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            max_num_crops=max_num_crops,
            crop_step_size=crop_step_size,
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
        ]
        super().__init__(dataset, pipeline)
