from typing import List, Optional
import os
from functools import partial
import librosa
import numpy as np
from pydub import AudioSegment
from torch.utils.data import Dataset
from recipes.musiclm.transforms.musiclm import MCCTransforms
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.utils.datastructures import select_keys
from recipes.musiclm.transforms.audio import LoudnessCheck


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
                if np.sqrt(np.mean(rand_slice**2)) > 1e-2 and self.is_loud(rand_slice):
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
        url2index: str,
        sample_rate: int,
        duration: float,
        batch_size: int,
        shuffle_buffer_size: int,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.5,
        aed_filtered: bool = True,
        avoid_sound_effect: bool = True,
        avoid_vocal: bool = True,
        max_vocal_threshold: float = 0.25,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: Optional[int] = 3,   # recommended for 30s crops
        **kwargs,
    ):
        dataset = IndexedWebDataset(
            url2index=url2index,
            **kwargs,
        )

        audio_transforms = MCCTransforms(
            n_samples=int(duration * sample_rate),
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            aed_filtered=aed_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            max_num_crops=max_num_crops,
            crop_step_size=int(duration * sample_rate / 5),
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        pipeline=[
            {"select": {"predicate": partial(select_keys, keys=[audio_key])}},
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
            {"shuffle": [shuffle_buffer_size]},
            {"to_tuple": ["audio.npy"]},
            {"batched": [batch_size]}
        ]
        super().__init__(dataset, pipeline)
