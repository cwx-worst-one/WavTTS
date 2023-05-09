import os

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from torch.utils.data import Dataset


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


class AudioLMDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        sample_rate=24000,
        sample_duration=10,
        use_cache=False,
        cache_path=None,
    ):
        self.sample_rate = sample_rate
        self.durations = sample_duration
        self.use_cache = use_cache
        self.segment_size = sample_duration * self.sample_rate
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
                if np.sqrt(np.mean(rand_slice**2)) > 1e-2 and get_active_frames(
                    rand_slice, threshold=0.05, sample_rate=self.sample_rate
                ):
                    rand_slice = rand_slice / scale * 0.95
                    break
                else:
                    path = np.random.choice(self.audio_paths)
            except Exception:
                print("File {} cant load, choose another one".format(path))
                path = np.random.choice(self.audio_paths)
        return rand_slice


class AudioPathDataset(Dataset):
    def __init__(self, dir, cache_dir=None, filter=["wav", "mp3", "mp4", "m4a", "npy"]):
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        local_path = (
            f"{cache_dir}/{os.path.basename(dir)}"
            if cache_dir
            else os.path.basename(dir)
        )

        from ..utils.hdfs_tools import hdfs_cp

        if dir != local_path:
            hdfs_cp(dir, local_path, True)

        self.filter = filter
        self.list = [
            os.path.join(root, f)
            for root, _, files in os.walk(local_path)
            for f in files
            if self.is_audio(f)
        ]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

    def is_audio(self, fname):
        fname = fname.split(".")[-1]
        if fname in self.filter:
            return True
        return False


class InferenceDataset(Dataset):
    def __init__(self, csv_path):
        self.items = []
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.items.append({"category": row["category"], "text": row["text"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
