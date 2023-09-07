"""
Resso Genre 34 dataset
It's in 24kHz which is upsampled from 22.05kHz sample rate
"""

import os
import random
import sys
import warnings

import librosa
import numpy as np
import tqdm

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
warnings.filterwarnings("ignore")


class RessoGenre34Dataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Genre34 Dataset
    """

    SAMPLING_RATE = 24000
    # DATASET_NAME = "genre34_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self, current_shard, shard_size=0, split="train") -> None:
        super().__init__()
        self.split = split
        self.shard_size = shard_size
        self.current_shard = current_shard
        self.audio_dir = "/mnt/bn/transcription/genre34/audio"
        self.dataset_name = "genre34_24kHz_%d" % current_shard

    @property
    def DATASET_NAME(self):
        return self.dataset_name

    def _split_data(self, data_dir):
        keys = np.load(
            "/mnt/bn/transcription/resso_cls/genre34/data/sampled_%s_keys.npy"
            % self.split[:5]
        )
        tag_binary = np.load(
            "/mnt/bn/transcription/resso_cls/genre34/data/sampled_%s_labels.npy"
            % self.split[:5]
        )

        # take only unique and shuffle
        random.seed(42)
        _, indices = np.unique(keys, return_index=True)
        if self.split == "train":
            random.shuffle(indices)
            shuffled_indices = indices[
                self.current_shard
                * self.shard_size : (self.current_shard + 1)
                * self.shard_size
            ]
        else:
            shuffled_indices = indices

        keys = keys[shuffled_indices]
        tag_binary = tag_binary[shuffled_indices]
        print(self.current_shard, self.shard_size, len(keys), len(tag_binary))
        return {self.split: {"keys": [keys, tag_binary, self.split]}}

    def _load_example(self, key, binary, split):
        sub_path = "/".join([k for k in key.split("-")[0][-3:]])
        audio_path = os.path.join(self.audio_dir, sub_path, key + ".wav")
        np_audio, _ = librosa.load(audio_path, sr=RessoGenre34Dataset.SAMPLING_RATE)
        np_audio = np_audio[: 24000 * 30]
        return {
            "dataset.txt": RessoGenre34Dataset.DATASET_NAME,
            "__key__": key,
            "split": split,
            "tag_binary.npy": binary,
            "audio.npy": np_audio.astype(RessoGenre34Dataset.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        keys, binaries, split = keys
        # generate audio beat pairs then iterate over them
        for key, binary in tqdm.tqdm(zip(keys, binaries)):
            yield self._load_example(key, binary, split)


if __name__ == "__main__":
    current_shard = int(sys.argv[1])
    shard_size = int(sys.argv[2])
    split = sys.argv[3]
    ds = RessoGenre34Dataset(current_shard, shard_size, split)
    ds.create_dataset()
