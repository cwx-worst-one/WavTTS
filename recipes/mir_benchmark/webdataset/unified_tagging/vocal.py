"""
Vocal tagging dataset
"""

import json
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


class VocalDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Vocal Tagging Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "vocal_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()
        self.audio_dir = "/mnt/bn/transcription/unified_tagging/vocal/audio"

    def _split_data(self, data_dir):
        self.train_json = json.load(
            open("/mnt/bn/transcription/unified_tagging/vocal/train.json")
        )
        self.valid_json = json.load(
            open("/mnt/bn/transcription/unified_tagging/vocal/valid.json")
        )

        train_keys = list(self.train_json.keys())
        valid_keys = list(self.valid_json.keys())
        random.seed(42)
        random.shuffle(train_keys)

        return {
            "train": {"keys": [train_keys, "train"]},
            "validation": {"keys": [valid_keys, "validation"]},
        }

    def _load_example(self, key, split):
        # load audio
        audio_path = os.path.join(self.audio_dir, key + ".wav")
        np_audio, _ = librosa.load(audio_path, sr=VocalDataset.SAMPLING_RATE)

        # get labels
        if split == "train":
            data = self.train_json[key]
        elif split == "validation":
            data = self.valid_json[key]

        return {
            "dataset.txt": VocalDataset.DATASET_NAME,
            "__key__": key,
            "split": split,
            "audio.npy": np_audio.astype(VocalDataset.AUDIO_NUMPY_DTYPE),
            "gender": data["gender"],
            "age": data["age"],
            "style": data["style"],
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        keys, split = keys

        # generate audio beat pairs then iterate over them
        for key in tqdm.tqdm(keys):
            yield self._load_example(key, split)


if __name__ == "__main__":
    ds = VocalDataset()
    ds.create_dataset()
