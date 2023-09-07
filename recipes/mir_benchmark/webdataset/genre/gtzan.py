"""
GTZAN Genre Classification Dataset.
"""

import os
import random

import librosa
import numpy as np
import tqdm

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
# warnings.filterwarnings("ignore")

# download paths
_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/" "gtzan/audio_original",
    "genre_labels": "hdfs://harunava/home/byte_speech_sv/data/" "gtzan/splits",
}


class GTZANDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    GTZAN Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "gtzan_genre_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        self.audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])

        # get labels
        self.label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["genre_labels"])

        return self.label_dir

    def _split_data(self, data_dir):
        with open(os.path.join(self.label_dir, "train.txt")) as f:
            lines = f.readlines()
        train = [line.strip().replace(".", "$$") for line in lines]
        with open(os.path.join(self.label_dir, "validation.txt")) as f:
            lines = f.readlines()
        validation = [line.strip().replace(".", "$$") for line in lines]
        with open(os.path.join(self.label_dir, "test.txt")) as f:
            lines = f.readlines()
        test = [line.strip().replace(".", "$$") for line in lines]
        return {
            "train": {"keys": train},
            "validation": {"keys": validation},
            "test": {"keys": test},
        }

    def _load_example(self, key):
        # read annotations
        genre = key.split("/")[0]
        audio_path = os.path.join(self.audio_dir, key.replace("$$", "."))
        np_audio, _ = librosa.load(audio_path, sr=GTZANDataset.SAMPLING_RATE)
        return {
            "dataset.txt": GTZANDataset.DATASET_NAME,
            "__key__": key,
            "genre.txt": genre,
            "audio.npy": np_audio.astype(GTZANDataset.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in tqdm.tqdm(keys):
            yield self._load_example(key)
