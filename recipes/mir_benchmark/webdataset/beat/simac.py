"""
Simac Beat Tracking Dataset.
"""

import os
import random
import warnings

import librosa
import numpy as np
import tqdm

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
warnings.filterwarnings("ignore")

# download paths
_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/raw_data/simac/simac_audio.zip",
    "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/simac_beats.zip",
}


class SimacDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Simac Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "simac_beat_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.audio_dir = self._dl_manager.extract(audio_dir)

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["beat_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "simac_beats")):
            k = k.split(".txt")[0]
            audio_fn = os.path.join(self.audio_dir, "audio", k + ".flac")
            if k != "" and os.path.exists(audio_fn):
                keys.append(k.replace(".", "$$"))
        return {"train": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(
            os.path.join(self.label_dir, "simac_beats", key.replace("$$", ".") + ".txt")
        ) as f:
            beats = f.readlines()
            beats = [[float(line.rstrip()), 0] for line in beats]
        audio_path = os.path.join(
            self.audio_dir, "audio", key.replace("$$", ".") + ".flac"
        )
        np_audio, _ = librosa.load(audio_path, sr=SimacDataset.SAMPLING_RATE)
        return {
            "dataset.txt": SimacDataset.DATASET_NAME,
            "__key__": key,
            "beats.pickle": beats,
            "audio.npy": np_audio.astype(SimacDataset.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in tqdm.tqdm(keys):
            yield self._load_example(key)
