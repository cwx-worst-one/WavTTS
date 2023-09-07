"""
Internal Clip500 Beat Tracking Dataset.
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
    "SheetDoctor/raw_data/bytedance_clip/audio500",
    "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/clip500_beats.zip",
}


class Clip500Dataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Clip500 Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "clip500_beat_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        self.audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["beat_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "clip500_beats")):
            k = k.split(".txt")[0]
            audio_fn = os.path.join(self.audio_dir, k + ".mp3")
            if k != "" and os.path.exists(audio_fn):
                keys.append(k.replace(".", "$$"))
        return {"test": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(
            os.path.join(
                self.label_dir, "clip500_beats", key.replace("$$", ".") + ".txt"
            )
        ) as f:
            beats = f.readlines()
            beats = [line.rstrip().split() for line in beats]
            beats = [[float(x[0]), int(x[1])] for x in beats]
        audio_path = os.path.join(self.audio_dir, key.replace("$$", ".") + ".mp3")
        np_audio, _ = librosa.load(audio_path, sr=Clip500Dataset.SAMPLING_RATE)
        return {
            "dataset.txt": Clip500Dataset.DATASET_NAME,
            "__key__": key,
            "beats.pickle": beats,
            "audio.npy": np_audio.astype(Clip500Dataset.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in tqdm.tqdm(keys):
            yield self._load_example(key)
