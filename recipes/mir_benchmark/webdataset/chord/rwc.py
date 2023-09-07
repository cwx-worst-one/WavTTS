"""
RWC Chord Recognition Dataset.
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
    "SheetDoctor/raw_data/rwc/audio",
    "chord_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/rwc_truth_chords.zip",
}


class RWCDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    RWC Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "rwc_chord_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        self.audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["chord_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "rwc_truth_chords")):
            if k[0] != ".":
                k = k.split(".txt")[0]
                audio_fn = os.path.join(self.audio_dir, k + ".mp3")
                if k != "" and os.path.exists(audio_fn):
                    keys.append(k.replace(".", "$$"))
        return {"train": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(
            os.path.join(
                self.label_dir, "rwc_truth_chords", key.replace("$$", ".") + ".txt"
            )
        ) as f:
            chords = f.readlines()
            intervals = []
            symbols = []
            for line in chords:
                s = line.strip().split("\t")
                if len(s) != 3:
                    s = line.strip().split(" ")
                if line.strip() != "":
                    start = float(s[0])
                    end = float(s[1])
                    symbol = s[2]
                    intervals.append([start, end])
                    symbols.append(symbol)
        audio_path = os.path.join(self.audio_dir, key.replace("$$", ".") + ".mp3")
        np_audio, _ = librosa.load(audio_path, sr=RWCDataset.SAMPLING_RATE)
        return {
            "dataset.txt": RWCDataset.DATASET_NAME,
            "__key__": key,
            "intervals.pickle": intervals,
            "chords.pickle": symbols,
            "audio.npy": np_audio.astype(RWCDataset.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio chord pairs then iterate over them
        for key in tqdm.tqdm(keys):
            yield self._load_example(key)
