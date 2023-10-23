"""
pop909 Chord Recognition Dataset.
"""

import os
import random

import h5py
import numpy as np

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/audio_h5/pop909_audio.h5",
    "chord_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/pop909_labels.zip",
}


class Pop909Dataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Pop909 Dataset
    """

    SAMPLING_RATE = 16000
    DATASET_NAME = "pop909_chord"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        h5_file = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.h5 = h5py.File(h5_file, "r")

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["chord_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "pop909_labels")):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                keys.append(k)
        return {"train": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(os.path.join(self.label_dir, "pop909_labels", key + ".txt")) as f:
            chords = f.read().splitlines()
            intervals = []
            symbols = []

            for i, line in enumerate(chords):
                s = line.split("\t")
                start = float(s[0])
                end = float(s[1])
                symbol = s[2]
                intervals.append([start, end])
                symbols.append(symbol)

        if len(intervals) != 0:
            np_audio = self.h5[key][:]
            return {
                "dataset.txt": self.DATASET_NAME,
                "__key__": key,
                "intervals.pickle": intervals,
                "chords.pickle": symbols,
                "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
            }
        else:
            return None

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in keys:
            data = self._load_example(key)
            if data is not None:
                yield data
