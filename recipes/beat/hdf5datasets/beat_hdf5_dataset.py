import os
import random

import h5py
import numpy as np

from samantha.dataio.hdf5 import AbstractHdf5GeneratorBasedBuilder


class BeatHDF5Dataset(AbstractHdf5GeneratorBasedBuilder):

    SAMPLING_RATE = 16000
    AUDIO_NUMPY_DTYPE = np.float32
    DATASET_NAME = "rwc_beat"
    LABEL_DIR = "rwc_truth_beats"
    _HDFS_DIRS = {
        "audio": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/audio_h5/rwc_audio.h5",
        "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/labels/rwc_truth_beats.zip",
    }

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        if "audio" in self._HDFS_DIRS:
            h5_file = self._dl_manager.download_hdfs(self._HDFS_DIRS["audio"])
            self.h5 = h5py.File(h5_file, "r")

        # get labels
        if "beat_labels" in self._HDFS_DIRS:
            label_dir = self._dl_manager.download_hdfs(self._HDFS_DIRS["beat_labels"])
            self.label_dir = self._dl_manager.extract(label_dir)

        return None

    def _split_data(self, _):
        keys = []
        return {"train": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(os.path.join(self.label_dir, self.LABEL_DIR, key + ".txt")) as f:
            beats = f.readlines()
            beats = [line.rstrip().split() for line in beats]
            beats = [[float(x[0]), int(x[1])] for x in beats]
        np_audio = self.h5[key][:]
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": key,
            "beats.pickle": beats,
            "audio.npy": {"data": np_audio, "dtype": self.AUDIO_NUMPY_DTYPE},
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in keys:
            yield self._load_example(key)
