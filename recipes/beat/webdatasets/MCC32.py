"""
MCC Beat Tracking Dataset.
"""

import os
import random

import h5py
import numpy as np
import webdataset
import soundfile as sf

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/audio_h5/MCC32_audio.h5",
    "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/MCC32_label.zip",
}


class MCC32Dataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    MCC32 Dataset
    """

    SAMPLING_RATE = 16000
    DATASET_NAME = "MCC32_vocal_beat"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        h5_file = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.h5 = h5py.File(h5_file, "r")

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["beat_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "MCC32_label")):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                keys.append(k.replace(".", "$$"))
        return {"test": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(
            os.path.join(self.label_dir, "MCC32_label", key.replace("$$", ".") + ".txt")
        ) as f:
            beats = f.readlines()
            beats = [line.rstrip().split() for line in beats]
            beats = [[float(x[0]), int(x[1])] for x in beats]
        np_audio = self.h5[key.replace("$$", ".")][:]
        return {
            "dataset.txt": MCC32Dataset.DATASET_NAME,
            "__key__": key,
            "beats.pickle": beats,
            "audio.npy": np_audio.astype(MCC32Dataset.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in keys:
            if key != "YXW-给电影人的情书-51BPM":  # this file has empty labels
                yield self._load_example(key)


def write_raw_audio():
    dataset = webdataset.WebDataset("/mnt/bn/mir-tasks/beat_vocal/MCC32_vocal_beat/test/shards-0000.tar").decode()
    for sample in dataset:
        audio = sample['audio.npy']
        key = sample['__key__']
        data = {}
        sf.write(f'../Ripple/data/beat/MCC32_audio/{key}.wav', audio, 16000)
