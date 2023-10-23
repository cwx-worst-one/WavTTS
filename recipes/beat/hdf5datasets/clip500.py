"""
Internal Clip500 Beat Tracking Dataset.
"""

import os

from recipes.beat.hdf5datasets.beat_hdf5_dataset import BeatHDF5Dataset


class Clip500Dataset(BeatHDF5Dataset):
    """
    Clip500 Dataset
    """

    DATASET_NAME = "clip500_beat"
    LABEL_DIR = "clip500_beats"
    _HDFS_DIRS = {
        "audio": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/audio_h5/clip500_audio.h5",
        "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/labels/clip500_beats.zip",
    }

    def _split_data(self, _):
        keys = []
        for k in os.listdir(os.path.join(self.label_dir, self.LABEL_DIR)):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                keys.append(k)
        return {"test": {"keys": keys}}
