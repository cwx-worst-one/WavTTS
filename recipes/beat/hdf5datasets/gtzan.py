"""
GTZAN Beat Tracking Dataset.
"""

import os

from recipes.beat.hdf5datasets.beat_hdf5_dataset import BeatHDF5Dataset


class GTZANDataset(BeatHDF5Dataset):
    """
    GTZAN Dataset
    """

    DATASET_NAME = "gtzan_beat"
    LABEL_DIR = "gtzan_truth_beats_new"
    _HDFS_DIRS = {
        "audio": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/audio_h5/gtzan_audio_new.h5",
        "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/labels/gtzan_truth_beats_new.zip",
    }

    def _split_data(self, _):
        keys = []
        for k in os.listdir(os.path.join(self.label_dir, self.LABEL_DIR)):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                keys.append(k)
        return {"validation": {"keys": keys}}
