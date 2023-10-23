"""
Karaoke Beat Tracking Dataset.
"""

import os

from recipes.beat.hdf5datasets.beat_hdf5_dataset import BeatHDF5Dataset


class KaraokeDataset(BeatHDF5Dataset):
    """
    Karaoke Dataset
    """

    DATASET_NAME = "karaoke_beat"
    LABEL_DIR = "karaoke_truth_beats"
    _HDFS_DIRS = {
        "audio": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/audio_h5/karaoke_audio.h5",
        "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/labels/karaoke_truth_beats.zip",
    }

    def _split_data(self, _):
        keys = []
        self.label_dir = os.path.join(self.label_dir, "data")
        for k in os.listdir(os.path.join(self.label_dir, self.LABEL_DIR)):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                keys.append(k)
        return {"train": {"keys": keys}}
