"""
RWC Beat Tracking Dataset.
"""

import os

from recipes.beat.hdf5datasets.beat_hdf5_dataset import BeatHDF5Dataset


class RWCDataset(BeatHDF5Dataset):
    """
    RWC Dataset
    """

    DATASET_NAME = "rwc_beat"
    LABEL_DIR = "rwc_truth_beats"
    _HDFS_DIRS = {
        "audio": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/audio_h5/rwc_audio.h5",
        "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/labels/rwc_truth_beats.zip",
    }

    def _split_data(self, _):
        keys = []
        for k in os.listdir(os.path.join(self.label_dir, self.LABEL_DIR)):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                if len(k.split("-")[-1]) == 5:
                    k = k.replace("0", "", 1)
                keys.append(k)
        return {"train": {"keys": keys}}
