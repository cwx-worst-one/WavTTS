"""
SMC Beat Tracking Dataset.
"""

import os

from recipes.beat.hdf5datasets.beat_hdf5_dataset import BeatHDF5Dataset


class SMCDataset(BeatHDF5Dataset):
    """
    SMC Dataset
    """

    DATASET_NAME = "smc_beat"
    LABEL_DIR = "smc_truth_beats"
    _HDFS_DIRS = {
        "audio": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/audio_h5/smc_audio.h5",
        "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
        "SheetDoctor/labels/smc_truth_beats.zip",
    }

    def _split_data(self, _):
        keys = []
        for k in os.listdir(os.path.join(self.label_dir, self.LABEL_DIR)):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                keys.append(k)
        return {"train": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(os.path.join(self.label_dir, "smc_truth_beats", key + ".txt")) as f:
            beats = f.readlines()
            beats = [[float(line.rstrip()), 0] for line in beats]
        np_audio = self.h5[key][:]
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": key,
            "beats.pickle": beats,
            "audio.npy": {"data": np_audio, "dtype": self.AUDIO_NUMPY_DTYPE},
        }
