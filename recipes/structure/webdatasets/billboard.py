"""
Billboard Structure Recognition Dataset.
"""

import os
import random

import h5py

from recipes.structure.webdatasets.structure_dataset import AbstractStructureDataset

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/h5/billboard_audio.h5",
    "structure_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/billboard_segments.zip",
}


class BillboardDataset(AbstractStructureDataset):
    """
    Billboard Dataset
    """

    DATASET_NAME = "billboard_structure"
    DATASET_ROOT_DIR = "/mnt/bn/mir-tasks/structure/billboard_structure/"

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        h5_file = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.h5 = h5py.File(h5_file, "r")

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["structure_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "billboard_segments")):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                keys.append(k)
        return {"train": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        np_audio = self.h5[key][:]
        segs = self._import_segment_annotation(
            os.path.join(self.label_dir, "billboard_segments", key + ".txt"),
            len(np_audio) / self.SAMPLING_RATE,
        )
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": key,
            "segment_type.txt": "r",
            "intervals.pickle": segs["interval"],
            "labels.pickle": segs["labels"],
            "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        for key in keys:
            yield self._load_example(key)
