"""
Billboard Structure Recognition Dataset.
"""

import os
import random

import h5py

from recipes.structure.webdatasets.structure_dataset import AbstractStructureDataset

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/" "SheetDoctor/h5/rwc_audio.h5",
    "structure_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/rwc_segments.zip",
}

RWC_VALIDATION_KEYS = [
    "RM-G001",
    "RM-G004",
    "RM-G007",
    "RM-G010",
    "RM-G013",
    "RM-G016",
    "RM-G019",
    "RM-G022",
    "RM-G025",
    "RM-G037",
    "RM-G067",
    "RM-P085",
    "RM-P099",
    "RM-P004",
    "RM-P066",
    "RM-P048",
    "RM-P072",
    "RM-P027",
    "RM-P047",
    "RM-P037",
    "RM-P005",
    "RM-P067",
    "RM-P057",
    "RM-P097",
    "RM-P021",
]


class RWCDataset(AbstractStructureDataset):
    """
    Rwc Dataset
    """

    DATASET_NAME = "rwc_structure"
    DATASET_ROOT_DIR = "/mnt/bn/mir-tasks/structure/rwc_structure/"

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
        train_keys, validation_keys = [], []
        for k in os.listdir(os.path.join(data_dir, "rwc_segments")):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                if k in RWC_VALIDATION_KEYS:
                    validation_keys.append(k)
                else:
                    train_keys.append(k)
        return {"train": {"keys": train_keys}, "validation": {"keys": validation_keys}}

    def _load_example(self, key):
        # read annotations
        np_audio = self.h5[key][:]
        segs = self._import_segment_annotation(
            os.path.join(self.label_dir, "rwc_segments", key + ".txt"),
            len(np_audio) / self.SAMPLING_RATE,
        )
        if len(segs["interval"]) == 0:
            return None
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
            data = self._load_example(key)
            if data is not None:
                yield data
