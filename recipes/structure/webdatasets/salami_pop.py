"""
Billboard Structure Recognition Dataset.
"""

import os
import random

import h5py

from recipes.structure.webdatasets.structure_dataset import AbstractStructureDataset

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/h5/salamipop_audio.h5",
    "structure_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/salami_function_segments.zip",
}


SALAMIPOP_VALIDATION_KEYS = [
    "0637_1",
    "0637_2",
    "0638_1",
    "0638_2",
    "0639_1",
    "0639_2",
    "0640_1",
    "0640_2",
    "0642_1",
    "0642_2",
    "0643_1",
    "0643_2",
    "0644_1",
    "0644_2",
    "0645_1",
    "0645_2",
    "0703_1",
    "0703_2",
    "0704_1",
    "0704_2",
    "0706_1",
    "0706_2",
    "0546_1",
    "0546_2",
    "0547_1",
    "0547_2",
    "0548_1",
    "0549_1",
    "0549_2",
    "0550_1",
    "0550_2",
    "0551_1",
    "0551_2",
    "1575_1",
    "1576_1",
    "1578_1",
    "1579_1",
    "1580_1",
    "1581_1",
    "1582_1",
    "1583_1",
    "1584_1",
    "0318_1",
    "0318_2",
    "0319_1",
    "0319_2",
    "0320_1",
    "0320_2",
    "0322_1",
    "0322_2",
    "0323_1",
    "0323_2",
    "0324_1",
    "0325_1",
    "0325_2",
    "0326_1",
    "0326_2",
    "0327_1",
    "0327_2",
    "0328_1",
    "0328_2",
    "0011_1",
    "0011_2",
    "0012_1",
    "0012_2",
    "0013_1",
    "0013_2",
    "0014_1",
    "0014_2",
    "1647_1",
    "1648_1",
    "1650_1",
    "1652_1",
    "1653_1",
    "1654_1",
    "1655_1",
]


class SalamiPopDataset(AbstractStructureDataset):
    """
    Salami Live Dataset
    """

    DATASET_NAME = "salamipop_structure"
    DATASET_ROOT_DIR = "/mnt/bn/mir-tasks/structure/salamipop_structure/"

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
        for k in os.listdir(os.path.join(data_dir, "salami_function_segments")):
            k = k.split(".txt")[0]
            if k != "" and k.split("_")[0] in self.h5.keys():
                if k in SALAMIPOP_VALIDATION_KEYS:
                    validation_keys.append(k)
                else:
                    train_keys.append(k)
        return {"train": {"keys": train_keys}, "validation": {"keys": validation_keys}}

    def _load_example(self, key):
        # read annotations
        np_audio = self.h5[key.split("_")[0]][:]
        segs = self._import_segment_annotation(
            os.path.join(self.label_dir, "salami_function_segments", key + ".txt"),
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
