"""
Billboard Structure Recognition Dataset.
"""

import os
import random
import webdataset
import soundfile as sf
import pickle

import h5py

from recipes.structure.webdatasets.structure_dataset import AbstractStructureDataset

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/h5/pop909_structure_audio.h5",
    "structure_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/pop909_segments_a.zip",
}


POP909_VALIDATION_KEYS = [
    "pop909_022",
    "pop909_027",
    "pop909_044",
    "pop909_051",
    "pop909_067",
    "pop909_072",
    "pop909_076",
    "pop909_078",
    "pop909_081",
    "pop909_094",
    "pop909_112",
    "pop909_131",
    "pop909_132",
    "pop909_151",
    "pop909_159",
    "pop909_178",
    "pop909_183",
    "pop909_186",
    "pop909_189",
    "pop909_203",
    "pop909_206",
    "pop909_213",
    "pop909_217",
    "pop909_225",
    "pop909_227",
    "pop909_242",
    "pop909_251",
    "pop909_279",
    "pop909_280",
    "pop909_291",
    "pop909_303",
    "pop909_321",
    "pop909_325",
    "pop909_350",
    "pop909_385",
    "pop909_387",
    "pop909_400",
    "pop909_414",
    "pop909_417",
    "pop909_430",
    "pop909_434",
    "pop909_436",
    "pop909_437",
    "pop909_439",
    "pop909_450",
    "pop909_451",
    "pop909_454",
    "pop909_456",
    "pop909_461",
    "pop909_462",
    "pop909_476",
    "pop909_480",
    "pop909_490",
    "pop909_495",
    "pop909_532",
    "pop909_535",
    "pop909_538",
    "pop909_558",
    "pop909_562",
    "pop909_578",
    "pop909_582",
    "pop909_586",
    "pop909_588",
    "pop909_608",
    "pop909_615",
    "pop909_619",
    "pop909_621",
    "pop909_622",
    "pop909_626",
    "pop909_635",
    "pop909_636",
    "pop909_684",
    "pop909_690",
    "pop909_694",
    "pop909_713",
    "pop909_740",
    "pop909_748",
    "pop909_757",
    "pop909_777",
    "pop909_785",
    "pop909_794",
    "pop909_803",
    "pop909_808",
    "pop909_810",
    "pop909_816",
    "pop909_821",
    "pop909_824",
    "pop909_826",
    "pop909_827",
    "pop909_829",
    "pop909_831",
    "pop909_834",
    "pop909_843",
    "pop909_845",
    "pop909_848",
    "pop909_857",
    "pop909_858",
    "pop909_865",
    "pop909_886",
    "pop909_891",
]


class Pop909Dataset(AbstractStructureDataset):
    """
    Pop909 Dataset
    """

    DATASET_NAME = "pop909_structure"
    DATASET_ROOT_DIR = "/mnt/bn/mir-tasks/structure/pop909_structure/"

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
        train_keys, test_keys = [], []
        for k in os.listdir(os.path.join(data_dir, "pop909_segments_a")):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                if k in POP909_VALIDATION_KEYS:
                    test_keys.append(k)
                else:
                    train_keys.append(k)
        return {"train": {"keys": train_keys}, "test": {"keys": test_keys}}

    def _load_example(self, key):
        # read annotations
        np_audio = self.h5[key][:]
        segs = self._import_segment_annotation(
            os.path.join(self.label_dir, "pop909_segments_a", key + ".txt"),
            len(np_audio) / self.SAMPLING_RATE,
        )
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": key,
            "segment_type.txt": 'r',
            "intervals.pickle": segs['interval'],
            "labels.pickle": segs['labels'],
            "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        for key in keys:
            yield self._load_example(key)


def write_raw_audio():
    dataset = webdataset.WebDataset("/mnt/bn/mir-tasks/structure/pop909_structure/test/shards-0000.tar").decode()
    for sample in dataset:
        audio = sample['audio.npy'] / 32768
        intervals = sample['intervals.pickle']
        labels = sample['labels.pickle']
        key = sample['__key__']
        data = {}
        data['intervals'] = intervals
        data['labels'] = labels
        sf.write(f'../Ripple/data/structure/pop909_audio/{key}.wav', audio, 16000)
        with open(f'../Ripple/data/structure/pop909_labels/{key}.pickle', 'wb') as handle:
            pickle.dump(data, handle)
