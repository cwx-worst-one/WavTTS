"""
Leadsheet Chord Recognition Dataset.
"""

import os
import random

import webdataset
import pickle
import braceexpand
import soundfile as sf
import h5py
import numpy as np

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

_HDFS_DIRS = {
    "audio_test": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/audio_h5/leadsheet_test_chord_audio.h5",
    "audio_train": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/audio_h5/leadsheet_train_chord_audio.h5",
    "chord_labels_test": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/leadsheet_test_truth_chord.zip",
    "chord_labels_train": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/leadsheet_train_truth_chord.zip",
}


class LeadsheetDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    leadsheet Dataset
    """

    SAMPLING_RATE = 16000
    DATASET_NAME = "leadsheet_chord"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        h5_test_file = self._dl_manager.download_hdfs(_HDFS_DIRS["audio_test"])
        self.h5_test = h5py.File(h5_test_file, "r")
        h5_train_file = self._dl_manager.download_hdfs(_HDFS_DIRS["audio_train"])
        self.h5_train = h5py.File(h5_train_file, "r")

        # get labels
        label_test_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["chord_labels_test"])
        self.label_test_dir = self._dl_manager.extract(label_test_dir)
        label_train_dir = self._dl_manager.download_hdfs(
            _HDFS_DIRS["chord_labels_train"]
        )
        self.label_train_dir = self._dl_manager.extract(label_train_dir)

        return None

    def _split_data(self, data_dir):
        train_keys = []
        for k in os.listdir(
            os.path.join(self.label_train_dir, "leadsheet_train_truth_chord")
        ):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5_train.keys():
                train_keys.append(k)

        validation_keys = []
        for k in os.listdir(
            os.path.join(self.label_test_dir, "data", "leadsheet_test_truth_chord")
        ):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5_test.keys():
                validation_keys.append(k)
        return {
            "train": {
                "keys_path_pairs": [
                    train_keys,
                    os.path.join(self.label_train_dir, "leadsheet_train_truth_chord"),
                    self.h5_train,
                ]
            },
            "validation": {
                "keys_path_pairs": [
                    validation_keys,
                    os.path.join(
                        self.label_test_dir, "data", "leadsheet_test_truth_chord"
                    ),
                    self.h5_test,
                ]
            },
        }

    def _load_example(self, key, path, h5):
        # read annotations
        with open(os.path.join(path, key + ".txt")) as f:
            chords = f.read().splitlines()
            intervals = []
            symbols = []

            for i, line in enumerate(chords):
                s = line.split("\t")
                if len(s) != 3:
                    s = line.split(" ")
                s = line.split("\t")
                start = float(s[0])
                end = float(s[1])
                symbol = s[2]
                intervals.append([start, end])
                symbols.append(symbol)

        if len(intervals) != 0:
            np_audio = h5[key][:]
            return {
                "dataset.txt": self.DATASET_NAME,
                "__key__": key.replace(".", ""),
                "intervals.pickle": intervals,
                "chords.pickle": symbols,
                "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
            }
        else:
            return None

    def _generate_examples(self, keys_path_pairs):
        keys, data_path, h5 = keys_path_pairs
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in keys:
            data = self._load_example(key, data_path, h5)
            if data is not None:
                yield data

def write_raw_audio():
    url = list(braceexpand.braceexpand("/mnt/bd/sheetdoctor/chord/leadsheet_chord/validation/shards-{0000..0005}.tar"))
    dataset = webdataset.WebDataset(url).decode()
    for sample in dataset:
        audio = sample['audio.npy']
        intervals = sample['intervals.pickle']
        labels = sample['chords.pickle']
        key = sample['__key__']
        data = {}
        data['intervals'] = intervals
        data['labels'] = labels
        sf.write(f'../Ripple/data/chord/leadsheet_audio/{key}.wav', audio, 16000)
        with open(f'../Ripple/data/chord/leadsheet_labels/{key}.pickle', 'wb') as handle:
            pickle.dump(data, handle)
