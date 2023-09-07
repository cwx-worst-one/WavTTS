"""
Leadsheet Chord Recognition Dataset.
"""

import glob
import os
import random
import warnings

import librosa
import numpy as np
import tqdm

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
warnings.filterwarnings("ignore")

# download paths
_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/raw_data/lead-sheet-dataset/audio.zip",
    "chord_labels_test": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/leadsheet_test_truth_chord.zip",
    "chord_labels_train": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/leadsheet_train_truth_chord.zip",
}


class LeadsheetDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Leadsheet Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "leadsheet_chord_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.audio_dir = self._dl_manager.extract(audio_dir)

        # get labels
        label_train_dir = self._dl_manager.download_hdfs(
            _HDFS_DIRS["chord_labels_train"]
        )
        self.label_train_dir = self._dl_manager.extract(label_train_dir)
        label_test_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["chord_labels_test"])
        self.label_test_dir = self._dl_manager.extract(label_test_dir)

        return self.label_train_dir

    def _split_data(self, data_dir):
        train_keys = []
        for k in os.listdir(
            os.path.join(self.label_train_dir, "leadsheet_train_truth_chord")
        ):
            if k[0] != ".":
                k = k.split(".txt")[0]
                audio_fn = os.path.join(self.audio_dir, "audio", k[:5] + "*.mp3")
                checker = glob.glob(audio_fn)
                if k != "" and (len(checker) > 0):
                    train_keys.append(k.replace(".", "$$"))
        test_keys = []
        for k in os.listdir(
            os.path.join(self.label_test_dir, "leadsheet_test_truth_chord")
        ):
            if k[0] != ".":
                k = k.split(".txt")[0]
                audio_fn = os.path.join(self.audio_dir, "audio", k[:5] + "*.mp3")
                checker = glob.glob(audio_fn)
                if k != "" and (len(checker) > 0):
                    test_keys.append(k.replace(".", "$$"))
        return {
            "train": {"keys": [train_keys, "train"]},
            "validation": {"keys": [test_keys, "validation"]},
        }

    def _load_example(self, key, split):
        # read annotations
        if split == "train":
            label_dir = self.label_train_dir
        elif split == "validation":
            label_dir = self.label_test_dir
            split = "test"
        with open(
            os.path.join(
                label_dir,
                "leadsheet_%s_truth_chord" % split,
                key.replace("$$", ".") + ".txt",
            )
        ) as f:
            chords = f.readlines()
            intervals = []
            symbols = []
            for line in chords:
                s = line.strip().split("\t")
                if len(s) != 3:
                    s = line.strip().split(" ")
                if line.strip() != "":
                    start = float(s[0])
                    end = float(s[1])
                    symbol = s[2]
                    intervals.append([start, end])
                    symbols.append(symbol)
        audio_path = os.path.join(self.audio_dir, "audio", key[:5] + "*.mp3")
        audio_path = glob.glob(audio_path)[0]
        np_audio, _ = librosa.load(audio_path, sr=LeadsheetDataset.SAMPLING_RATE)
        return {
            "dataset.txt": LeadsheetDataset.DATASET_NAME,
            "__key__": key,
            "intervals.pickle": intervals,
            "chords.pickle": symbols,
            "audio.npy": np_audio.astype(LeadsheetDataset.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        keys, split = keys
        random.shuffle(keys)
        # generate audio chord pairs then iterate over them
        for key in tqdm.tqdm(keys):
            yield self._load_example(key, split)
