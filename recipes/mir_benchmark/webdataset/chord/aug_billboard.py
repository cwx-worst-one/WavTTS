"""
Billboard Chord Recognition Dataset.
"""

import os
import random
import warnings

import librosa
import numpy as np
import torch
import torchaudio
import tqdm

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
warnings.filterwarnings("ignore")

# download paths
_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/raw_data/billboard/audio",
    "chord_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/billboard_chords.zip",
}


def note2num(note):
    return {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "Fb": 4,
        "E#": 5,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
        "Cb": 11,
    }.get(note, 0)


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class BillboardDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Billboard Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "billboard_chord_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        self.audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["chord_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "billboard_chords")):
            if k[0] != ".":
                k = k.split(".txt")[0][1:]
                audio_fn = os.path.join(self.audio_dir, str(int(k)) + ".mp3")
                if k != "" and os.path.exists(audio_fn):
                    keys.append(k.replace(".", "$$"))
        return {"train": {"keys": keys}}

    def _load_example(self, key):
        # read annotations
        with open(
            os.path.join(
                self.label_dir,
                "billboard_chords",
                "b" + key.replace("$$", ".") + ".txt",
            )
        ) as f:
            chords = f.readlines()
            intervals = []
            symbols = []
            for line in chords:
                s = line.strip().split("\t")
                if line.strip() != "":
                    start = float(s[0])
                    end = float(s[1])
                    symbol = s[2]
                    intervals.append([start, end])
                    symbols.append(symbol)
        audio_path = os.path.join(
            self.audio_dir, str(int(key.replace("$$", "."))) + ".mp3"
        )
        np_audio, _ = librosa.load(audio_path, sr=BillboardDataset.SAMPLING_RATE)

        return_dict = []
        for pitch_shift_rate in [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
            shift = pitch_shift_rate * 100
            shifted, _ = torchaudio.sox_effects.apply_effects_tensor(
                torch.tensor(np_audio).unsqueeze(0),
                BillboardDataset.SAMPLING_RATE,
                [["pitch", f"{shift}"], ["rate", f"{BillboardDataset.SAMPLING_RATE}"]],
                channels_first=True,
            )
            new_symbols = []
            for symbol in symbols:
                elems = symbol.split(":")
                if len(elems) == 1:
                    new_symbols.append(symbol)
                else:
                    root, tension = elems
                    new_root_ix = note2num(root) + pitch_shift_rate
                    if new_root_ix < 0:
                        new_root_ix += len(NOTES)
                    elif new_root_ix >= 12:
                        new_root_ix -= len(NOTES)
                    new_symbols.append(NOTES[new_root_ix] + ":" + tension)
            return_dict.append(
                {
                    "dataset.txt": BillboardDataset.DATASET_NAME,
                    "__key__": key + "_%d" % pitch_shift_rate,
                    "intervals.pickle": intervals,
                    "chords.pickle": symbols,
                    "audio.npy": np_audio.astype(BillboardDataset.AUDIO_NUMPY_DTYPE),
                }
            )
        return return_dict

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio chord pairs then iterate over them
        for key in tqdm.tqdm(keys):
            shifted_dicts = self._load_example(key)
            for single_dict in shifted_dicts:
                yield single_dict
