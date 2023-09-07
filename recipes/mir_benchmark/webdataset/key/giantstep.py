"""
GiantStep Key Detection Dataset.
"""

import io
import os
import random

import numpy as np
import soundfile as sf

from recipes.beat.utils.ffmpeg_utils import convert_audio_ffmpeg_path_to_bytes
from recipes.key_detection.constants import KEY_MAP
from samantha.dataio.webdataset.builder import AbstractWebDatasetGeneratorBasedBuilder


class GiantStepDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    GiantStep Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "giantstep_key_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32
    DATASET_ROOT_DIR = os.path.join(
        "/mnt/bn/audio-diffusion/mir_benchmark/key/", DATASET_NAME
    )

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        self.dir = "/mnt/bn/mir-tasks/GiantStep/"
        return self.dir

    def _split_data(self, data_dir):
        keys = []
        audio_list = os.listdir(os.path.join(data_dir, "giantstep_audio"))
        for k in audio_list:
            if k == ".DS_Store":
                continue
            k = k.split(".mp3")[0]
            if os.path.exists(os.path.join(f"../key/", k + ".key")):
                keys.append(k)

        return {"test": {"keys": keys}}

    def _load_example(self, _id):
        intervals = []
        symbols = []

        byte_audio = convert_audio_ffmpeg_path_to_bytes(
            os.path.join(self.dir, "giantstep_audio", _id + ".mp3"),
            self.SAMPLING_RATE,
            True,
        )
        np_audio, samplerate = sf.read(io.BytesIO(byte_audio))

        # read annotations
        with open(os.path.join(f"../key/", _id + ".key"), "r") as f:
            ss = f.read().splitlines()[0]
            key, _mode = ss.strip().split(" ")
            if _mode == "minor":
                mode = 2
            elif _mode == "major":
                mode = 1

            intervals.append([0, len(np_audio) / self.SAMPLING_RATE])
            symbols.append((KEY_MAP[key], mode))

        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": _id.replace(".", "_"),
            "intervals.pickle": intervals,
            "keys.pickle": symbols,
            "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio beat pairs then iterate over them
        for key in keys:
            yield self._load_example(key)


ds = GiantStepDataset()
ds.create_dataset()
