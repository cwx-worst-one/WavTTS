"""
Billboard Key Recognition Dataset.
"""

import io
import os
import random
import warnings

import numpy as np
import soundfile as sf
import tqdm

from recipes.beat.utils.ffmpeg_utils import convert_audio_ffmpeg_path_to_bytes
from recipes.key_detection.constants import KEY_MAP
from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
warnings.filterwarnings("ignore")

# download paths
_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/raw_data/billboard/audio",
    "key_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/billboard_keys.zip",
}


class BillboardDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Billboard Dataset with fixed labels
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "billboard_key_24kHz"
    DATASET_ROOT_DIR = os.path.join(
        "/mnt/bn/audio-diffusion/mir_benchmark/key/", DATASET_NAME
    )
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        self.audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["key_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "billboard_keys")):
            k = k.split(".txt")[0]
            if k != ".DS_Store":
                keys.append(k)
        return {"train": {"keys": keys}}

    def _load_example(self, _id):
        intervals = []
        symbols = []

        # read annotations
        with open(
            os.path.join(self.label_dir, "billboard_keys", _id + ".txt"), "r"
        ) as f:
            ss = f.read().splitlines()
            for i, line in enumerate(ss):
                s = line.split("\t")
                try:
                    next_s = ss[i + 1].split("\t")
                except IndexError:
                    break

                start = float(s[0])
                end = float(next_s[0])

                if s[1] == "silence" or s[1] == "N" or s[1] == "end":
                    key = KEY_MAP[s[1]]
                else:
                    key, mode = s[1].split(":")
                    key = KEY_MAP[key]
                    mode = int(mode)

                if key == 0:
                    mode = 0

                intervals.append([start, end])
                symbols.append((key, mode))

        audio_path = os.path.join(self.audio_dir, str(int(_id[1:])) + ".mp3")

        conv_audio_bytes = convert_audio_ffmpeg_path_to_bytes(
            audio_path, self.SAMPLING_RATE, mono=True
        )
        np_audio, _ = sf.read(io.BytesIO(conv_audio_bytes))
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": _id,
            "intervals.pickle": intervals,
            "keys.pickle": symbols,
            "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        # generate audio chord pairs then iterate over them
        for key in tqdm.tqdm(keys):
            yield self._load_example(key)


ds = BillboardDataset()
ds.create_dataset()
