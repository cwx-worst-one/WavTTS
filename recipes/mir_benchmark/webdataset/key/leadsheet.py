"""
Leadsheet Key Recognition Dataset.
"""

import io
import os
import random
import warnings

import numpy as np
import soundfile as sf
import tqdm

from recipes.beat.utils.ffmpeg_utils import convert_audio_ffmpeg_path_to_bytes
from recipes.key_detection.constants import KEY_MAP, leadsheet_test_id
from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
warnings.filterwarnings("ignore")

# download paths
_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/raw_data/lead-sheet-dataset/audio.zip",
    "key_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/leadsheet_keys.zip",
}


class LeadsheetDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Leadsheet Dataset with fixed labels
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "leadsheet_key_24kHz"
    DATASET_ROOT_DIR = os.path.join(
        "/mnt/bn/audio-diffusion/mir_benchmark/key/", DATASET_NAME
    )
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.audio_dir = self._dl_manager.extract(audio_dir)

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["key_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        train_keys = []
        test_keys = []
        for k in os.listdir(os.path.join(data_dir, "leadsheet_keys")):
            k = k.split(".txt")[0]
            if k != ".DS_Store":
                if k in leadsheet_test_id:
                    test_keys.append(k)
                else:
                    train_keys.append(k)

        return {"train": {"keys": train_keys}, "validation": {"keys": test_keys}}

    def _load_example(self, _id):
        intervals = []
        symbols = []

        audio_path = os.path.join(self.audio_dir, "audio", _id + ".mp3")
        try:
            conv_audio_bytes = convert_audio_ffmpeg_path_to_bytes(
                audio_path, self.SAMPLING_RATE, mono=True
            )
            np_audio, _ = sf.read(io.BytesIO(conv_audio_bytes))
        except:
            print(_id)
            return None

        # read annotations
        with open(
            os.path.join(self.label_dir, "leadsheet_keys", _id + ".txt"), "r"
        ) as f:
            ss = f.read().splitlines()

            # has a silence at the beginning
            if len(ss) > 2:
                ss = ss[1:]

            s = ss[0].split("\t")

            start = float(0)
            end = float(len(np_audio) / 16000)

            if s[1] == "silence" or s[1] == "N" or s[1] == "end":
                key = KEY_MAP[s[1]]
            else:
                key, mode = s[1].split(":")
                key = KEY_MAP[key]
                if mode == "None":
                    mode = 0
                mode = int(mode)

            if key == 0:
                mode = 0

            intervals.append([start, end])
            symbols.append((key, mode))

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
            data = self._load_example(key)
            if data is not None:
                yield data


ds = LeadsheetDataset()
ds.create_dataset()
