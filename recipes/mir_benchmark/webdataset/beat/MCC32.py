"""
MCC Beat Tracking Dataset.
"""

import glob
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
    "SheetDoctor/raw_data/MCC158_audio.zip",
    "beat_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/MCC32_label.zip",
}


class MCC32Dataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    MCC32 Dataset with fixed labels
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "MCC32_beat_24kHz"
    DATASET_ROOT_DIR = os.path.join(
        "/mnt/bn/audio-diffusion/mir_benchmark/beat/", DATASET_NAME
    )
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        audio_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.audio_dir = self._dl_manager.extract(audio_dir)

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["beat_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        keys = []
        for k in os.listdir(os.path.join(data_dir, "MCC32_label")):
            k = k.split(".txt")[0]
            if k != ".DS_Store":
                keys.append(k)
        return {"test": {"keys": keys}}

    def _load_example(self, _id):
        audio_path = os.path.join(self.audio_dir, "MCC158_audio", _id + "**.mp3")
        audio_path = glob.glob(audio_path)[0]

        conv_audio_bytes = convert_audio_ffmpeg_path_to_bytes(
            audio_path, self.SAMPLING_RATE, mono=True
        )
        np_audio, _ = sf.read(io.BytesIO(conv_audio_bytes))

        # read annotations
        with open(os.path.join(self.label_dir, "MCC32_label", _id + ".txt"), "r") as f:
            beats = f.readlines()
            beats = [line.rstrip().split() for line in beats]
            beats = [[float(x[0]), int(x[1])] for x in beats]

        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": _id,
            "beats.pickle": beats,
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


ds = MCC32Dataset()
ds.create_dataset()
