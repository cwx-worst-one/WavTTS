"""
MIRST500 vocal2midi Dataset.
"""

import io
import os
from glob import glob

import soundfile as sf

from recipes.beat.utils.ffmpeg_utils import convert_audio_ffmpeg_path_to_bytes
from recipes.vocal2midi.webdatasets.vocal2midi_builder import Vocal2MidiDataset

_HDFS_DIRS = {
    "dataset": "hdfs://harunava/home/byte_speech_sv/weitsung.lu/"
    "vocal_melody_extraction/dataset/MIR_ST500.tar.gz"
}


class MIRST500Dataset(Vocal2MidiDataset):
    """
    MIRST500 Dataset
    """

    DATASET_NAME = "mirst500_v2m_24kHz"
    SAMPLING_RATE = 24000
    CHUNK_LENGTH = 24000 * 20
    DATASET_ROOT_DIR = os.path.join(
        "/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/", DATASET_NAME
    )

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get labels
        dataset_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["dataset"])
        self.dataset_dir = self._dl_manager.extract(dataset_dir)
        return self.dataset_dir

    def _split_data(self, data_dir):
        train_keys, validation_keys, test_keys = [], [], []
        for _id in os.listdir(os.path.join(data_dir, "MIR_ST500", "test")):
            test_keys.append(_id)

        train_list = os.listdir(os.path.join(data_dir, "MIR_ST500", "train"))
        train_list.sort()

        for _id in train_list[: int(len(train_list) * 0.9)]:
            train_keys.append(_id)

        for _id in train_list[int(len(train_list) * 0.9) :]:
            validation_keys.append(_id)

        return {
            "train": {"keys": [train_keys, "train"]},
            "validation": {"keys": [validation_keys, "validation"]},
            "test": {"keys": [test_keys, "test"]},
        }

    def _get_audio(self, key):
        sample_dir = glob(os.path.join(self.dataset_dir, "MIR_ST500", "**", key))[0]
        audio_path = os.path.join(sample_dir, key + ".wav")
        conv_audio_bytes = convert_audio_ffmpeg_path_to_bytes(
            audio_path, self.SAMPLING_RATE, mono=True
        )
        np_audio, _ = sf.read(io.BytesIO(conv_audio_bytes))
        return np_audio

    def _load_example(self, key):
        try:
            np_audio = self._get_audio(key)
            note_seq = self._midi2note(
                os.path.join(self.dataset_dir, "MIR_ST500", "midi", key + ".mid")
            )
            onset_seq = self._get_onset(note_seq)

            return {
                "dataset.txt": self.DATASET_NAME,
                "__key__": key,
                "note_seq.pickle": note_seq,
                "onset_seq.pickle": onset_seq,
                "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
            }
        except:
            print(key)


ds = MIRST500Dataset()
ds.create_dataset()
