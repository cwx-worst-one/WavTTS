"""
KSong vocal2midi Dataset.
"""

import io
import os

import soundfile as sf

from recipes.beat.utils.ffmpeg_utils import convert_audio_ffmpeg_path_to_bytes
from recipes.vocal2midi.webdatasets.vocal2midi_builder import Vocal2MidiDataset

_HDFS_DIRS = {
    "dataset": "hdfs://harunava/home/byte_speech_sv/weitsung.lu/"
    "vocal_melody_extraction/dataset/k_song_dataset_internal.tar.gz"
}


class KsongDataset(Vocal2MidiDataset):
    """
    Ksong Dataset
    """

    DATASET_NAME = "ksong_v2m_24kHz"
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
        train_keys, test_keys = [], []

        train_list = os.listdir(
            os.path.join(data_dir, "k_song_dataset_internal", "audio")
        )
        train_list.sort()

        for _id in train_list[: int(len(train_list) * 0.9)]:
            train_keys.append(_id.split(".")[0])

        for _id in train_list[int(len(train_list) * 0.9) :]:
            test_keys.append(_id.split(".")[0])

        return {
            "train": {"keys": [train_keys, "train"]},
            "test": {"keys": [test_keys, "test"]},
        }

    def _get_audio(self, key):
        audio_path = os.path.join(
            self.dataset_dir, "k_song_dataset_internal", "audio", key + ".mp3"
        )
        if not os.path.exists(audio_path):
            audio_path = os.path.join(
                self.dataset_dir, "k_song_dataset_internal", "audio", key + ".wav"
            )

        conv_audio_bytes = convert_audio_ffmpeg_path_to_bytes(
            audio_path, self.SAMPLING_RATE, mono=True
        )
        np_audio, _ = sf.read(io.BytesIO(conv_audio_bytes))
        return np_audio

    def _load_example(self, key):
        try:
            np_audio = self._get_audio(key)
            note_seq = self._midi2note(
                os.path.join(
                    self.dataset_dir, "k_song_dataset_internal", "midi", key + ".mid"
                )
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


ds = KsongDataset()
ds.create_dataset()
