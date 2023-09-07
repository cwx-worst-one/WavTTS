"""
Slakh Transcription Dataset.
"""

import glob
import io
import os
import random
import warnings

import numpy as np
import pretty_midi
import soundfile as sf
import tqdm
import yaml

from recipes.beat.utils.ffmpeg_utils import convert_audio_ffmpeg_path_to_bytes
from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

# ignore warning messages from librosa
warnings.filterwarnings("ignore")

# global instruments
SLAKH_INSTRUMENTS = [
    "Bass",
    "Brass",
    "Chromatic Percussion",
    "Drums",
    "Guitar",
    "Organ",
    "Piano",
    "Pipe",
    "Reed",
    "Strings",
    "Synth Lead",
    "Synth Pad",
]

# download paths
_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/SheetDoctor/raw_data/slakh2100_ori.zip"
}


class SlakhDataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    Slakh Dataset
    """

    SAMPLING_RATE = 24000
    DATASET_NAME = "slakh_transcription_24kHz"
    AUDIO_NUMPY_DTYPE = np.float32
    DATASET_ROOT_DIR = os.path.join(
        "/mnt/bn/audio-diffusion/mir_benchmark/transcription/", DATASET_NAME
    )

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        # slakh_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        # self.slakh_dir = self._dl_manager.extract(slakh_dir)
        self.slakh_dir = "/mnt/bn/mir-tasks/slakh2100_ori/"

        return self.slakh_dir

    def _split_data(self, data_dir):
        train_keys = os.listdir(os.path.join(self.slakh_dir, "train"))
        validation_keys = os.listdir(os.path.join(self.slakh_dir, "validation"))
        test_keys = os.listdir(os.path.join(self.slakh_dir, "test"))

        return {
            "train": {"keys": [train_keys, "train"]},
            "validation": {"keys": [validation_keys, "validation"]},
            "test": {"keys": [test_keys, "test"]},
        }

    def _get_audio(self, audio_path):
        conv_audio_bytes = convert_audio_ffmpeg_path_to_bytes(
            audio_path, self.SAMPLING_RATE, mono=True
        )
        np_audio, _ = sf.read(io.BytesIO(conv_audio_bytes))
        return np_audio

    def _load_example(self, key, split):
        audio_dict, note_dict = self._get_audio_and_note(key, split)
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": key,
            "audio.pickle": audio_dict,
            "notes.pickle": note_dict,
        }

    def _get_audio_and_note(self, key, split):
        audio_dict = {}
        note_dict = {}

        with open(
            os.path.join(self.slakh_dir, split, key, "metadata.yaml"), "r"
        ) as stream:
            data = yaml.safe_load(stream)
            for stem in data["stems"]:
                instrument = data["stems"][stem]["inst_class"].split(" (")[0]
                if instrument not in SLAKH_INSTRUMENTS and instrument not in [
                    "Synth Effects",
                    "Ethnic",
                    "Percussive",
                    "Sound Effects",
                    "Sound effects",
                ]:
                    raise f"Illegal instrument {instrument} in {stem} {key}"

                # get audio
                audio_path = os.path.join(
                    self.slakh_dir, split, key, "stems", stem + ".flac"
                )
                if os.path.exists(audio_path):
                    if instrument in audio_dict:
                        stem_audio = self._get_audio(audio_path)
                        if len(stem_audio) < len(audio_dict[instrument]):
                            stem_audio = np.pad(
                                stem_audio,
                                (0, (len(audio_dict[instrument]) - len(stem_audio))),
                                "constant",
                                constant_values=0,
                            )
                        if len(stem_audio) > len(audio_dict[instrument]):
                            audio_dict[instrument] = np.pad(
                                audio_dict[instrument],
                                (0, (len(stem_audio) - len(audio_dict[instrument]))),
                                "constant",
                                constant_values=0,
                            )
                        audio_dict[instrument] += stem_audio
                    else:
                        audio_dict[instrument] = self._get_audio(audio_path)

                    # get labels
                    label_path = os.path.join(
                        self.slakh_dir, split, key, "MIDI", stem + ".mid"
                    )
                    midi_data = pretty_midi.PrettyMIDI(label_path)
                    instrument_midi = midi_data.instruments
                    if len(instrument_midi) > 1:
                        raise f"Multiple instrument in one track {instrument} {stem} {key}"
                    instrument_midi = instrument_midi[0]
                    if instrument not in note_dict:
                        note_dict[instrument] = []
                    for note in instrument_midi.notes:
                        note_dict[instrument].append(
                            {"start": note.start, "end": note.end, "pitch": note.pitch}
                        )

        if audio_dict.keys() != note_dict.keys():
            raise f"Audio and Midi instrument not match {key}"

        for inst in note_dict:
            note_dict[inst] = sorted(note_dict[inst], key=lambda x: x["start"])

        return audio_dict, note_dict

    def _generate_examples(self, keys):
        keys, split = keys
        random.shuffle(keys)
        for key in keys:
            output = self._load_example(key, split)
            if output is not None:
                yield output


ds = SlakhDataset()
ds.create_dataset()
