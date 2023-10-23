import atexit
import os
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import List
import shlex
import subprocess

import tqdm
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


def _make_ffmpeg_options(sampling_rate: float = 0.0, mono: bool = False):
    # downmixing option
    channel_cmd = "-ac 1 " if mono else "-ac 2 "

    # downsampling option
    resampling_cmd = f"-ar {str(sampling_rate)}" if sampling_rate else ""

    return channel_cmd + resampling_cmd


def convert_audio_ffmpeg_file_to_file(
    input_file: str, output_file: str, sampling_rate: float = 0.0, mono: bool = False
) -> bytes or None:
    """Resamples audio using ffmpeg given an audio file.

    Args:
        input_file (file): Input audio file
        sample_rate (float): Target sample rate
        mono (bool): Convert the input to mono if True.
                     Convert the input to stereo if False.

    Returns:
        out (bytes): Processed audio
    """
    cmd = f"ffmpeg -vn -i {input_file} \
        {_make_ffmpeg_options(sampling_rate, mono)} -f wav {output_file}"

    cp = subprocess.run(
        shlex.split(cmd),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    if "invalid data found when processing input" in str(cp.stderr).lower():
        raise RuntimeError(cp.stderr)

    return cp.stdout


class AudioFolderIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, audio_dir: str, sampling_rate: int, mono: bool):
        self.audio_dir = audio_dir
        self.sampling_rate = sampling_rate
        self.mono = mono

    def __iter__(self):
        for f in os.listdir(self.audio_dir):
            f_path = Path(f)
            if sf.check_format(f_path.suffix.split(".")[-1].upper()):
                audio, sr = sf.read(os.path.join(self.audio_dir, f_path))
                if sr != self.sampling_rate:
                    raise RuntimeError(
                        f"{f} sampling rate {sr} does not match expected sampling rate "
                        "{self.sampling_rate}. Please convert the audio."
                    )
                if self.mono and len(audio.shape) != 1:
                    raise RuntimeError(f"{f} is not mono but expected mono.")
                if not self.mono and len(audio.shape) != 2:
                    raise RuntimeError(f"{f} is not stereo but expected stereo.")

                # C x num_samples
                yield torch.from_numpy(audio), str(f_path.stem)


class AudioInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        sampling_rate,
        mono,
        audio_dir: str = None,
        audio_files: List[str] = [],
        audio_urls: List[str] = [],
        batch_size: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.audio_dir = audio_dir
        self.audio_files = audio_files
        self.audio_urls = audio_urls
        self.batch_size = batch_size

        atexit.register(self.cleanup)

        self._converted_temp_dir = tempfile.TemporaryDirectory()

        if self.audio_dir is None and not self.audio_files and not self.audio_urls:
            warnings.warn(
                "At least one of audio_dir, audio_files, "
                "or audio_urls must be populated for inference.",
                UserWarning,
            )

    def _convert_audios(self, audio_files: List):
        for audio_file in tqdm.tqdm(audio_files):
            output_file = os.path.join(
                self._converted_temp_dir.name, Path(audio_file).stem + ".wav"
            )
            convert_audio_ffmpeg_file_to_file(
                audio_file,
                output_file,
                sampling_rate=self.sampling_rate,
                mono=self.mono,
            )

    def prepare_data(self):
        all_audio_files = []

        # for each audio file in folder, convert with ffmpeg, open with soundfile
        if self.audio_dir:
            for f in os.listdir(self.audio_dir):
                f_path = Path(f)
                if sf.check_format(f_path.suffix.split(".")[-1].upper()):
                    all_audio_files.append(os.path.join(self.audio_dir, f_path))

        if self.audio_files:
            all_audio_files.extend(self.audio_files)

        if self.audio_urls:
            self._download_temp_dir = tempfile.TemporaryDirectory()
            for url in self.audio_urls:
                # Split on the rightmost / and take everything on the right side of that
                name = url.rsplit("/", 1)[-1]

                # Combine the name and the downloads directory to get the local filename
                filename = os.path.join(self._download_temp_dir.name, name)

                # Download the file if it does not exist
                if not os.path.isfile(filename):
                    self._download_file(url, filename)

                all_audio_files.append(filename)

        self._convert_audios(all_audio_files)

    def _download_file(self, url, filename):
        urllib.request.urlretrieve(url, filename)

    def setup(self, stage):
        if "predict" in stage.lower():
            self.predict_dataset = AudioFolderIterableDataset(
                self._converted_temp_dir.name, self.sampling_rate, self.mono
            )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, self.batch_size)

    def cleanup(self):
        if hasattr(self, "_download_temp_dir"):
            self._download_temp_dir.cleanup()
        self._converted_temp_dir.cleanup()
