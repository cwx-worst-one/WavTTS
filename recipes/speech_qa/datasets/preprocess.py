import logging
from abc import abstractmethod
from typing import Dict, Generator
import torch
import torch.nn as nn
import io
import numpy as np
from pydub import AudioSegment
import librosa


logger = logging.getLogger(__name__)


def get_active_frames(audio, threshold=0.05, sample_rate=24000):
    window_size = int(sample_rate * 0.1)

    frames = librosa.util.frame(
        x=audio, frame_length=window_size, hop_length=window_size
    ).T
    energy = np.max(np.abs(frames), axis=-1)  # shape: (frames_num,)
    rate = np.sum(energy > threshold) / energy.shape[0]

    if rate < 1 / 10:
        return False
    return True


class BufferPreprocessorBase:
    def __init__(self, transforms: nn.Module) -> nn.Module:
        self.transforms = transforms
        logger.info(f"Preprocessor transforms:\n{self.transforms}")

    @abstractmethod
    def train_buffer_preprocessor(self, batch: Generator) -> Generator:
        pass


class WebDatasetBufferPreprocessor(BufferPreprocessorBase):
    def __init__(self, transforms: Dict[str, nn.Module]):
        super().__init__(transforms=transforms)
        self.count = 0
        self.skipped = 0

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            self.count += 1
            item = self.transforms(item)
            if item.get("__skip__", False):
                self.skipped += 1
                if self.skipped % 100 == 0:
                    print(f"Skipped {self.skipped}/{self.count} items")
                continue
            yield item


class AudioTransforms(nn.Module):
    def __init__(
        self,
        durations: int = 10,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.durations = durations
        self.sample_rate = sample_rate
        self.segment_size = self.durations * self.sample_rate

    def forward(self, item) -> Dict[str, torch.Tensor]:
        output = {}

        clip_duration = float(item["metadata.json"]["clip_duration"])
        if not ((clip_duration >= (self.durations - 0.05) * 1000) and (clip_duration <= 360 * 1000)):
            output["__skip__"] = True
            return output

        audio = AudioSegment.from_file(io.BytesIO(item["mp3"]), format="mp3")
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(self.sample_rate)
        wav = np.asarray(audio.get_array_of_samples())
        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2_147_483_648.0
        wav = wav / np.max(np.abs(wav)) * 0.95

        wav_len = wav.shape[0]
        if wav_len < self.segment_size - 0.05 * self.sample_rate:
            output["__skip__"] = True
            return output
        if wav_len < self.segment_size:
            wav = np.pad(wav, ((0, 0), (0, self.segment_size - wav_len)))
        else:
            beg = np.random.randint(low=0, high=wav_len - self.segment_size + 1)
            wav = wav[beg : beg + self.segment_size]
        if np.sqrt(np.mean(wav**2)) <= 1e-4 and get_active_frames(
            wav, threshold=0.05, sample_rate=self.sample_rate
        ):
            output["__skip__"] = True
            return output
        output["wav"] = torch.from_numpy(wav)
        return output
