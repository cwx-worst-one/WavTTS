import torch
import pyloudnorm as pyln

class AudioStats:

    def __init__(self, sample_rate: int, channels: int, duration: float, loudness_db_lufs: float):
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration
        self.loudness_db_lufs = loudness_db_lufs
    
    def to_dict(self) -> dict:
        return self.__dict__

    @staticmethod
    def measure(audio: torch.Tensor, sample_rate: int):
        assert audio.ndim == 2  # [channels. time]

        channels = audio.shape[0]
        duration = audio.shape[1] / sample_rate

        meter = pyln.Meter(sample_rate)
        loudness_db_lufs = meter.integrated_loudness(audio.transpose(0, 1).cpu().numpy())
        return AudioStats(
            sample_rate=sample_rate,
            channels=channels,
            duration=duration,
            loudness_db_lufs=loudness_db_lufs,
        )