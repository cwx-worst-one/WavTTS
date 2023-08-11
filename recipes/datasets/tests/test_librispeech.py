import random

import pytest
import torchaudio
from tqdm import tqdm

from recipes.datasets.librispeech import LibriSpeechWebDataModule

@pytest.mark.skip()
def test_librispeech_datamodule():
    batch_size = 64
    sample_rate = 16000
    duration = 10.0
    pl_datamodule = LibriSpeechWebDataModule(
        sample_rate=sample_rate,
        duration=duration,
        batch_size=batch_size,
        shuffle_buffer_size=100,
    )
    train_loader = pl_datamodule.train_dataloader()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        assert "audio" in batch
        if batch_idx > 5:
            break

        audio = batch["audio"]
        assert audio.shape == (batch_size, 1, pl_datamodule.n_audio_samples)
        for a_idx, a in enumerate(audio):
            torchaudio.save(
                f"librispeech-{batch_idx}-{a_idx}-test.mp3",
                a,
                pl_datamodule.sample_rate,
            )
