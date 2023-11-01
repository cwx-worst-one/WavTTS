"""thanks to @liuyang.1314"""
import warnings

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from scripts.data_processing.audio.stft import ISTFT, STFT

warnings.simplefilter(action="ignore", category=FutureWarning)
import soundfile as sf


@torch.no_grad()
def process_batch(filename, sr, device, max_duration):
    model = torch.jit.load("model_best.pt").eval()
    model = model.to(device)

    stft = (
        STFT(
            n_fft=512,
            hop_length=256,
            win_length=512,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        .to(device)
        .eval()
    )

    inverse_stft = (
        ISTFT(
            n_fft=512,
            hop_length=256,
            win_length=512,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        .to(device)
        .eval()
    )

    batch_wavs = [librosa.load(filename, sr=None, mono=True)[0]]
    length = [wav.shape[0] for wav in batch_wavs]
    max_length = max(max_duration * sr, max(length))
    batch_wavs = [
        np.pad(wav, pad_width=(0, max_length - length[i])).reshape((1, -1))
        for i, wav in enumerate(batch_wavs)
    ]
    batch_wavs = np.vstack(batch_wavs)
    batch_wavs = torch.from_numpy(batch_wavs).to(device)

    max_value = torch.max(batch_wavs, dim=1)[0][:, None]
    min_value = torch.min(batch_wavs, dim=1)[0][:, None]
    threhold = torch.where(
        torch.logical_or(max_value > 0.85, min_value < -0.85),
        torch.ones_like(max_value) * 0.5,
        torch.ones_like(max_value),
    )
    low_waveform = batch_wavs * threhold

    audio_length = low_waveform.shape[-1]
    (low_real, low_imag) = stft(low_waveform)
    result = model(low_real, low_imag)
    audio_data = inverse_stft(result[0], result[1], audio_length).cpu().numpy()
    print(max(abs(audio_data[0])))
    audios = [audio[:ilen] for ilen, audio in zip(length, audio_data)]
    write("0.8.1.wav", sr, (audios[0] * 32767).astype(np.int16))

    # write("ori_float.wav", sr, audios[0])


process_batch("ori.wav", 24000, "cpu", 10)
