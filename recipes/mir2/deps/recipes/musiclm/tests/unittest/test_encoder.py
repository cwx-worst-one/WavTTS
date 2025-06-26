import pytest
import torch
import torchaudio
from einops import repeat

from recipes.musiclm.lightning.audio_model import EncodecModel, SoundStreamModel

segment_length_sec = 10
target_sample_rate = 24000


@pytest.mark.parametrize(
    "model",
    [
        EncodecModel(sample_rate=target_sample_rate, target_bandwidth=24.0),
        SoundStreamModel(sample_rate=24000, n_output_frames=800),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_encode_decode(model, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n_samples = target_sample_rate * segment_length_sec

    audio, sr = torchaudio.load("bohemian_rhapsody_excerpt.mp3")
    audio = audio.mean(dim=0, keepdim=True)

    if sr != target_sample_rate:
        resample = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resample(audio)

    audio = audio[:, :n_samples]
    audio = repeat(audio, "... -> b ...", b=batch_size)
    audio = audio.to(device)

    with torch.no_grad():
        quant = model.forward(audio)
        print(audio.shape, quant.shape)
        decoded = model.decode(quant)

    for i in range(batch_size):
        torchaudio.save(f"test_{i}.mp3", decoded[i].cpu(), target_sample_rate)


@pytest.mark.parametrize(
    "model", [SoundStreamModel(sample_rate=24000, n_output_frames=800)]
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.skip()
def test_encode_decode_plot(model, batch_size):
    target_sample_rate = 24000
    n_samples = target_sample_rate * 10

    audio, sr = torchaudio.load("bohemian_rhapsody_excerpt.mp3")
    resample = torchaudio.transforms.Resample(sr, target_sample_rate)
    audio = resample(audio)
    audio = audio[None, :, :n_samples]

    with torch.no_grad():
        quant = model.forward(audio)

    import matplotlib.pyplot as plt

    _, ax = plt.subplots(quant.shape[1], 1, figsize=(10, 10))
    for q_idx in range(quant.shape[1]):
        ax[q_idx].plot(quant[0, q_idx])
        ax[q_idx].set_title(f"Q_{q_idx}")

    plt.tight_layout()
    plt.savefig("quantizer_token_ids_over_time.png")
    plt.close()

    # assert list(quant.shape) == [batch_size, 6, n_output_frames]
    with torch.no_grad():
        decoded = model.decode(quant[:, 2:])

    for i in range(batch_size):
        torchaudio.save(f"test_{i}.mp3", decoded[i], 24000)
