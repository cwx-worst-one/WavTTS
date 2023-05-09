import pytest
import torch
import torchaudio

try:
    from recipes.audio_diffusion.modules.transforms.encoder import SoundStreamTransform
except Exception as e:
    print(e)


@pytest.mark.skip("Only run on Merlin for model verification")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_encode_decode(batch_size):
    model = SoundStreamTransform(sample_rate=24000, n_output_frames=800)
    target_sample_rate = 24000
    n_samples = target_sample_rate * 10
    n_output_frames = n_samples // 300

    waveform = torch.randn(batch_size, 1, n_samples)
    sr = 24000

    resample = torchaudio.transforms.Resample(sr, target_sample_rate)

    waveform = resample(waveform)
    waveform = waveform[..., None, :n_samples]

    waveform = waveform.repeat(batch_size, 1, 1)

    with torch.no_grad():
        quant = model.forward(waveform)

    assert list(quant.shape) == [batch_size, 6, n_output_frames]
    decoded = model.decode(quant)

    for i in range(batch_size):
        torchaudio.save(f"test_{i}.mp3", decoded[i], 24000)
