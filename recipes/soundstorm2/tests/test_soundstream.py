import torch
import torchaudio
from recipes.soundstorm2.lightning.soundstream import SoundStreamSpeech24k 
from tests.helpers.testing_utils import torch_device


def test_soundstream():
    batch_size = 1
    n_seconds = 10
    soundstream = SoundStreamSpeech24k().to(torch_device)
    n_samples = n_seconds * soundstream.sample_rate

    audio = torch.randn(batch_size, 1, n_samples, device=torch_device)
    tokens = soundstream(audio)

    assert tokens.shape == (batch_size, soundstream.n_quantizers, soundstream.n_frames(n_seconds))

def test_soundstream_qa():
    batch_size = 8
    audio, sr = torchaudio.load("1188-133604-0017_gt.wav")
    audio = audio.to(torch_device)
    audio = audio.unsqueeze(dim=0).repeat(batch_size, 1, 1)

    soundstream = SoundStreamSpeech24k().to(torch_device)
    assert sr == soundstream.sample_rate

    tokens = soundstream(audio)

    decoded_audio = soundstream.decode(tokens)

    torchaudio.save("1188-133604-0017_rec.wav", decoded_audio[0].cpu(), soundstream.sample_rate)
