import librosa
import torch
from resemblyzer import VoiceEncoder, hparams, normalize_volume, trim_long_silences


def preprocess_audio(audio_bin, *_, **__):
    wav, sr = librosa.load(audio_bin, sr=hparams.sampling_rate)
    if wav.size == 0:
        return None, None
    audio_dur = wav.shape[0] / float(sr)
    wav = normalize_volume(wav, -30, increase_only=True)
    wav = trim_long_silences(wav)
    return wav, audio_dur


@torch.no_grad()
def process_batch(model, batch, *_, **__):
    for wav in batch:
        embed = model.embed_utterance(wav)
        yield embed


def load_model(device, *_, **__):
    model = VoiceEncoder(device=device).eval()
    return model


# spk_emb
