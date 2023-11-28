import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from recipes.waveformvae.models.waveformvae import WaveformVAE
from recipes.waveformvae.utils.audio_utils import mel_spectrogram_torch, spectrogram_torch
from scipy.io.wavfile import write, read
from torch.nn import functional as F
from tqdm import tqdm
from collections import OrderedDict


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint_dict['state_dict']
    saved_state_dict = OrderedDict()
    for k, v in state_dict.items():
        parts = k.split('.')
        parts = parts[1:]
        new_k = '.'.join(parts)
        if new_k in model.state_dict():
            saved_state_dict[new_k] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(saved_state_dict)
    else:
        model.load_state_dict(saved_state_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', "-c", type=str)
    parser.add_argument('--wave_path', "-w", type=str)
    parser.add_argument('--save_path', "-s", type=str)
    args = parser.parse_args()

    model = WaveformVAE(
        sd_channels=32,
        spec_channels=2048 // 2 + 1,
        hidden_channels=256,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        downsample_rates=[2, 2, 2, 3, 5, 5],
        downsample_initial_channel=16,
        segment_size=24000
    )
    model = load_checkpoint(args.checkpoint_path, model)
    model = model.cuda()
    model.eval()
    print("Model loaded.")

    # 载入参考prosody的音频和文本
    path_to_recon = os.path.join(args.wave_path)
    recon_audios = {}
    recon_specs = {}
    recon_mels = {}
    path_to_recons = os.listdir(path_to_recon)
    for i in tqdm(range(len(path_to_recons))):
        filename = path_to_recons[i]
        try:
            path = os.path.join(path_to_recon, filename)
            if path.endswith(".wav"):
                sr, wav = read(path)
                wav = wav / 32767.0
                wav *= (1.0 * 0.95) / max(0.01, np.max(np.abs(wav)))
                wav = librosa.resample(wav, sr, 24000)
                wav, _ = librosa.effects.trim(wav)
                wav = torch.stack([torch.from_numpy(wav)]).unsqueeze(1).float().cuda()
                wav = F.pad(wav, (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1), 0, 0, 0, 0), value=0.)
                recon_audios.update({filename[:-4]: wav})
                spec = spectrogram_torch(wav.squeeze(1), 2048, 24000, 300, 1200).cuda()
                mel = mel_spectrogram_torch(wav.squeeze(1), 2048, 80, 24000, 300, 1200, 0.0, None).cuda()
                recon_specs.update({filename[:-4]: spec})
                recon_mels.update({filename[:-4]: mel})
        except Exception as e:
            pass

    cnt = 1
    os.makedirs(args.save_path, exist_ok=True)
    sid = 1
    with torch.no_grad():
        key_list = list(recon_audios.keys())
        for i in tqdm(range(len(key_list))):
            filename = key_list[i]
            wav = recon_audios[filename]
            spec = recon_specs[filename]
            mel = recon_mels[filename]
            o = model.infer(wav, spec, mel)
            audio = o[0][0].cpu().numpy()
            audio *= (32768.0 - 1) / max(0.01, np.max(np.abs(audio)))
            write(os.path.join(args.save_path, f"{filename}_recon.wav"), 24000, audio.astype(np.int16))

            wav = wav[0][0].cpu().numpy()
            wav *= (32768.0 - 1) / max(0.01, np.max(np.abs(wav)))
            write(os.path.join(args.save_path, f"{filename}.wav"), 24000, wav.astype(np.int16))
            cnt += 1
