from operator import mod
from re import M
import torch
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from recipes.waveformvae.utils.audio_utils import mel_spectrogram_torch, spectrogram_torch
from scipy.io.wavfile import write
from scipy.io.wavfile import write, read
from recipes.waveformvae.models.waveformvae import VIFSpeechTrn_Tacolabel, MelGPT, WaveformVAE
from recipes.waveformvae.utils.utils import load_checkpoint, get_hparams
# from text import symbols
from torch.nn import functional as F
from tqdm import tqdm
import sys

import ruamel.yaml
import argparse

def load_yaml(path):
    yaml = ruamel.yaml.YAML()
    with open(path, "r") as file:
        data = yaml.load(file)
    return data

def load_ckpt(ckpt_path, model):
    ## ckpt state_dict
    saved_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    saved_state_dict = {k.replace("generator.",""):v for k, v in saved_state_dict.items()}

    ## model state_dict
    state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v    

    ## load state_dict
    model.load_state_dict(new_state_dict)
    print("Loading state_dict from {} successfully.".format(ckpt_path))
    return model

def load_model(ckpt_path, hps):
    model = WaveformVAE(
        sd_channels=32,
        spec_channels=hps['data_cls'].get('filter_length') // 2 + 1,
        hidden_channels=256,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        downsample_rates=[2, 2, 2, 3, 5, 5],
        downsample_initial_channel=16,
        segment_size=12000,
    )
    model = load_ckpt(ckpt_path, model)
    model = model.cuda()
    model.eval()
    print("Model loaded.")
    return model

@torch.no_grad()
def main(args):
    hps = load_yaml(args.config_path)
    model = load_model(args.ckpt_path, hps)

    path_to_recon = args.input_dir

    recon_audios = {}
    recon_specs = {}
    for filename in tqdm(os.listdir(path_to_recon)):
        path = os.path.join(path_to_recon, filename)
        sr, wav = read(path)
        wav = wav / 32767.0
        wav *= 1.0 / max(0.01, np.max(np.abs(wav)))
        wav = librosa.resample(wav, sr, hps['data_cls'].get('sampling_rate'))
        wav, _ = librosa.effects.trim(wav)
        wav = torch.stack([torch.from_numpy(wav)]).unsqueeze(1).float().cuda()
        wav = F.pad(wav, (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1), 0, 0, 0, 0), value=0.)
        recon_audios.update({filename[:-4]: wav})
        spec = spectrogram_torch(wav.squeeze(1),
                                 hps['data_cls'].get('filter_length'),
                                 hps['data_cls'].get('sampling_rate'),
                                 hps['data_cls'].get('hop_length'),
                                 hps['data_cls'].get('win_length')).cuda()
        recon_specs.update({filename[:-4]: spec})

    cnt = 1
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'/recon_wavs', exist_ok=True)
    os.makedirs(save_dir+'/gt_wavs', exist_ok=True)
    sid = 1
    with torch.no_grad():
        key_list = list(recon_audios.keys())
        for i in tqdm(range(len(key_list))):
            filename = key_list[i]
            wav = recon_audios[filename]
            spec = recon_specs[filename]
            o = model.infer(wav, spec)
            audio = o[0][0].cpu().numpy()
            audio *= (32768.0 - 1) / max(0.01, np.max(np.abs(audio)))
            write(os.path.join(save_dir, 'recon_wavs', f"{filename}_recon.wav"), hps['data_cls'].get('sampling_rate'), audio.astype(np.int16))
            #wav = wav[0][0].cpu().numpy()
            #wav *= (32768.0 - 1) / max(0.01, np.max(np.abs(wav)))
            #write(os.path.join(save_dir, 'gt_wavs', f"{filename}.wav"), hps['data_cls'].get('sampling_rate'), wav.astype(np.int16))
            cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True, help="text file")
    parser.add_argument("--output_dir", type=str, required=True, help="text file")
    args = parser.parse_args()
    main(args)

# python3 recipes/waveformvae/scripts/test.py \
#    --config_path aaa \
#    --ckpt_path bbb \
#    --input_dir ccc \
#    --output_dir ddd

