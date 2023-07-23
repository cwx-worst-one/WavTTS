import os
import librosa
from tqdm import tqdm

import torch
import numpy as np
from scipy.io.wavfile import write
import sys
import hdfs_helper as hh
import time

MAX_WAV_VALUE = 32768.0

def save_wav(audio, output_file, sr=24000):
    audio = audio * MAX_WAV_VALUE
    audio = audio.astype('int16')
    write(output_file, sr, audio)

def save_wav_int16(audio, output_file, sr=24000):
    write(output_file, sr, audio)
    return

def trim_silence(wav, trim_top_db=30, trim_fft_size=512, trim_hop_size=128, num_silent_frames=4, hop_size=300):
    '''Trim leading and trailing silence'''
    # These params are separate and tunable per dataset.
    unused_trimed, index = librosa.effects.trim(
        wav,
        top_db=trim_top_db,
        frame_length=trim_fft_size,
        hop_length=trim_hop_size)
    num_sil_samples = int(num_silent_frames * hop_size)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - int(num_sil_samples // 2), 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + num_sil_samples * 2, len(wav))
    trimmed_wav = wav[start_idx:stop_idx]
    return trimmed_wav

def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device

def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module

def init_sound_stream_encoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_encoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_encoder_{local_rank}.pt"

    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_path):
            if not hh.get(h_ss, local_path):
                raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        return {"ss_enc": load_torch_script_module(local_path, device)}
    else:
        return {"ss_enc": load_torch_script_module(h_ss, device)}

def init_sound_stream_decoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_decoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_decoder_{local_rank}.pt"

    if not os.path.exists(local_path):
        if not hh.get(h_ss, local_path):
            raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
    return {"ss_dec": load_torch_script_module(local_path, device)}

if __name__ == '__main__':
    ss_export_dir = sys.argv[1]
    wav_list_path = sys.argv[2]
    local_rank = int(sys.argv[3])
    out_dir_prefix = sys.argv[4]
    wav2split_path = None
    if len(sys.argv) == 6:
        wav2split_path = sys.argv[5]

    sample_rate = 24000

    time_now = int(time.time())
    vqgan_model_encoder = init_sound_stream_encoder(ss_export_dir, local_rank, cache_dir="/tmp/.cache_ss_dir" + str(time_now))["ss_enc"]
    vqgan_model_encoder.eval()

    vqgan_model_decoder = init_sound_stream_decoder(ss_export_dir, local_rank, cache_dir="/tmp/.cache_ss_dir" + str(time_now))["ss_dec"]
    vqgan_model_decoder.eval()

    wav2split = None
    if wav2split_path is not None:
        f = open(wav2split_path)
        lines = f.readlines()
        f.close()
        wav2split = dict()
        for line in lines:
            line = line.strip()
            wav_path, split = line.split(' ')
            wav2split[wav_path] = split

    f = open(wav_list_path)
    lines = f.readlines()
    f.close()
    for line in tqdm(lines):
        wav_path = line.strip()
        uttname = wav_path.split('/')[-1][:-4]
        if wav2split_path is not None:
            out_ss_dir = os.path.join(out_dir_prefix, wav2split[wav_path], 'wav_id')
        else:
            out_ss_dir = os.path.join(out_dir_prefix, 'wav_id')
        
        os.makedirs(out_ss_dir, exist_ok=True)
        out_ss_path = os.path.join(out_ss_dir, uttname + '.npy')
        if os.path.exists(out_ss_path):
            continue

        # load and resample
        wav, sr = librosa.load(wav_path, sr=None)
        # out_wav_ori_path = os.path.join(out_ss_dir, uttname + '_ori.wav')
        # save_wav(wav, out_wav_ori_path)

        if sr != sample_rate:
            wav = librosa.core.resample(wav, sr, sample_rate)

        # out_wav_24k_path = os.path.join(out_ss_dir, uttname + '_24k.wav')
        # save_wav(wav, out_wav_24k_path)

        # norm 1.0
        wav = wav * 1.0 / max(0.01, np.max(np.abs(wav)))
        # out_wav_24k_norm_path = os.path.join(out_ss_dir, uttname + '_24k_norm1.0.wav')
        # save_wav(wav, out_wav_24k_norm_path)

        # trim sil
        wav = trim_silence(wav)
        # out_wav_24k_norm_trim_path = os.path.join(out_ss_dir, uttname + '_24k_norm1.0_trim.wav')
        # save_wav(wav, out_wav_24k_norm_trim_path)

        # gpu
        wav = torch.from_numpy(wav).float().unsqueeze(0).to("cuda:" + str(local_rank)) # [1, T]
        # print("wav: ", wav.shape)

        # extract wav_id
        wav_id = torch.stack(vqgan_model_encoder(wav)[2], dim=2)
        # print("wav_id: ", wav_id.shape)
        np.save(out_ss_path, wav_id[0, :, :].detach().cpu().numpy()) # [t, n_codebook]


        # # reconstruction
        # wav_rec = vqgan_model_decoder(wav_id.transpose(1, 2))
        # out_wav_24k_norm_trim_rec_path = os.path.join(out_ss_dir, uttname + '_24k_norm1.0_trim_rec.wav')
        # save_wav(wav_rec.detach().cpu().numpy(), out_wav_24k_norm_trim_rec_path)