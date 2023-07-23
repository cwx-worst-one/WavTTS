import torch
import numpy as np
import os
import sys
from scipy.io.wavfile import write
import hdfs_helper as hh
from tqdm import tqdm

np.set_printoptions(threshold=1000000)

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

def save_wav_int16(audio, output_file, sr=24000):
    write(output_file, sr, audio)
    return


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

def init_sound_stream(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_encoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_encoder_{local_rank}.pt"

    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_path):
            if not hh.get(h_ss, local_path):
                raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        return {"ss": load_torch_script_module(local_path, device)}
    else:
        return {"ss": load_torch_script_module(h_ss, device)}


if __name__ == "__main__":
    local_rank = 0
    vqgan_model_encoder = init_sound_stream_encoder("hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export", local_rank, cache_dir=".cache_dir")["ss_enc"]
    vqgan_model_encoder.eval()
    vqgan_model_decoder = init_sound_stream_decoder("hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export", local_rank, cache_dir=".cache_dir")["ss_dec"]
    vqgan_model_decoder.eval()

    in_wav_id_dir = sys.argv[1]
    out_wav_dir = sys.argv[2]

    os.makedirs(out_wav_dir, exist_ok=True)

    wav_id_names = os.listdir(in_wav_id_dir)
    for wav_id_name in tqdm(wav_id_names):
        wav_id_path = os.path.join(in_wav_id_dir, wav_id_name)
        wav_id = np.load(wav_id_path)
        wav_id = torch.from_numpy(wav_id).unsqueeze(0).transpose(1, 2)

        wav = vqgan_model_decoder(wav_id.to('cuda:' + str(local_rank)))
        wav = wav.detach().cpu().squeeze(1).squeeze(0).numpy()

        save_wav(wav, os.path.join(out_wav_dir, wav_id_name[:-4] + '.wav'))
