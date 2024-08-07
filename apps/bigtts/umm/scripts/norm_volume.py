import sys, os
import librosa
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np

def wav_save_orgamp(path, sr, wav):
    wav = wav * 32767
    wavfile.write(path, sr, wav.astype(np.int16))

in_wav_dir = sys.argv[1]
out_wav_dir = sys.argv[2]

os.makedirs(out_wav_dir, exist_ok=True)

in_wav_names = os.listdir(in_wav_dir)
for in_wav_name in tqdm(in_wav_names):
    if in_wav_name[-4:] == ".wav":
        in_wav_path = os.path.join(in_wav_dir, in_wav_name)
        wav, sr = librosa.load(in_wav_path, sr=None)
        wav = (wav * 1.0) / max(0.01, np.max(np.abs(wav)))
        out_wav_path = os.path.join(out_wav_dir, in_wav_name)
        wav_save_orgamp(out_wav_path, sr, wav)
