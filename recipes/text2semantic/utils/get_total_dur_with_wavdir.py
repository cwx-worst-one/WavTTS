import sys, os
import librosa
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np

def wav_save_orgamp(path, sr, wav):
    wav = wav * 32767
    wavfile.write(path, sr, wav.astype(np.int16))

in_wav_dir = sys.argv[1]
out_dur_path = sys.argv[2]

f_w = open(out_dur_path, 'w')
in_wav_names = os.listdir(in_wav_dir)
for in_wav_name in tqdm(in_wav_names):
    in_wav_path = os.path.join(in_wav_dir, in_wav_name)
    wav, sr = librosa.load(in_wav_path, sr=None)

    dur = len(wav) / sr
    f_w.write(in_wav_path + ' ' + str(dur) + '\n')
f_w.close()

