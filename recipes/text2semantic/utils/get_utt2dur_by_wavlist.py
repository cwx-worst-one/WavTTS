import sys, os
import librosa
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np

def wav_save_orgamp(path, sr, wav):
    wav = wav * 32767
    wavfile.write(path, sr, wav.astype(np.int16))

in_wav_list = sys.argv[1]
out_utt2dur_path = sys.argv[2]

f_w = open(out_utt2dur_path, 'w')

f = open(in_wav_list)
lines = f.readlines()
f.close()

total_dur = 0.0
for in_wav_path in tqdm(lines):
    in_wav_path = in_wav_path.strip()
    wav, sr = librosa.load(in_wav_path, sr=None)

    dur = len(wav) / sr
    f_w.write(in_wav_path + ' ' + str(dur) + '\n')

    total_dur += dur

f_w.write("total dur: " + str(round(total_dur / 60 / 60, 3)))
f_w.close()

