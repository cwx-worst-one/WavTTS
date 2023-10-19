import sys, os
import numpy as np
import torchaudio
import pyworld
from tqdm import tqdm
import matplotlib.pyplot as plt

in_wav_dir = sys.argv[1]
utt2pitchmeanvar_path = sys.argv[2]

f_w = open(utt2pitchmeanvar_path, 'w')
hop_length = 240

pitch_means = []
pitch_vars = []
pitch_all = None
for in_wav_name in tqdm(os.listdir(in_wav_dir)):
    in_wav_path = os.path.join(in_wav_dir, in_wav_name)
    if in_wav_name[-4:] == '.wav':
        basename = in_wav_name[:-4]
        waveform, sample_rate = torchaudio.load(in_wav_path)
        _waveform = waveform.squeeze(0).double().numpy()
        pitch, t = pyworld.dio(
            _waveform, sample_rate, frame_period=hop_length / sample_rate * 1000
        )
        pitch = pyworld.stonemask(_waveform, pitch, t, sample_rate)

        pitch_rm0 = pitch[pitch!=0]
        pitch_mean = np.mean(pitch_rm0)
        pitch_var = np.var(pitch_rm0)
        
        pitch_means.append(pitch_mean)
        pitch_vars.append(pitch_var)

        f_w.write(basename + '\t' + str(pitch_mean) + '\t' + str(pitch_var) + '\n')
        if pitch_all is None:
            pitch_all = pitch
        else:
            pitch_all = np.concatenate((pitch_all, pitch))
f_w.close()

pitch_all_rm0 = pitch_all[pitch_all!=0]
pitch_mean_all = np.mean(pitch_all_rm0)
print("pitch_mean_all: ", pitch_mean_all)
print("pitch, mean_min: ", min(pitch_means), ", mean_max: ", max(pitch_means), ", mean_mean: ", sum(pitch_means) / len(pitch_means))
print("pitch, var_min: ", min(pitch_vars), ", var_max: ", max(pitch_vars), ", var_mean: ", sum(pitch_vars) / len(pitch_vars))