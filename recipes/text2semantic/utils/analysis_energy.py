import sys, os
import librosa
from scipy.io.wavfile import read
import numpy as np
# import yaml
# sys.path.append('/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_wvae_sft_tobe_merge/recipes/text2semantic/utils/FastSpeech2')
# import audio as Audio
from tqdm import tqdm

in_wav_dir = sys.argv[1]
out_energy_path = sys.argv[2]

# config = yaml.load(open('/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_wvae_sft_tobe_merge/recipes/text2semantic/utils/FastSpeech2/config/LJSpeech_paper/preprocess.yaml', "r"), Loader=yaml.FullLoader)

# STFT = Audio.stft.TacotronSTFT(
#     config["preprocessing"]["stft"]["filter_length"],
#     config["preprocessing"]["stft"]["hop_length"],
#     config["preprocessing"]["stft"]["win_length"],
#     config["preprocessing"]["mel"]["n_mel_channels"],
#     config["preprocessing"]["audio"]["sampling_rate"],
#     config["preprocessing"]["mel"]["mel_fmin"],
#     config["preprocessing"]["mel"]["mel_fmax"],
# )

f_w = open(out_energy_path, 'w')
in_wav_names = os.listdir(in_wav_dir)
for in_wav_name in tqdm(in_wav_names):
    in_wav_path = os.path.join(in_wav_dir, in_wav_name)

    sr, wav = read(in_wav_path)
    if len(wav.shape) == 2 and wav.shape[-1] == 2:
        wav = wav[:, 0]
    wav = wav / 32767.0
    wav *= 1.0 / max(0.01, np.max(np.abs(wav)))
    if sr != 24000:
        wav = librosa.core.resample(wav, sr, 24000)
    # mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)

    n_samples_5s = 2 * 24000
    if wav.shape[0] <= n_samples_5s:
        var = 0
    else:
        max_vols = []
        for i in range(0, int(wav.shape[0] / n_samples_5s)):
            cur_wav = wav[i * n_samples_5s : (i + 1) * n_samples_5s]
            cur_max_vol = np.max(np.abs(cur_wav))
            if cur_max_vol == 0:
                continue
            max_vols.append(cur_max_vol)
        max_vols = np.asarray(max_vols)
        var = np.std(max_vols)
        # new_energy = []
        # for i in range(0, int(energy.shape[0] / 500)):
        #     cur_energy = energy[i * 500 : (i + 1) * 500]
        #     cur_energy_rm0 = cur_energy[cur_energy != 0]
        #     new_energy.append(np.mean(cur_energy_rm0))
        # cur_energy = energy[i * 500 : ]
        # cur_energy_rm0 = cur_energy[cur_energy != 0]
        # new_energy.append(np.mean(cur_energy_rm0))
        # new_energy = np.asarray(new_energy)
        # var = np.std(new_energy)
    f_w.write(in_wav_path + '\t' + str(var) + '\n')
    f_w.flush()
f_w.close()
    # print(in_wav_path, var)
    # print(wav.shape)
    # print(energy.shape)