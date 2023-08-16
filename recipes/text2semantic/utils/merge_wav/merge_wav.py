import sys, os
from scipy.io import wavfile
import numpy as np
import librosa
from tqdm import tqdm

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

def wav_save_orgamp(path, sr, wav):
    wav = wav * 32767
    wavfile.write(path, sr, wav.astype(np.int16))

in_wav_dir = sys.argv[1]
out_wav_dir = sys.argv[2]

os.makedirs(out_wav_dir, exist_ok=True)

sil_600ms = np.zeros(7200)
in_wav_names = os.listdir(in_wav_dir)

utt2wav_paths = dict()
for in_wav_name in in_wav_names:
     utt = '_'.join(in_wav_name.split('_')[:-1])
     if utt not in utt2wav_paths.keys():
          utt2wav_paths[utt] = []
     if in_wav_name[-4:] == '.wav':
          utt2wav_paths[utt].append(os.path.join(in_wav_dir, in_wav_name))

# print("utt2wav_paths: ", utt2wav_paths['long_emotion_0011'], len(utt2wav_paths['long_emotion_0011']))
for utt in tqdm(utt2wav_paths.keys()):
     wav_paths = utt2wav_paths[utt]
     id2wav_path = dict()
     for wav_path in wav_paths:
          id = wav_path.split('/')[-1].split('.')[0].split('_')[-1]
          id2wav_path[int(id)] = wav_path
     id2wav_path_sorted = dict(sorted(id2wav_path.items()))
     # if utt == 'long_emotion_0011':
     #      print("id2wav_path_sorted: ", id2wav_path_sorted)
     wav_paths = id2wav_path_sorted.values()
     # if utt == 'long_emotion_0011':
     #      print("wav_paths: ", wav_paths)

     # wav_paths.sort()

     final_wav = None
     sr = None
     
     for wav_path in wav_paths:
          wav, sr = librosa.load(wav_path, sr=None)
          # wav = trim_silence(wav)

          # out_wav_path = os.path.join(out_wav_dir, wav_path.split('/')[-1])
          # wav_save_orgamp(out_wav_path, sr, wav)

          if final_wav is None:
               final_wav = wav
          else:
               final_wav = np.concatenate((final_wav, sil_600ms, wav), axis=0)

     out_wav_path = os.path.join(out_wav_dir, utt + '.wav')
     wav_save_orgamp(out_wav_path, sr, final_wav)
