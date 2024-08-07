from collections import defaultdict
import librosa
import numpy as np
import os
from scipy.io import wavfile
import sys
from tqdm import tqdm


def concat_wavs(src_paths, tgt_path, sil_interval=0.0):
    sil_samples = np.zeros(int(sil_interval * 24000))
    total_samples = None
    for sp in src_paths:
        temp_samples, _ = librosa.load(sp, sr=24000)
        if total_samples is None:
            total_samples = temp_samples
        else:
            total_samples = np.concatenate([total_samples, sil_samples, temp_samples], axis=-1)
    wavfile.write(tgt_path, rate=24000, data=(total_samples * 32767).astype(np.int16))


def merge_split_wavs(meta_file, src_dir, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    lines = [l.split('|')[0] for l in open(meta_file)]
    utt2segs = defaultdict(list)
    for l in tqdm(lines):
        try:
            assert os.path.exists(f'{src_dir}/{l}.wav'), f'{src_dir}/{l}.wav'
            utt = l[:-10] # xxx_split0001
            utt2segs[utt].append(f'{src_dir}/{l}.wav')
        except:
            print(l)
            continue
    for utt in tqdm(utt2segs):
        concat_wavs(sorted(utt2segs[utt]), f'{tgt_dir}/{utt}.wav')


if __name__ == '__main__':
    meta_file, src_dir, tgt_dir = sys.argv[1:]
    merge_split_wavs(meta_file, src_dir, tgt_dir)
