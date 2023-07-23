import sys, os
import numpy as np
from tqdm import tqdm

in_meta_path = sys.argv[1]
utt2len_path = sys.argv[2]
out_meta_path = sys.argv[3]
n_positions = 2048
n_eos = 2
max_len = 2045

f = open(utt2len_path)
lines = f.readlines()
f.close()
utt2len = dict()
for line in lines:
    line = line.strip()
    utt, split = line.split(' ')
    utt2len[utt] = split

f = open(in_meta_path)
lines = f.readlines()
f.close()

f_w = open(out_meta_path, 'w')
for line in tqdm(lines):
    line = line.strip()
    wav_path, text_path = line.split('|')

    if not os.path.exists(wav_path) or not os.path.exists(text_path):
        continue

    # wav_len = np.load(wav_path).shape[0]
    # text_len = np.load(text_path).shape[0]

    utt = wav_path.split('/')[-1][:-4]
    cur_len = int(utt2len[utt])
    if cur_len > max_len:
        continue
    # if wav_len + text_len > max_len:#n_positions - n_eos:
    #     continue
    
    f_w.write(line + '\n')
f_w.close()
