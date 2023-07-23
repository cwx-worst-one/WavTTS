import sys, os
import numpy as np
from tqdm import tqdm

in_meta_path = sys.argv[1]
utt2ratio_path = sys.argv[2]

f = open(in_meta_path)
lines = f.readlines()
f.close()

f_w = open(utt2ratio_path, 'w')
for line in tqdm(lines):
    line = line.strip()
    wav_path, text_path, _ = line.split('|')

    # utt = '='.join(wav_path.split('/')[-3:-2] + [wav_path.split('/')[-1][:-4]])
    utt = wav_path.split('/')[-1][:-4]

    if not os.path.exists(wav_path) or not os.path.exists(text_path):
        print(line)
        ratio = 9
    else:
        wav_len = np.load(wav_path).shape[0]
        text_len = np.load(text_path).shape[0]
        ratio = wav_len / text_len

    f_w.write(utt + ' '+ str(ratio) + '\n')
f_w.close()
