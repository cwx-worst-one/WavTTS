import sys, os
import numpy as np
from tqdm import tqdm

in_meta_path = sys.argv[1]
out_meta_path = sys.argv[2]
min_ratio = int(sys.argv[3])
max_ratio = int(sys.argv[4])

f = open(in_meta_path)
lines = f.readlines()
f.close()

f_w = open(out_meta_path, 'w')
for line in tqdm(lines):
    line = line.strip()
    wav_path, text_path, metalen, ratio = line.split('|')

    if float(ratio) < min_ratio or float(ratio) > max_ratio:
        continue

    f_w.write(line + '\n')
f_w.close()
