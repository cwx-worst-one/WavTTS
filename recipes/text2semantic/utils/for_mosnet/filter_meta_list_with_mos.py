import sys, os
from tqdm import tqdm

in_meta_list_path = sys.argv[1]
utt2mos_path = sys.argv[2]
out_meta_list_path = sys.argv[3]

mos_th = 2.8

utt2mos = dict()
lines = open(utt2mos_path).readlines()
for line in tqdm(lines):
    utt, mos = line.strip().split('\t')
    utt2mos[utt] = float(mos)

f_w = open(out_meta_list_path, 'w')
lines = open(in_meta_list_path).readlines()
for line in tqdm(lines):
    wav_id, text_id, metalen = line.strip().split('|')

    utt = wav_id.split('/')[-1][:-4]

    if utt not in utt2mos.keys():
        if "libri_light" not in wav_id:
            f_w.write(line)
        continue

    mos = utt2mos[utt]

    if mos < mos_th:
        continue
    f_w.write(line)
f_w.close()