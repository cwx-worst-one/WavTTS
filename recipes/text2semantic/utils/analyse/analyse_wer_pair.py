import numpy as np
import random
import sys
import os


part1_name=sys.argv[1]
part1_file=sys.argv[2]
part2_name=sys.argv[3]
part2_file=sys.argv[4]

out_dir=sys.argv[5]

wers = []
text_lens = []

output_num=20

def get_meta(in_file):
    q1, q2, q3, q4, q5, q6, q7, q8, q9 = [], [], [], [], [], [], [], [], []
    title = [">0.7", "0.7-0.6", "0.6-0.5", "0.5-0.4", "0.4-0.3", "0.3-0.2", "0.2-0.1", "0.1-0.0", "0.0"]
    utt2meta = {}
    for line in open(in_file, "r").readlines()[1:-3]: 
        if len(line.strip().split("\t")) == 6:
            utt, wav_ref, wav_res, wer, ref, text_res = line.strip().split("\t")
        else:
            utt, wav_res, wer, ref, text_res = line.strip().split("\t")
        wer = float(wer)
        
        if wer > 0.7:
            q1.append((line.strip(), wer))
        elif wer > 0.6:
            q2.append((line.strip(), wer))
        elif wer > 0.5:
            q3.append((line.strip(), wer))
        elif wer > 0.4:
            q4.append((line.strip(), wer))
        elif wer > 0.3:
            q5.append((line.strip(), wer))
        elif wer > 0.2:
            q6.append((line.strip(), wer))
        elif wer > 0.1:
            q7.append((line.strip(),wer))
        elif wer > 0.0:
            q8.append((line.strip(),wer))
        else:
            q9.append((line.strip(),wer))

        wers.append(wer)
        text_lens.append(float(len(ref.split(" "))))
        utt2meta[utt] = (text_res, ref, wav_res, wer)
    return [q1, q2, q3, q4, q5, q6, q7, q8, q9], utt2meta

part1_q, part1_utt2meta = get_meta(part1_file)
part2_q, part2_utt2meta = get_meta(part2_file)

# utt | text | part_1_wer | part_1_text | part_2_wer | part_2_text \n
count=0
os.makedirs(out_dir, exist_ok=True)
part1_out_wavdir = os.path.join(out_dir, part1_name)
part2_out_wavdir = os.path.join(out_dir, part2_name)
os.makedirs(part1_out_wavdir, exist_ok=True)
os.makedirs(part2_out_wavdir, exist_ok=True)

# print("uttname/wer\ttext")
for i, item in enumerate(part1_q):
    for line in item:
        utt = line[0].split("\t")[0]
        count = count + 1
        if count > output_num:
            break
        part1_wav_path = part1_utt2meta[utt][2]
        part2_wav_path = part2_utt2meta[utt][2]

        part1_out_wav_path = os.path.join(part1_out_wavdir, f"{count:02}-{utt}.wav")
        part2_out_wav_path = os.path.join(part2_out_wavdir, f"{count:02}-{utt}.wav")

        os.system(f"cp {part1_wav_path} {part1_out_wav_path}")
        os.system(f"cp {part2_wav_path} {part2_out_wav_path}")

        # print(f"{utt}")
        # print(f"text_ref\t{part1_utt2meta[utt][1]}")
        # print(f"{part1_utt2meta[utt][3]}\t{part1_utt2meta[utt][0]}")
        # print(f"{part2_utt2meta[utt][3]}\t{part2_utt2meta[utt][0]}\n")

        print(f"{utt}")
        print("%3f\t%s" % (1.0, part1_utt2meta[utt][1]))
        print("%3f\t%s" % (part1_utt2meta[utt][3], part1_utt2meta[utt][0]))
        print("%3f\t%s" % (part2_utt2meta[utt][3], part2_utt2meta[utt][0]))
