import numpy as np
import random
import sys
from jiwer import compute_measures

in_file=sys.argv[1]

qujian = []
wers = []
text_lens = []

sub_list = []
del_list = []
ins_list = []

q1, q2, q3, q4, q5, q6, q7, q8, q9 = [], [], [], [], [], [], [], [], []
title = [">0.7", "0.7-0.6", "0.6-0.5", "0.5-0.4", "0.4-0.3", "0.3-0.2", "0.2-0.1", "0.1-0.0", "0.0"]


for line in open(in_file, "r").readlines()[1:-3]: 
    # print(line.strip().split("\t"))

    if len(line.strip().split("\t")) == 6:
        utt, wav_ref, wav_res, wer, ref, text_res = line.strip().split("\t")
    else:
        utt, wav_res, wer, ref, text_res = line.strip().split("\t")

    puncs = ",.?!:\'\"[]-<>~|$&*%@()"
    res = text_res
    for x in puncs:
        ref = ref.replace(x, '')
        res = res.replace(x, '')

    # split to list
    ref_list = ref.split(' ')
    measures = compute_measures(ref, res)
    # wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    sub_list.append(subs)
    del_list.append(dele)
    ins_list.append(inse)

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

# for i, item in enumerate([q1, q2, q3, q4, q5, q6, q7, q8, q9]):
#     print(f"q{i+1}\t{title[i]}\t{len(item)}")
#     if len(item) > 30:
#         for line in random.sample(item, 30):
#             print(f"{line[1]}\t{line[0]}")
#     else:
#         for line in item:
#             print(f"{line[1]}\t{line[0]}")

wers = np.array(wers)
title_str="\t".join(title)
value_str=[]
bars=[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
for i, item in enumerate([q1, q2, q3, q4, q5, q6, q7, q8, q9]):
    if i < len(bars):
        bar=bars[i]
        if i == len(bars) - 1:
            wer = 0
        else:
            wer = np.mean(wers[wers<bar])
        value_str.append(f"{len(item)} ({round(wer*100.0,2)})%")
    else:
        value_str.append(f"{len(item)}")

# print(title_str)
# print("\t".join(value_str))

wer = round(np.mean(wers)*100,3)
# print("IDS")
avg_ins = round(np.mean(np.array(ins_list))*100,3)
ins_ratio = round(avg_ins / wer, 3)
avg_del = round(np.mean(np.array(del_list))*100,3)
del_ratio = round(avg_del / wer, 3)
avg_sub = round(np.mean(np.array(sub_list))*100,3)
sub_ration = round(avg_sub / wer, 3)
print(f"{avg_ins}%({ins_ratio})|{avg_del}%({del_ratio})|{avg_sub}%({sub_ration})")

