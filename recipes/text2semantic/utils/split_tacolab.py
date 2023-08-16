from genericpath import exists
import sys, os
from tqdm import tqdm

in_tacolab_dir = sys.argv[1]
out_tacolab_dir = sys.argv[2]

os.makedirs(out_tacolab_dir, exist_ok=True)

head_v3_en = 'phn\ttone\tws\tpwpp\tsentype\tword'
in_tacolab_names = os.listdir(in_tacolab_dir)
for in_tacolab_name in tqdm(in_tacolab_names):
    if in_tacolab_name[-4:] != '.lab':
        continue
    in_tacolab_path = os.path.join(in_tacolab_dir, in_tacolab_name)
    utt_name = in_tacolab_name[:-4]

    f = open(in_tacolab_path)
    lines = f.readlines()
    f.close()
    lines = [x for x in lines if x != ""]

    f_w = None
    utt_index = 0
    for line in tqdm(lines):
        if line.strip() == head_v3_en:
            continue
        phn, tone, ws, pwpp, sentype, word = line.strip('\n').split('\t')
        if phn == 'sil':
            if f_w:
                f_w.close()
                utt_index += 1
            out_tacolab_path = os.path.join(out_tacolab_dir, utt_name + '_' + str(utt_index) + '.lab')
            f_w = open(out_tacolab_path, 'w')
            f_w.write(head_v3_en + '\n')
        f_w.write(line)
    f_w.close()