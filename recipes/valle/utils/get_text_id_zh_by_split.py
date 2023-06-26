import sys, os
# ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
# sys.path.append(os.path.join(ABSPATH, '..'))
# sys.path.append(os.path.join(ABSPATH, '../..'))

import numpy as np
from babble.datasets.building.text.encoding import enc_taco_label_no_bytes
from tqdm import tqdm

f = open('/mnt/bd/huangzhiying-lq-valle-volume5/data/bytegen/valle/text_converter_en-zh_6wh_1wh_1400h_1000h.txt')
lines = f.readlines()
f.close()
text_converter_list = [int(line.strip()) for line in lines]
text_converter = {t: i + 1 for i, t in enumerate(text_converter_list)}

def encode_labs(file_list):
    utts = []
    results = []
    for filename in tqdm(file_list):
        try:
            taco_labels = enc_taco_label_no_bytes(
                os.path.join(filename),
                None,
                {"use_prsdword": False, "forced_refix": True}
            )
            temp = []
            for t in taco_labels[0:4]:
                temp.append(t.astype(np.int64))
            utts.append(filename)
            results.append(temp) # phoneme, tone, word_categ, prosody
        except Exception as e:
            print(f"ERROR: {str(e)}")
    return utts, results


in_text_path = sys.argv[1]
utt2split_path = sys.argv[2]
taco_lab_dir_prefix = sys.argv[3]
text_id_dir_prefix = sys.argv[4]

f = open(utt2split_path)
lines = f.readlines()
f.close()
utt2split = dict()
for line in lines:
    line = line.strip()
    utt, split = line.split(' ')
    utt2split[utt] = split

sucess_labs = []
f = open(in_text_path)
lines = f.readlines()
f.close()
for line in tqdm(lines):
    utt, text = line.strip().split('\t')
    split = utt2split[utt]
    lab_output_dir = os.path.join(taco_lab_dir_prefix, split)
    output_path = os.path.join(lab_output_dir, f'{utt}.lab')
    if not os.path.exists(output_path):
        print(output_path, 'not exists, skip.')
        continue

    sucess_labs.append(os.path.abspath(output_path))

# prepare meta_list
utts, encoded_results = encode_labs(sucess_labs)

metas = []
for i, encoded_text in tqdm(enumerate(encoded_results)):
    utt = utts[i].split('/')[-1][:-4]
    text_id_path = os.path.join(text_id_dir_prefix, utt2split[utt], utt + '.npy')
    if os.path.exists(text_id_path):
        continue

    # x['phones'] * 1_000_000_000 + x['tones'] * 1_000_000 + x['word_categs'] * 1_000 + x['prosodies']
    text_ids = encoded_text[0] * 1_000_000_000 + encoded_text[1] * 1_000_000 + encoded_text[2] * 1_000 + encoded_text[3]
    text_ids_cvt = []
    flag = True
    for x in text_ids:
        if x not in text_converter.keys():
            flag = False
            break
        text_ids_cvt.append(text_converter[x])
    if not flag:
        print("text_id not in text_converter, text_id_path: ", text_id_path)
        continue
    text_ids = np.asarray(text_ids_cvt).astype(np.int64)

    # utt = utts[i].split('/')[-1][:-4]
    os.makedirs(os.path.join(text_id_dir_prefix, utt2split[utt]), exist_ok=True)
    # text_id_path = os.path.join(text_id_dir_prefix, utt2split[utt], utt + '.npy')
    np.save(text_id_path, text_ids)
