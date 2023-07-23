import sys, os
ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(ABSPATH, '..'))
sys.path.append(os.path.join(ABSPATH, '../../'))

import numpy as np
# from babble.datasets import generate_tacolabels_from_text
from babble.datasets.building.text.encoding import enc_taco_label_no_bytes
from tqdm import tqdm
import json

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


fs_lab_dir = sys.argv[1]
text_id_dir = sys.argv[2]
fix_metaid_to_textid_path = sys.argv[3]

with open(fix_metaid_to_textid_path, "r") as f:
    text_converter_dict = json.load(f)

os.makedirs(text_id_dir, exist_ok=True)

sucess_labs = [os.path.join(fs_lab_dir, x) for x in os.listdir(fs_lab_dir)]
utts, encoded_results = encode_labs(sucess_labs)

metas = []
oov_phones = {}
for i, encoded_text in tqdm(enumerate(encoded_results)):
    # x['phones'] * 1_000_000_000 + x['tones'] * 1_000_000 + x['word_categs'] * 1_000 + x['prosodies']
    text_ids = encoded_text[0] * 1_000_000_000 + encoded_text[1] * 1_000_000 + encoded_text[2] * 1_000 + encoded_text[3]
    text_ids_cvt = []
    oov_flag = False
    remove_utt = False
    for index, x in enumerate(text_ids):
        x = str(x)
        if x not in text_converter_dict.keys():
            oov_flag = True
            print(f"{x} not in text_converter_dict, skip this utt.")
            break
            # text_converter_dict[x] = len(text_converter_dict) + 1
            # print(f"{x} not in text_converter_dict, append to dict.")
        if text_converter_dict[x] > 200:
            print(f"{x} not in text_converter_dict, skip this utt")
            remove_utt = True
            break
        text_ids_cvt.append(text_converter_dict[x])

    if oov_flag or remove_utt:
        continue
    text_ids = np.asarray(text_ids_cvt).astype(np.int64)
    utt = utts[i].split('/')[-1][:-4]
    text_id_path = os.path.join(text_id_dir, utt + '.npy')
    np.save(text_id_path, text_ids)
 
# with open(update_metaid_to_textid_path, "w") as f:
#     json.dump(text_converter_dict, f)