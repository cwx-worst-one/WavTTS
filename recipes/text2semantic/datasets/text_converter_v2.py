import json
import numpy as np
import argparse
import os
from tqdm import tqdm
from recipes.text2semantic.datasets.sami_tacolabel import generate_tacolabels_from_textstr, enc_taco_label_no_bytes
from zhon.hanzi import punctuation
import string
punctuation_all = punctuation + string.punctuation

# Conver tacolab -> textid
class TextToTacolabID:
    def __init__(self, metaid_to_textid_path, textid_version = 'v2', use_sy = False) -> None:
        with open(metaid_to_textid_path, "r") as f:
            self.text_converter_dict = json.load(f)
            self.textid_version = textid_version
            self.use_sy = use_sy

    def convert_v3_to_v1(self, tacolab):
        tacolab_v1 = []
        if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
            tacolab = tacolab[1:]
        for x in tacolab:
            x_split = x.split('\t')
            if len(x_split) == 7:
                phone, tone, ws, pw, stype, word, _ = x_split
            elif len(x_split) == 6:
                phone, tone, ws, pw, stype, word = x_split
            else:
                print("Wrong tacolab", x_split, tacolab)
                return None
            tacolab_v1.append('\t'.join([phone, tone, '0.0 0.0 0.0 1.0', ws, pw]))
        return tacolab_v1

    def get_lang(self, tacolab):
        if len(tacolab[0].split('\t')) != 5:
            if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                tacolab = tacolab[1:]
        prefix_phn_list = [x.split('\t')[0][:2] for x in tacolab]
        if 'C0' in prefix_phn_list:
            if 'E0' in prefix_phn_list:
                lang = 'zh_en'
            else:
                lang = 'zh'
        else:
            lang = 'en'
        return lang

    def __call__(self, tacolab: list):
        if self.textid_version == 'v1':
            if len(tacolab[0].split('\t')) != 5: # tacolab_version: V3
                tacolab = self.convert_v3_to_v1(tacolab)

            metas = tacolab
            taco_labels=metas
            for i, item in enumerate(taco_labels):
                if item.split("\t")[0] == "：":
                    taco_labels[i]="：\t0\t0.0 0.0 0.0 1.0\tS\t2"
            metas = taco_labels

            metas = enc_taco_label_no_bytes(None, metas, {"use_prsdword": False, "forced_refix": True})
            text_ids =  metas[0].astype(np.int64) * 1_000_000_000 + \
                        metas[1].astype(np.int64) * 1_000_000 + \
                        metas[2].astype(np.int64) * 1_000 + \
                        metas[3].astype(np.int64)
            text_id = np.asarray([self.text_converter_dict.get(str(x), self.text_converter_dict.get("oov")) for x in text_ids]).astype(np.int64)
        elif self.textid_version == 'v2':
            if len(tacolab[0].split('\t')) != 5: # tacolab_version: V3
                tacolab = self.convert_v3_to_v1(tacolab)

            labs = []
            for i in range(len(tacolab)):
                x = tacolab[i]
                if i != 0 and x.split('\t')[0] == 'sil':
                    continue
                phone, tone, _, ws, pw = x.split('\t')
                lab = '_'.join([phone, tone, ws, pw])
                labs.append(lab)
            text_id = []
            for x in labs:
                if self.text_converter_dict.get(x) is None:
                    print("x: ", x)
                cur_text_id = self.text_converter_dict.get(x, self.text_converter_dict.get("oov"))
                text_id.append(cur_text_id)
            text_id = np.asarray(text_id).astype(np.int64)
        elif self.textid_version == 'v3':
            # if len(tacolab[0].split('\t')) != 5:
            #     tacolab = self.convert_v3_to_v1(tacolab)
            lang = self.get_lang(tacolab)
            # print("lang: ", lang)
            if lang in ['zh_en', 'zh']:
                assert len(tacolab[0].split('\t')) == 7, (len(tacolab[0].split('\t')), tacolab[0])
                if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                    tacolab = tacolab[1:]
                labs = []
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, ws, pw, stype, word, unit = x_split
                    lab = '_'.join([phone, tone])
                    if (phone in ['sil', 'sp'] or phone in punctuation_all or phone[0] in punctuation_all):
                        if len(labs) >= 1:
                            if labs[-1] in ["ws", "sy_zh"]:
                                labs[-1] = lab
                        else:
                            labs.append(lab)
                        continue
                    else:
                        labs.append(lab)
                    
                    syllable_flag = False
                    word_segment = False
                    unit = unit.strip()
                    if phone[:2] == 'C0':
                        if unit in ['S', 'E']:
                            if self.use_sy:
                                syllable_flag = True
                            if ws in ['S', 'E']:
                                word_segment = True
                    elif phone[:2] == 'E0':
                        if pw != '0':
                            word_segment = True
                    else:
                        print("Wrong phone", phone)
                        return None

                    if word_segment:
                        labs.append("ws")
                    elif syllable_flag:
                        labs.append("sy_zh")
            elif lang == 'en':
                tacolab = self.convert_v3_to_v1(tacolab)
                labs = []
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, _, ws, pw = x.split('\t')
                    lab = '_'.join([phone, tone])
                    # print("lab: ", lab)
                    # print("phone: ", phone)
                    # print("labs: ", labs)
                    # labs.append(lab)
                    if (phone in ['sil', 'sp'] or phone in punctuation_all or phone[0] in punctuation_all):
                        if len(labs) >= 1:
                            if labs[-1] == "ws":
                                labs[-1] = lab
                        else:
                            labs.append(lab)
                        continue
                    else:
                        labs.append(lab)

                    word_segment = False
                    if pw != '0':
                        word_segment = True

                    if word_segment:
                        labs.append("ws")

            # print("labs: ", labs)
            text_id = []
            for x in labs:
                cur_text_id = self.text_converter_dict.get(x, self.text_converter_dict.get("oov"))
                # if cur_text_id is None:
                #     print("x: ", x)
                #     exit()
                text_id.append(cur_text_id)
            text_id = np.asarray(text_id).astype(np.int64)
        return text_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tacolab_dir", type=str, help="tacolab_dir"
    )
    parser.add_argument(
        "--lab2id_path", type=str, help="lab2id_path"
    )
    parser.add_argument(
        "--textid_dir", type=str, help="textid_dir"
    )
    args = parser.parse_args()

    os.makedirs(args.textid_dir, exist_ok=True)

    text2id = TextToTacolabID(args.lab2id_path)

    tacolab_names = os.listdir(args.tacolab_dir)
    for tacolab_name in tqdm(tacolab_names):
        if tacolab_name[-4:] != '.lab':
            continue
        utt = tacolab_name[:-4]

        tacolab_path = os.path.join(args.tacolab_dir, tacolab_name)
        tacolab = open(tacolab_path).read()
        tacolab = list(filter(lambda x: x != "", tacolab.split('\n')))
        text_id = text2id(tacolab)

        textid_path = os.path.join(args.textid_dir, utt + '.npy')
        np.save(textid_path, text_id)
