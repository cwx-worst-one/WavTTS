import json
import numpy as np
import time
from recipes.text2semantic.datasets.sami_tacolabel import generate_tacolabels_from_textstr, enc_taco_label_no_bytes

# Conver raw text -> tacolab -> metaids
class TextToTacolabID:
    def __init__(self, metaid_to_textid_path, lab='old', symbol_sets=None) -> None:
        with open(metaid_to_textid_path, "r") as f:
            self.text_converter_dict = json.load(f)
        self.lab = lab
        self.symbol_sets = symbol_sets

    def convert_new2old(self, metas):
        new_metas = []
        for meta in metas:
            # wordpost
            phone, tone, wordcateg, prosody, _, _ = meta.split('\t')
            if phone not in self.symbol_sets and prosody not in ['0', '1']:
                prosody = '1'
            wordpost = "0.0 0.0 0.0 0.0"
            new_meta = '\t'.join([phone, tone, wordpost, wordcateg, prosody])
            new_metas.append(new_meta)
        return new_metas

    def __call__(self, text: str):
        if self.lab == 'old':
            taco_labels = generate_tacolabels_from_textstr(text, "English")
            while taco_labels is None:
                time.sleep(1)
                taco_labels = generate_tacolabels_from_textstr(text, "English")
        elif self.lab == 'new':
            taco_labels = generate_tacolabels_from_textstr(text, "English_new")
            while taco_labels is None:
                time.sleep(1)
                taco_labels = generate_tacolabels_from_textstr(text, "English_new")
        if self.lab == 'old':
            metas = taco_labels.decode().split("\n")[:-1] # metas[-1] is "".
        elif self.lab == 'new':
            metas = taco_labels.decode().split("\n")

        if self.lab == 'new':
            metas = self.convert_new2old(metas)

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
        text_ids = np.asarray([self.text_converter_dict[str(x)] for x in text_ids]).astype(np.int64)
        return text_ids
