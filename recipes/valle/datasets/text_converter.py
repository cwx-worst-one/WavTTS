import json
import numpy as np
from recipes.valle.datasets.sami_tacolabel import generate_tacolabels_from_textstr, enc_taco_label_no_bytes

import re

def split_sentences(text):
    # 定义正则表达式，匹配句号、问号和感叹号
    pattern = r'(?<=[.?!])\s+'
    
    # 使用正则表达式进行分句
    sentences = re.split(pattern, text)
    
    # 返回分句结果
    return sentences

# Conver raw text -> tacolab -> metaids
class TextToTacolabID:
    def __init__(self, metaid_to_textid_path) -> None:
        with open(metaid_to_textid_path, "r") as f:
            self.text_converter_dict = json.load(f)

    def __call__(self, text: str):
        retry = 10
        metas = []
        for sub_text in split_sentences(text):
            for i in range(retry):
                taco_labels = generate_tacolabels_from_textstr(sub_text, "English")
                if taco_labels is not None:
                    break
            metas.extend(taco_labels.decode().split("\n")[:-1]) # metas[-1] is "".
        metas = enc_taco_label_no_bytes(None, metas, {"use_prsdword": False, "forced_refix": True})
        text_ids =  metas[0].astype(np.int64) * 1_000_000_000 + \
                    metas[1].astype(np.int64) * 1_000_000 + \
                    metas[2].astype(np.int64) * 1_000 + \
                    metas[3].astype(np.int64)
        text_ids = np.asarray([self.text_converter_dict[str(x)] for x in text_ids]).astype(np.int64)
        return text_ids