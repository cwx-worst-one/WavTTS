import json
import string

import numpy as np
from zhon.hanzi import punctuation

from samantha.utils.sami_tacolabel import (
    enc_taco_label_no_bytes,
    generate_tacolabels_from_textstr,
)

punctuation_all = punctuation + string.punctuation


# Conver raw text -> tacolab -> metaids
class TextToTacolabID:
    def __init__(
        self,
        metaid_to_textid_path,
        lab="old",
        symbol_sets=None,
        text2id_version="v1",
        use_sy=False,
        tacolab_version="oldv1",
    ) -> None:
        with open(metaid_to_textid_path, "r") as f:
            self.text_converter_dict = json.load(f)
        self.lab = lab
        self.symbol_sets = symbol_sets
        self.text2id_version = text2id_version
        self.use_sy = use_sy
        self.tacolab_version = tacolab_version

    def convert_new2old(self, metas):
        new_metas = []
        for meta in metas:
            # wordpost
            phone, tone, wordcateg, prosody, _, _ = meta.split("\t")
            if phone not in self.symbol_sets and prosody not in ["0", "1"]:
                prosody = "1"
            wordpost = "0.0 0.0 0.0 0.0"
            new_meta = "\t".join([phone, tone, wordpost, wordcateg, prosody])
            new_metas.append(new_meta)
        return new_metas

    def __call__(self, text=None, tacolab_list=None):
        if self.text2id_version == "v1":
            if self.tacolab_version == "oldv1":
                assert text != None, text
                if self.lab == "old":
                    taco_labels = generate_tacolabels_from_textstr(text, "English")
                    while taco_labels is None:
                        taco_labels = generate_tacolabels_from_textstr(text, "English")
                elif self.lab == "new":
                    taco_labels = generate_tacolabels_from_textstr(text, "English_new")
                    while taco_labels is None:
                        taco_labels = generate_tacolabels_from_textstr(
                            text, "English_new"
                        )
                if self.lab == "old":
                    metas = taco_labels.decode().split("\n")[:-1]  # metas[-1] is "".
                elif self.lab == "new":
                    metas = taco_labels.decode().split("\n")

                if self.lab == "new":
                    metas = self.convert_new2old(metas)
            elif self.tacolab_version == "newv3":
                assert tacolab_list != None, tacolab_list
                metas = tacolab_list
                if len(metas[0].split("\t")) != 5:  # tacolab_version: V3
                    metas = self.convert_v3_to_v1(metas)
            else:
                print("Wrong tacolab_version", self.tacolab_version)
                exit()

            taco_labels = metas
            for i, item in enumerate(taco_labels):
                if item.split("\t")[0] == "：":
                    taco_labels[i] = "：\t0\t0.0 0.0 0.0 1.0\tS\t2"
            metas = taco_labels

            metas = enc_taco_label_no_bytes(
                None, metas, {"use_prsdword": False, "forced_refix": True}
            )
            text_ids = (
                metas[0].astype(np.int64) * 1_000_000_000
                + metas[1].astype(np.int64) * 1_000_000
                + metas[2].astype(np.int64) * 1_000
                + metas[3].astype(np.int64)
            )
            text_ids = np.asarray(
                [
                    self.text_converter_dict.get(
                        str(x), self.text_converter_dict.get("oov")
                    )
                    for x in text_ids
                ]
            ).astype(np.int64)
        elif self.text2id_version == "v2":
            assert tacolab_list != None, tacolab_list
            if len(tacolab_list[0].split("\t")) != 5:  # tacolab_version: V3
                tacolab_list = self.convert_v3_to_v1(tacolab_list)

            labs = []
            for i in range(len(tacolab_list)):
                x = tacolab_list[i]
                # skip sil in utt
                if i != 0 and x.split("\t")[0] == "sil":
                    continue
                phone, tone, _, ws, pw = x.split("\t")
                lab = "_".join([phone, tone, ws, pw])
                labs.append(lab)

            text_ids = []
            for x in labs:
                if self.text_converter_dict.get(x) is None:
                    print("x: ", x)
                cur_text_id = self.text_converter_dict.get(
                    x, self.text_converter_dict.get("oov")
                )
                text_ids.append(cur_text_id)
            text_ids = np.asarray(text_ids).astype(np.int64)
        elif self.text2id_version == "v3":
            lang = self.get_lang_by_tacolab_list(tacolab_list)
            labs = self.get_labs_v3(tacolab_list, lang)
            if labs is None:
                return None

            text_ids = []
            for x in labs:
                if self.text_converter_dict.get(x) is None:
                    print("x: ", x)
                cur_text_id = self.text_converter_dict.get(
                    x, self.text_converter_dict.get("oov")
                )
                text_ids.append(cur_text_id)
            text_ids = np.asarray(text_ids).astype(np.int64)

        return text_ids

    def convert_v3_to_v1(self, tacolab):
        tacolab_v1 = []
        if (
            tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
            or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
        ):
            tacolab = tacolab[1:]
        for x in tacolab:
            x_split = x.split("\t")
            if len(x_split) == 7:
                phone, tone, ws, pw, stype, word, _ = x_split
            elif len(x_split) == 6:
                phone, tone, ws, pw, stype, word = x_split
            else:
                print("Wrong tacolab", x_split, tacolab)
                return None
            tacolab_v1.append("\t".join([phone, tone, "0.0 0.0 0.0 1.0", ws, pw]))
        return tacolab_v1

    def get_lang_by_tacolab_list(self, tacolab_list):
        if len(tacolab_list[0].split("\t")) != 5:
            if (
                tacolab_list[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
                or tacolab_list[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
            ):
                tacolab_list = tacolab_list[1:]
        prefix_phn_list = [x.split("\t")[0][:2] for x in tacolab_list]
        if "C0" in prefix_phn_list:
            if "E0" in prefix_phn_list:
                lang = "zh_en"
            else:
                lang = "zh"
        else:
            lang = "en"
        return lang

    def get_labs_v3(self, tacolab_list, lang):
        labs = []
        if lang == "zh":
            assert len(tacolab_list[0].split("\t")) == 7, (
                len(tacolab_list[0].split("\t")),
                tacolab_list[0],
            )
            if tacolab_list[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit":
                tacolab_list = tacolab_list[1:]

            for i in range(len(tacolab_list)):
                x = tacolab_list[i]
                if i != 0 and x.split("\t")[0] == "sil":
                    continue
                x_split = x.split("\t")
                phone, tone, ws, pw, stype, word, unit = x_split
                lab = "_".join([phone, tone])
                if (
                    phone in ["sil", "sp"]
                    or phone in punctuation_all
                    or phone[0] in punctuation_all
                ):
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
                if phone[:2] == "C0":
                    if unit in ["S", "E"]:
                        if self.use_sy:
                            syllable_flag = True
                        if ws in ["S", "E"]:
                            word_segment = True
                elif phone[:2] == "E0":
                    if pw != "0":
                        word_segment = True
                else:
                    print("Wrong phone", phone)
                    return None

                if word_segment:
                    labs.append("ws")
                elif syllable_flag:
                    labs.append("sy_zh")
        elif lang == "zh_en":
            assert (
                len(tacolab_list[0].split("\t")) == 7
                or len(tacolab_list[0].split("\t")) == 6
            ), (len(tacolab_list[0].split("\t")), tacolab_list[0])
            if (
                tacolab_list[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
                or tacolab_list[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
            ):
                tacolab_list = tacolab_list[1:]

            for i in range(len(tacolab_list)):
                x = tacolab_list[i]
                if i != 0 and x.split("\t")[0] == "sil":
                    continue
                x_split = x.split("\t")
                if len(x_split) == 7:
                    phone, tone, ws, pw, stype, word, unit = x_split
                elif len(x_split) == 6:
                    phone, tone, ws, pw, stype, word = x_split
                else:
                    print("Wrong tacolab", x_split)
                    return None
                lab = "_".join([phone, tone])
                if (
                    phone in ["sil", "sp"]
                    or phone in punctuation_all
                    or phone[0] in punctuation_all
                ):
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
                if phone[:2] == "C0":
                    if unit.strip() in ["S", "E"]:
                        if self.use_sy:
                            syllable_flag = True
                        if ws in ["S", "E"]:
                            word_segment = True
                elif phone[:2] == "E0":
                    if pw != "0":
                        word_segment = True
                else:
                    print("Wrong phone", phone)
                    return None

                if word_segment:
                    labs.append("ws")
                elif syllable_flag:
                    labs.append("sy_zh")
        elif lang == "en":
            tacolab_list = self.convert_v3_to_v1(tacolab_list)

            for i in range(len(tacolab_list)):
                x = tacolab_list[i]
                if i != 0 and x.split("\t")[0] == "sil":
                    continue
                x_split = x.split("\t")
                phone, tone, _, ws, pw = x.split("\t")
                lab = "_".join([phone, tone])
                if (
                    phone in ["sil", "sp"]
                    or phone in punctuation_all
                    or phone[0] in punctuation_all
                ):
                    if len(labs) >= 1:
                        if labs[-1] == "ws":
                            labs[-1] = lab
                    else:
                        labs.append(lab)
                    continue
                else:
                    labs.append(lab)

                word_segment = False
                if pw != "0":
                    word_segment = True

                if word_segment:
                    labs.append("ws")
        else:
            print("Wrong lang", lang)
            return None
        return labs
