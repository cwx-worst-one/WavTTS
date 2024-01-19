import re

from samantha.utils.sami_tacolabel.symbols import seperate_set

EN_symbols = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
EN_sym_pattern = r"[A-Za-z']+"
EN_ph_pattern = r"/*/"


def convert_pinyin_to_phonememe(filename):
    phonemes = {}
    erhuayin_pinyin = set()
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split("\t")
            phonemes[items[0]] = "".join(["C" + ph + " " for ph in items[1].split()])
            if items[0].endswith("r") and items[0] != "er":
                erhuayin_pinyin.add(items[0])

    return phonemes, erhuayin_pinyin


def phonemes_add_sp_with_text(text_seq, phonemes, py2ph_dict, erhuayin_set):
    phoneme_seq = phonemes.split()
    phoneme_index = 0
    prosody_flag = False
    erhuayin_flag = False
    En_len = 0
    phoneme_seq = ""
    for i, text in enumerate(text_seq):
        if En_len > 0:
            En_len -= 1
            continue
        if text in seperate_set:
            continue
        if erhuayin_flag:
            assert text == "儿"
            erhuayin_flag = False
            continue
        # process prosody info
        if text == "#":
            prosody_flag = True
            continue
        if prosody_flag:
            assert text in ["1", "2", "3", "4"]
            if text in ["3", "4"]:
                phoneme_seq += "sp "
            elif text == "2":
                if text_seq[i + 1] in seperate_set:
                    phoneme_seq += "sp "

            prosody_flag = False
            continue
        tone = phoneme_seq[phoneme_index]
        if tone[:-2] in ["10", "11", "12", "13", "14", "15", "16", "17"]:
            pinyin = tone[:-2]
        elif tone[-1] in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            pinyin = tone[:-1]
        else:
            pinyin = tone
        if pinyin in erhuayin_set:
            assert text != "儿" and pinyin[-1] == "r"
            erhuayin_flag = True
        # English case
        if text in EN_symbols:
            word_seq = re.match(EN_sym_pattern, text_seq[i:])
            temp_phoneme_seq = ""
            if pinyin == "/":
                phoneme_index += 1
            for phoneme_anno in phoneme_seq[phoneme_index:]:
                phoneme_index += 1
                if phoneme_anno == "/":
                    break
                elif phoneme_anno == ".":
                    continue
                else:
                    temp_phoneme_seq = (
                        temp_phoneme_seq + ("E" + phoneme_anno.lower()) + " "
                    )
            assert word_seq is not None
            word = word_seq[0]
            phoneme_seq = phoneme_seq + temp_phoneme_seq
            En_len = len(word) - 1
            print(f"EN {word} -- {temp_phoneme_seq}")
        else:
            phonemes = py2ph_dict[pinyin]
            phonemes = phonemes[:-1] + tone[-1] + " "
            print(f"ZH {text} -- {pinyin} -- {phonemes}")
            phoneme_seq = phoneme_seq + phonemes
            phoneme_index += 1

    if phoneme_seq[-3:] == "sp ":
        phoneme_seq = phoneme_seq[:-3]
    return phoneme_seq
