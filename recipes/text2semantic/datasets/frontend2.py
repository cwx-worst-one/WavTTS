import enum
import enum
from zhon.hanzi import punctuation
import string


### phones
# silence symbol
sil_symbols = ["sil", "sp", "pau"]

# punc
punctuation_all = list(punctuation + string.punctuation)
special_symbols = ["......", "...", "……", "--", "——"]

# break_symbols: silence symbol + punc
sil_punc_symbols = sil_symbols + punctuation_all + special_symbols

# en
EN_consonant = ["E0b", "E0ch", "E0d", "E0dh", "E0f", "E0g", "E0h", "E0hh", "E0jh",
             "E0k", "E0l", "E0m", "E0n", "E0p", "E0r", "E0s", "E0sh", "E0t", "E0th",
             "E0v", "E0w", "E0y", "E0z", "E0zh"]

EN_vowel = ["E0aa", "E0ae", "E0ah", "E0ao", "E0aw", "E0ax", "E0ay", "E0eh",  # cmu_dict
         "E0er", "E0ey", "E0ih", "E0iy", "E0ng", "E0ow", "E0oy", "E0uh", "E0uw",  # cmu_dict
         "E0en", ]

# zh
ZH_consonant = ["C0b", "C0c", "C0ch", "C0d", "C0f", "C0g", "C0h", "C0j",
             "C0k", "C0l", "C0m", "C0n", "C0p", "C0q", "C0r", "C0s", "C0sh", "C0t",
             "C0x", "C0z", "C0zh"]

ZH_vowel = ["C0a", "C0ai", "C0air", "C0an", "C0ang", "C0angr", "C0anr", "C0ao",
         "C0aor", "C0ar", "C0e", "C0ei", "C0eir", "C0en", "C0eng", "C0engr",
         "C0enr", "C0er", "C0i", "C0ia", "C0ian", "C0iang", "C0iangr", "C0ianr",
         "C0iao", "C0iaor", "C0iar", "C0ie", "C0ier", "C0ii", "C0iii",
         "C0in", "C0ing", "C0ingr", "C0inr", "C0io", "C0iong", "C0iongr",
         "C0iou", "C0iour", "C0ir", "C0ng", "C0o", "C0or", "C0ong", "C0ongr",
         "C0ou", "C0our", "C0u", "C0ua", "C0uai", "C0uair", "C0uan", "C0uang",
         "C0uangr", "C0uanr", "C0uar", "C0uei", "C0ueir", "C0uen", "C0ueng",
         "C0uengr", "C0uenr", "C0uer", "C0uo", "C0uor", "C0ur", "C0v", "C0van",
         "C0vanr", "C0ve", "C0ver", "C0vn", "C0vnr", "C0vr", "C0iir", "C0iiir"]

sep_strs = ["zh_word_sep", "en_word_sep", "syl_sep"]

all_phones = sil_punc_symbols + EN_consonant + EN_vowel + ZH_consonant + ZH_vowel + sep_strs


### tones
all_tones = [str(i) for i in range(15)] + sep_strs

# 0 for padding, 1 for eos1, 2 for eos2, ... 
offset = 10

phone_to_int = dict()
for i, phone in enumerate(all_phones):
    if phone not in phone_to_int:
        phone_to_int[phone] = i + offset

tone_to_int = dict()
for i, tone in enumerate(all_tones):
    if tone not in tone_to_int:
        tone_to_int[tone] = i + offset


phonetone_to_int = dict()
index = 0
for i, phone in enumerate(all_phones):
    for j, tone in enumerate(all_tones):
        phonetone_to_int[phone + '_' + tone] = index + offset
        index += 1
