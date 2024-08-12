from .punctuation import punctuation_all

# phones
# silence symbol
sil_symbols = ["sil", "sp", "pau"]

# punc
special_symbols = ["......", "...", "……", "--", "——"]

# break_symbols: silence symbol + punc
sil_punc_symbols = sil_symbols + punctuation_all + special_symbols

# en
EN_consonant = [
    "E0b",
    "E0ch",
    "E0d",
    "E0dh",
    "E0f",
    "E0g",
    "E0h",
    "E0hh",
    "E0jh",
    "E0k",
    "E0l",
    "E0m",
    "E0n",
    "E0p",
    "E0r",
    "E0s",
    "E0sh",
    "E0t",
    "E0th",
    "E0v",
    "E0w",
    "E0y",
    "E0z",
    "E0zh",
]

EN_vowel = [
    "E0aa",
    "E0ae",
    "E0ah",
    "E0ao",
    "E0aw",
    "E0ax",
    "E0ay",
    "E0eh",  # cmu_dict
    "E0er",
    "E0ey",
    "E0ih",
    "E0iy",
    "E0ng",
    "E0ow",
    "E0oy",
    "E0uh",
    "E0uw",  # cmu_dict
    "E0en",
]

# zh
ZH_consonant = [
    "C0b",
    "C0c",
    "C0ch",
    "C0d",
    "C0f",
    "C0g",
    "C0h",
    "C0j",
    "C0k",
    "C0l",
    "C0m",
    "C0n",
    "C0p",
    "C0q",
    "C0r",
    "C0s",
    "C0sh",
    "C0t",
    "C0x",
    "C0z",
    "C0zh",
]

ZH_vowel = [
    "C0a",
    "C0ai",
    "C0air",
    "C0an",
    "C0ang",
    "C0angr",
    "C0anr",
    "C0ao",
    "C0aor",
    "C0ar",
    "C0e",
    "C0ei",
    "C0eir",
    "C0en",
    "C0eng",
    "C0engr",
    "C0enr",
    "C0er",
    "C0i",
    "C0ia",
    "C0ian",
    "C0iang",
    "C0iangr",
    "C0ianr",
    "C0iao",
    "C0iaor",
    "C0iar",
    "C0ie",
    "C0ier",
    "C0ii",
    "C0iii",
    "C0in",
    "C0ing",
    "C0ingr",
    "C0inr",
    "C0io",
    "C0iong",
    "C0iongr",
    "C0iou",
    "C0iour",
    "C0ir",
    "C0ng",
    "C0o",
    "C0or",
    "C0ong",
    "C0ongr",
    "C0ou",
    "C0our",
    "C0u",
    "C0ua",
    "C0uai",
    "C0uair",
    "C0uan",
    "C0uang",
    "C0uangr",
    "C0uanr",
    "C0uar",
    "C0uei",
    "C0ueir",
    "C0uen",
    "C0ueng",
    "C0uengr",
    "C0uenr",
    "C0uer",
    "C0uo",
    "C0uor",
    "C0ur",
    "C0v",
    "C0van",
    "C0vanr",
    "C0ve",
    "C0ver",
    "C0vn",
    "C0vnr",
    "C0vr",
    "C0iir",
    "C0iiir",
]

sep_strs = ["zh_word_sep", "en_word_sep", "syl_sep"]
wordseg_strs = ["B", "E", "M", "S"]  # Begin, End, Middle, Single.

# gobal phone add phone 1129 liuxudong
add_sep_strs = ["jp_accent_sep", 'mx_word_sep']
add_sil_symbols = ['jp_sp', 'mx_sp']
add_sil_symbols_2 = ['id_sp', 'br_sp']

add_special_symbols = ["¡", "¿"]
GP_consonant = ['B', 'B-Y', 'CH', 'CL', 'CX', 'D', 'D-Y', 'DC', 'F', 'FP', 'G', 'G-W', 'G-Y', 'H', 'J', 'K', 'K-W', 'K-Y', 'L', 'M', 'M-Y', 'N', 'N-Y', 'NC', 'NN', 'P', 'P-Y', 'RD', 'RR', 'RR-Y', 'S', 'SH', 'T', 'T-Y', 'TS', 'V', 'V-Y', 'X', 'XC', 'XJ', 'Z']
GP_vowel = ['A', 'AI', 'AU', 'E', 'EI', 'EU', 'I', 'IU', 'O', 'OI', 'U', 'UI', 'UX', 'W', 'Y']

GP_consonant = ['ML0' + i for i in GP_consonant]
GP_vowel = ['ML0' + i for i in GP_vowel]

# 临时sft方案，后续重新pretrain需要预留GP全phone的index位置 id br
GP_consonant_2 = ['ML0W', 'ML0V', 'ML0Y', 'ML0JH', 'ML0ZH', 'ML0Z', 'ML0YC', "ML0B", "ML0CH", "ML0D", "ML0F", "ML0G", "ML0H", "ML0HQ", "ML0JH", "ML0K", "ML0L", "ML0M", "ML0N", "ML0NC", "ML0NG", "ML0P", "ML0RR", "ML0S", "ML0SH", "ML0T", "ML0T", "ML0W", "ML0X", "ML0Y", "ML0Z"]
GP_vowel_2 = ['ML0AY', 'ML0OHW', 'ML0OY~', 'ML0EY~', 'ML0E~', 'ML0EY', 'ML0EW', 'ML0AUY~', 'ML0I~', 'ML0EHY', 'ML0OW', 'ML0U~', 'ML0OW~', 'ML0EH', 'ML0EHW', 'ML0O~', 'ML0IW', 'ML0OH', 'ML0UY~', 'ML0UY', 'ML0OHY', 'ML0AW~', 'ML0AW', 'ML0AU~', 'ML0OY', "ML0A", "ML0AX", "ML0E", "ML0EH", "ML0IH", "ML0IY", "ML0OH", "ML0OW", "ML0OY", "ML0UH", "ML0UW"]
# GP_consonant_vowel_2 = ['ML0AUY~', 'ML0AU~', 'ML0AW', 'ML0AW~', 'ML0AX', 'ML0AY', 'ML0EH', 'ML0EHW', 'ML0EHY', 'ML0EW', 'ML0EY', 'ML0EY~', 'ML0E~', 'ML0HQ', 'ML0IH', 'ML0IW', 'ML0IY', 'ML0I~', 'ML0JH', 'ML0NG', 'ML0OH', 'ML0OHW', 'ML0OHY', 'ML0OW', 'ML0OW~', 'ML0OY', 'ML0OY~', 'ML0O~', 'ML0UH', 'ML0UW', 'ML0UY', 'ML0UY~', 'ML0U~', 'ML0YC', 'ML0ZH']

# id br mx
add_all_phones = add_sep_strs + add_sil_symbols + add_special_symbols + GP_consonant + GP_vowel

# fr de ko
de_global_phones = ['I:', 'YH', 'OE', 'O:', 'EH:', 'RQ', 'PF', 'EO:', 'E:', 'U:', 'IU:', 'A:']
de_global_phones = ['ML0' + i for i in de_global_phones]

fr_global_phones = ['EO', 'EH~', 'OH~', 'AA', 'YW', 'AA~', 'OE~']
fr_global_phones = ['ML0' + i for i in fr_global_phones]

ko_global_phones = ['IX', 'T-H', 'P]', 'K-H', 'K]', 'HQP', 'AH', 'J-H', 'T]', 'HQS', 'HQK', 'HQT', 'P-H', 'HQJ']
ko_global_phones = ['ML0' + i for i in ko_global_phones]

de_fr_ko_sil_symbols = ['de_sp', 'fr_sp', 'ko_sp']


lang_strs = ["others", "zh", "en", "jp"]
all_phones = (
        sil_punc_symbols + EN_consonant + EN_vowel + ZH_consonant + ZH_vowel + sep_strs
        + add_all_phones + GP_consonant_2 + GP_vowel_2 + add_sil_symbols_2
        + de_global_phones + fr_global_phones + ko_global_phones + de_fr_ko_sil_symbols
)


# tones
# all_tones = [str(i) for i in range(15)] + sep_strs
all_tones = [str(i) for i in range(15)] + sep_strs + ['15', '16'] + add_sep_strs

# 0 for padding, 1 for eos
offset = 2

phone_to_int = dict()
for i, phone in enumerate(all_phones):
    if phone not in phone_to_int:
        phone_to_int[phone] = i + offset

tone_to_int = dict()
for i, tone in enumerate(all_tones):
    if tone not in tone_to_int:
        tone_to_int[tone] = i + offset

wordseg_to_int = dict()
for i, wordseg in enumerate(wordseg_strs):
    if wordseg not in wordseg_to_int:
        wordseg_to_int[wordseg] = i + offset

lang_to_int = dict()
for i, lang in enumerate(lang_strs):
    if lang not in lang_to_int:
        lang_to_int[lang] = i + offset
