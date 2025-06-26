import logging
import re
from functools import partial
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__file__)

import numpy as np


def convert_phonemes(phone, lang):
    """
    sil     0       S       0       O               S
    C0m     4       B       0       O       蔓家    B
    C0an    4       B       0       O       蔓家    E
    C0j     1       E       1       O               B
    C0ia    1       E       1       O               E
    C0uan   6       B       0       O       挽手    S
    C0sh    3       E       0       O               B
    C0ou    3       E       0       O               E
    C0l     2       S       4       O       喽      B
    C0ou    2       S       4       O       喽      E
    。      0       S       4       O               S


    sil     0       S       0       O               S
    C1m  19       B  0       O       蔓家    B
    C1an 19       B  0       O       蔓家    E
    C1j  16       E  1       O               B
    C1ia 16       E  1       O               E
    C1uan        21       B  0       O       挽手    S
    C1sh 18       E  0       O               B
    C1ou 18       E  0       O               E
    C1l  17       S  4       O       喽      B
    C1ou 17       S  4       O       喽      E
    。      0       S       4       O               S
    """
    # print('lang: ', lang)
    # print('before phone: ', phone)
    if isinstance(lang, str):
        lang = lang.split(',')
    if phone:
        if "Cantonese" in lang:
            new_phone = ""
            phone_list = phone.split("\n")
            for item in phone_list:
                item_list = item.split("\t")
                if len(item_list) == 7 or len(item_list) == 6:
                    _phone = item_list[0]
                    _tone = int(item_list[1])
                    _ws = item_list[2]
                    _pw = item_list[3]
                    _stype = item_list[4]
                    _word = item_list[5]
                    if len(item_list) == 7:
                        _unit = item_list[6]
                    else:
                        _unit = ""
                    if _phone[:2] in ["C0"]:
                        _phone = "C1" + _phone[2:]
                        _tone = str(_tone + _N_TONES) # 1~14 -> 16~29
                        new_phone += (
                            "\t".join([_phone, _tone, _ws, _pw, _stype, _word, _unit])
                            + "\n"
                        )
                    else:
                        new_phone += item + "\n"
                else:
                    new_phone += phone + "\n"
            # print('after phone: ', new_phone)
            return new_phone.strip()
        elif "Japanese" in lang:
            new_phone = ''
            phone_list = phone.split('\n')
            for item in phone_list:
                item_list = item.split('\t')
                if len(item_list) == 7 or len(item_list) == 6:
                    _phone = item_list[0]
                    _tone = int(item_list[1])
                    _ws = item_list[2]
                    _pw = item_list[3]
                    _stype = item_list[4]
                    _word = item_list[5]
                    if len(item_list) == 7:
                        _unit = item_list[6]
                    else:
                        _unit = ''
                    if _phone[:6] in ["JP_ML0"]:
                        if _tone != 0:
                            _tone = str(_tone + _N_TONES) # 15,16 -> 30,31
                        else:
                            _tone = str(_tone)
                        new_phone += '\t'.join([_phone, _tone, _ws, _pw, _stype, _word, _unit]) + '\n'
                    else:
                        new_phone += item + '\n'
            return new_phone.strip()
    return phone


def _get_phone_tone_token(phone: str, tone: str) -> str:
    """Merge phone and tone if the phone is vowel"""
    if phone in _ALL_VOWELS:
        return f"{phone}{_TONE_SEP}{tone}"
    return phone


def _merge_phone_tone_list(phonemes: List[str], language=None) -> List[str]:
    if language == "Cantonese":
        return [
            _get_phone_tone_token(phone, tone)
            for phone in phonemes
            for tone in ALL_TONES[_N_TONES:_N_TONES*2]
        ]
    elif language == "Japanese":
        return [
            _get_phone_tone_token(phone, tone)
            for phone in phonemes
            for tone in ALL_TONES[_N_TONES*2:]
        ]
    else:
        return [
            _get_phone_tone_token(phone, tone)
            for phone in phonemes
            for tone in ALL_TONES[:_N_TONES]
        ]


def _symbol_to_int(symbol_list: List[str]) -> Dict[str, int]:
    """Initialize a symbol-to-int dict given a symbol list"""
    symbol_to_int_dict = {"[PAD]": 0, "[EOS]": 1}
    offset = len(symbol_to_int_dict)
    for i, symbol in enumerate(symbol_list):
        if symbol not in symbol_to_int_dict:
            symbol_to_int_dict[symbol] = i + offset
    return symbol_to_int_dict


VOCAB_TYPES = ["phoneme", "phoneme+tone"]
_N_TONES = 15  # number of tones in total
_TONE_SEP = "@"  # separator for phone tone merged symbol


##########################################################################
# SPECIAL SYMBOLS
##########################################################################

_PUNCTUATION_A = [
    "＂",
    "＃",
    "＄",
    "％",
    "＆",
    "＇",
    "（",
    "）",
    "＊",
    "＋",
    "，",
    "－",
    "／",
    "：",
    "；",
    "＜",
    "＝",
    "＞",
    "＠",
    "［",
    "＼",
    "］",
    "＾",
    "＿",
    "｀",
    "｛",
    "｜",
    "｝",
    "～",
    "｟",
    "｠",
    "｢",
    "｣",
    "､",
    "\u3000",
    "、",
    "〃",
    "〈",
    "〉",
    "《",
    "》",
    "「",
    "」",
    "『",
    "』",
    "【",
    "】",
    "〔",
    "〕",
    "〖",
    "〗",
    "〘",
    "〙",
    "〚",
    "〛",
    "〜",
    "〝",
    "〞",
    "〟",
    "〰",
    "〾",
    "〿",
    "–",
    "—",
    "‘",
    "’",
    "‛",
    "“",
    "”",
    "„",
    "‟",
    "…",
    "‧",
    "﹏",
    "﹑",
    "﹔",
    "·",
    "！",
    "？",
    "｡",
    "。",
]
_PUNCTUATION_B = [
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
]
_ALL_PUNCTUATION = _PUNCTUATION_A + _PUNCTUATION_B
_SPECIAL_SYMBOLS = ["......", "...", "……", "--", "——"]
_SIL_SYMBOLS = ["sil", "sp", "pau"]

SIL_PUNC_SYMBOLS = _SIL_SYMBOLS + _ALL_PUNCTUATION + _SPECIAL_SYMBOLS

SEP_SYMBOLS = ["zh_word_sep", "en_word_sep", "syl_sep", "cant_word_sep"]

# NOTE (Yilin): Adding special tags changes the vocab, which might affect the
# compatiblity with previous models.
# For singer<#> tags:
# Yilin: There is no upper limit of singer number. To make
# the tags more deterministic, I set 20 singers as the max
# value (It is probably a bad idea to have too many singers).
# Any sample that has a singer tag other than these should
# be discarded.

# NOTE (QQ) to ensure sft experiment based on V5 pretrian ckpt, makesure exactly 23 SINGER_TAGS 
# TODO (QQ, vibertthio) Use it for new sections, so we can ship this to v5. Restore SINGER_TAGS when new deepchorus is fully integrated.
# SINGER_TAGS = ["singer" + str(i) for i in range(14)] + ["合", "女", "男", "中", "童", "念", "多", "伴", "对"]
SINGER_TAGS = ["interlude", "head in", "solo", "head out",
               "exposition", "development", "recapitulation", "cadenza", 
               "coda", "hook", "breakdown", "build-up", 
               "drop", "silence", "spoken", "applause",
               "noise", "fill", "break", "hold"
               ] + ["pre-chorus", "other", "end"]
# NOTE: **Always update the regex pattern whenver the tags are updated**
# Pattern: "<singer_tag>:<text>"
# The pattern should match all the singer tags in the dataset
# What we select eventually is a subset `singer_tags`.
# Filtering should be done after matching.
SINGER_PATTERN = r"^(singer\d+|男|女|合)?:\s*((?:.|\n)*)"

SECTION_TAGS = ["silence", "chorus", "verse", "bridge", "inst", "outro", "intro"]

# Additioanl section tags from DeepChorus 3: https://bytedance.us.larkoffice.com/docx/WPPqdJ08ToRhy3xRIGIuXxjNshe
ADDITIONAL_SECTION_TAGS = ["interlude", "head in", "solo", "head out",
                           "exposition", "development", "recapitulation", "cadenza", 
                           "coda", "hook", "breakdown", "build-up", 
                           "drop", "silence", "spoken", "applause",
                           "noise", "fill", "break", "hold", "pre-chorus", "other", "end"]
# We assume the followings are without utterances
NO_VOCAL_SECTION_TAGS = ["silence", "inst", "outro", "intro",
                            "interlude", "head in", "solo", "head out",
                            "exposition", "development", "recapitulation", "cadenza",
                            "coda"]

SECTION_PARENS = ["[]", "【】", "()", "（）", "<>", "《》", "{}", "「」"]
SECTION_PATTERNS = [
    re.compile(f"^\\{paren[0]}(.*?)\\{paren[1]}\\s*((?:.|\n)*)")
    for paren in SECTION_PARENS
]
STANDARD_SECTION_PATTERN = SECTION_PATTERNS[0]

SECTION_INSTRUMENTS = [
    "Other Inst",
    "Woodwinds",
    "Piccolo",
    "Flute",
    "Clarinet",
    "Oboe",
    "English_Horn",
    "Pipe",
    "Saxophone",
    "Soprano/Alto_Sax",
    "Tenor_Sax",
    "Baritone_Sax",
    "Bassoon",
    "Brass",
    "Trumpet",
    "French_Horn",
    "Trombone",
    "Tuba",
    "Brass_Section",
    "Synth_Brass",
    "Percussion",
    "Drums",
    "Drum_Set",
    "Drum_set",
    "Bassdrum",
    "Bass_Drum",
    "Snare",
    "Hi_hat",
    "Tom",
    "Tom_Tom",
    "Cymbals",
    "Crash_Cymbal",
    "Chromatic_Percussion",
    "Marimba",
    "Bells",
    "Belltree",
    "Xylophone",
    "Glockenspiel",
    "Vibraphone",
    "Chimes",
    "Tubular_Bells",
    "Percussive",
    "Sandhammer",
    "Tambourine",
    "Timpani",
    "Synth_Drums",
    "Synth_Kick",
    "Synth_Snare",
    "Synth_Hi_hat",
    "Synth_Tom",
    "Synth_Cymbals",
    "Synth_Clave",
    "Fx",
    "Clap",
    "Keys",
    "Acoustic_Piano",
    "Electric_piano",
    "Organ",
    "Accordion",
    "Strings",
    "Violin",
    "Viola",
    "Cello",
    "Contrabass",
    "Double_Bass",
    "String_Ensemble",
    "Synth_Strings",
    "Bass",
    "Electric_Bass",
    "Synth_Bass",
    "Guitar",
    "Acoustic_Guitar",
    "Electric_Guitar",
    "Clean_Electric_Guitar",
    "Distorted_Electric_Guitar",
    "Chinese Traditional Instruments",
    "Di",
    "Xiao",
    "Suona",
    "Erhu",
    "Guzheng",
    "Pipa",
    "Yangqin",
    "Sheng",
    "Hulusi",
    "Panflute",
    "Xun",
    "Matouqin",
    "Ruan",
    "Sanxian",
    "Guqin",
    "Bianzhong",
    "Chinese_Drums",
    "Synthesizers",
    "Synth_Pluck",
    "Synth_Lead",
    "Synth_Pad",
    "Synth_Effects",
    "Plucked Strings",
    "Ukelele",
    "Orchestral_Harp",
    "Banjo",
    "Mandolin",
    "Ethnic",
    "Bagpipe",
    "Dulcimer",
    "Hangdrum",
    "Harmonica",
    "Irishwhistle",
    "Ocarina",
    "Sitar",
    "Whistle",
    "Musicbox",
    "Sound_Effects",
    "Vocal",
    "Backingvocal",
    "Chorus",
    "Choir_and_Voice",
    "Vocal_Chops",
    "Ride_Cymbal",
    "Harpsichord",
    "Pipe_Organ",
    "Synth_Bell",
    "Double_Bass_Pizzicato",
    "Church_Bells",
    "Singing_Bowl",
    "Castanets",
    "Triangle",
    "Claves",
    "Cowbell",
    "Mark_Tree",
    "Congas/Bongos",
    "Cajón/Box_drum",
    "Wind_Chimes",
    "Orchestral_Drums",
    "Orchestral_Bassdrum",
    "Orchestral_Snare",
    "Orchestral_Cymbals",
    "Taiko_Drums",
    "Tam_Tam",
    "Body_Percussion",
    "Finger_snaps",
    "Beat_box",
    "Chinese_Percussion",
    "Gongs",
    "Wood_Block",
    "Tang_Drums",
    "Ban_Drums",
    "Chinese_Clappers",
    "Chinese_Cymbals",
    "Yunluo",
    "Bangzi",
    "Synth_Chord",
]
##########################################################################
# PHONEMES
##########################################################################

# en
EN_CONSONANTS = [
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
]  # 24

EN_VOWELS = [
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
]  # 18

# zh
ZH_CONSONANTS = [
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
]  # 21

ZH_VOWELS = [
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
]  # 77

# cantonese
CANTO_CONSONANTS = [
    "C1b",
    "C1c",
    "C1ch",
    "C1d",
    "C1f",
    "C1g",
    "C1h",
    "C1j",
    "C1k",
    "C1l",
    "C1m",
    "C1n",
    "C1p",
    "C1q",
    "C1r",
    "C1s",
    "C1sh",
    "C1t",
    "C1x",
    "C1z",
    "C1zh",
]  # 21

CANTO_VOWELS = [
    "C1a",
    "C1ai",
    "C1air",
    "C1an",
    "C1ang",
    "C1angr",
    "C1anr",
    "C1ao",
    "C1aor",
    "C1ar",
    "C1e",
    "C1ei",
    "C1eir",
    "C1en",
    "C1eng",
    "C1engr",
    "C1enr",
    "C1er",
    "C1i",
    "C1ia",
    "C1ian",
    "C1iang",
    "C1iangr",
    "C1ianr",
    "C1iao",
    "C1iaor",
    "C1iar",
    "C1ie",
    "C1ier",
    "C1ii",
    "C1iii",
    "C1in",
    "C1ing",
    "C1ingr",
    "C1inr",
    "C1io",
    "C1iong",
    "C1iongr",
    "C1iou",
    "C1iour",
    "C1ir",
    "C1ng",
    "C1o",
    "C1or",
    "C1ong",
    "C1ongr",
    "C1ou",
    "C1our",
    "C1u",
    "C1ua",
    "C1uai",
    "C1uair",
    "C1uan",
    "C1uang",
    "C1uangr",
    "C1uanr",
    "C1uar",
    "C1uei",
    "C1ueir",
    "C1uen",
    "C1ueng",
    "C1uengr",
    "C1uenr",
    "C1uer",
    "C1uo",
    "C1uor",
    "C1ur",
    "C1v",
    "C1van",
    "C1vanr",
    "C1ve",
    "C1ver",
    "C1vn",
    "C1vnr",
    "C1vr",
    "C1iir",
    "C1iiir",
]  # 77

JA_CONSONANTS = [
    "JP_ML0P",
    "JP_ML0B",
    "JP_ML0B-Y",
    "JP_ML0CL",
    "JP_ML0CX",
    "JP_ML0D",
    "JP_ML0DC",
    "JP_ML0D-Y",
    "JP_ML0FP",
    "JP_ML0G",
    "JP_ML0G-W",
    "JP_ML0G-Y",
    "JP_ML0H",
    "JP_ML0J",
    "JP_ML0K",
    "JP_ML0K-W",
    "JP_ML0K-Y",
    "JP_ML0M",
    "JP_ML0M-Y",
    "JP_ML0N",
    "JP_ML0NN",
    "JP_ML0N-Y",
    "JP_ML0P-Y",
    "JP_ML0RD",
    "JP_ML0RR-Y",
    "JP_ML0S",
    "JP_ML0T",
    "JP_ML0TS",
    "JP_ML0T-Y",
    "JP_ML0V",
    "JP_ML0V-Y",
    "JP_ML0W",
    "JP_ML0XC",
    "JP_ML0Y",
    "JP_ML0Z",
]
# 35

JA_VOWELS = [
    "JP_ML0A",
    "JP_ML0I",
    "JP_ML0UX",
    "JP_ML0E",
    "JP_ML0O",
]
# 5

#########################################################
# VOWELS, TONES
#########################################################
_ALL_VOWELS = EN_VOWELS + ZH_VOWELS + CANTO_VOWELS
ALL_TONES = [str(i) for i in range(_N_TONES*2+2)]

EN_VOWELxTONES = _merge_phone_tone_list(EN_VOWELS)
ZH_VOWELxTONES = _merge_phone_tone_list(ZH_VOWELS)
CANT_VOWELxTONES = _merge_phone_tone_list(CANTO_VOWELS, language="Cantonese")
JA_VOWELxTONES = _merge_phone_tone_list(JA_VOWELS, language="Japanese")

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

#########################################################
# PHONE LEGACY
#########################################################
ALL_PHONES_LEGACY = (
    SIL_PUNC_SYMBOLS
    + EN_CONSONANTS
    + EN_VOWELS
    + ZH_CONSONANTS
    + ZH_VOWELS
    + SEP_SYMBOLS
    # special tags
    + SINGER_TAGS
    + SECTION_TAGS
    # Cantonese
    + CANTO_CONSONANTS
    + CANTO_VOWELS
)
PHONE_TO_INT_LEGACY = _symbol_to_int(ALL_PHONES_LEGACY)  # offset: 0, 1, phones: 2~296

#########################################################
# PHONExTONE LEGACY
#########################################################
# phone tone merged tokens
ALL_PHONExTONES_LEGACY = (
    SIL_PUNC_SYMBOLS
    + EN_CONSONANTS
    + EN_VOWELxTONES
    + ZH_CONSONANTS
    + ZH_VOWELxTONES
    + SEP_SYMBOLS
    # special tags
    + SINGER_TAGS
    + SECTION_TAGS
    # Cantonese
    + CANTO_CONSONANTS
    + CANT_VOWELxTONES
)

# Use `phone_to_int` if vocab_type is "phoneme", use `token_to_int` if vocab_type is "phoneme+tone"
PHONExTONE_TO_INT_LEGACY = _symbol_to_int(ALL_PHONExTONES_LEGACY)  # offset: 0, 1, phones: 2~1626

#########################################################
# PHONExTONE V2
#########################################################
ALL_PHONExTONES_V2 = (
    EN_CONSONANTS
    + EN_VOWELxTONES
    + ZH_CONSONANTS
    + ZH_VOWELxTONES
    + CANTO_CONSONANTS
    + CANT_VOWELxTONES
    + SIL_PUNC_SYMBOLS
    + SEP_SYMBOLS
    + SINGER_TAGS
    + SECTION_TAGS
)
PHONExTONE_TO_INT_V2 = _symbol_to_int(ALL_PHONExTONES_V2)

ALL_PHONExTONES_V3 = (
    ALL_PHONExTONES_V2
    + SECTION_INSTRUMENTS
)
PHONExTONE_TO_INT_V3 = _symbol_to_int(ALL_PHONExTONES_V3)

ALL_PHONExTONES_V4 = (
    ALL_PHONExTONES_V3
    + JA_CONSONANTS
    + JA_VOWELxTONES
    + ["ja_word_sep"]
)
PHONExTONE_TO_INT_V4 = _symbol_to_int(ALL_PHONExTONES_V4)

ALL_PHONExTONES_V5 = (
    ALL_PHONExTONES_V4 
    + ADDITIONAL_SECTION_TAGS    
)
PHONExTONE_TO_INT_V5 = _symbol_to_int(ALL_PHONExTONES_V5)
# ------------------------------------------
# Special tag utils
# ------------------------------------------


def _extract_prefix_tag(text: str, pattern: str) -> Tuple[Optional[str], str]:
    match = re.search(pattern, text)
    if match is None:
        return None, text
    tag, rest_text = match.group(1), match.group(2)
    if tag is not None:
        tag = tag.strip()
    if rest_text is None:
        raise ValueError("Invalid text or regex pattern.")
    return tag, rest_text


def extract_singer_tag(text: str) -> Tuple[Optional[str], str]:
    """Extract the singer tag from the input text.
    :param text: A string that might or might not contain a leading singer tag, followed by a colon.
    :return: The singer tag (None if empty) and the rest of the text without colon and leading whitespace.
    """
    return _extract_prefix_tag(text, SINGER_PATTERN)


def add_singer_tag(singer_tag: Optional[str], text: str) -> str:
    if singer_tag:
        return f"{singer_tag}:{text}"
    return text


def norm_section_tag(section_tag: str) -> Optional[str]:
    # Tag validation happens in Phrase.
    norm_tag = section_tag.lower().strip()
    # Map pre-chorus to bridge
    if norm_tag in ["pre_chorus", "prechorus", "pre-chorus"]:
        return "bridge"
    if norm_tag.find("主歌") != -1 or norm_tag.find("verse") != -1:
        return "verse"
    if norm_tag.find("副歌") != -1 or norm_tag.find("chorus") != -1:
        return "chorus"
    return norm_tag if norm_tag in SECTION_TAGS else None


def extract_section_tag(
    text: str, normalize_tag: bool = False
) -> Tuple[Optional[str], str]:
    """Extract the section tag from the input text.
    :param text: A string that might or might not contain a leading section tag.
    :param normalize_tag:
        True:
            1. Other parentheses in `section_parens` are also allowed.
            2. Tag is normalized to lower case. If the normalized tag is not allowed, the tag is discarded.
        False: The section name must be enclosed by "[]".
    :return: The section tag (None if empty) and the rest of the text without colon and leading whitespace.
    """
    if not normalize_tag:
        return _extract_prefix_tag(text, STANDARD_SECTION_PATTERN)
    for _section_pattern in SECTION_PATTERNS:
        tag, rest_text = _extract_prefix_tag(text, _section_pattern)
        if tag is not None:
            section_tag = norm_section_tag(tag)
            # if it is not norm section tag, keep prefix_tag as lyrics
            return section_tag, rest_text if section_tag else tag + rest_text
    return None, text


def add_section_tag(section_tag: Optional[str], text: str) -> str:
    if section_tag:
        sep = " " if text else ""
        return f"[{section_tag}]{sep}{text}"
    return text


# ------------------------------------------
# Segment parsing
# ------------------------------------------
def convert_v3_to_v1(tacolab):
    tacolab_v1 = []
    # en, zh
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
            raise ValueError(f"Wrong tacolab {x_split}")
        tacolab_v1.append("\t".join([phone, tone, "0.0 0.0 0.0 1.0", ws, pw]))
    return tacolab_v1


def is_english_spanish_char(char):
    special_Spanish_chars_list = [
        "á",
        "é",
        "í",
        "ó",
        "ú",
        "Á",
        "É",
        "Í",
        "Ó",
        "Ú",
        "ñ",
        "Ñ",
        "¡",
        "¿",
        "ü",
        "Ü",
    ]
    if (
        ("\u0041" <= char <= "\u005a")
        or ("\u0061" <= char <= "\u007a")
        or char in special_Spanish_chars_list
    ):
        return True
    else:
        return False


def get_lang(tacolab):
    if len(tacolab[0].split("\t")) != 5:
        if (
            tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
            or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
        ):
            tacolab = tacolab[1:]
    prefix_phn_list = [x.split("\t")[0][:2] for x in tacolab]
    if "C1" in prefix_phn_list:
        lang = "zh_cant_en"
    elif "C0" in prefix_phn_list:
        if "E0" in prefix_phn_list:
            lang = "zh_en"
        else:
            lang = "zh"
    elif "JP" in prefix_phn_list:
        lang = "ja"
    else:
        lang = "en"
    return lang


def get_lang_by_text(text):
    text = text.replace("'", "")
    # en, zh
    len_en_word = 0
    len_zh_char = 0
    i = 0
    while i < len(text):
        x = text[i]
        if x in _ALL_PUNCTUATION:  # punc
            i += 1
            continue
        elif "\u4e00" <= x <= "\u9fff":  # zh
            len_zh_char += 1
            i += 1
        elif is_english_spanish_char(x):  # en with little spanish
            i += 1
            if i >= len(text):
                len_en_word += 1
                break
            while is_english_spanish_char(text[i]):
                i += 1
                if i >= len(text):
                    break
            len_en_word += 1
            continue
        else:  # blank or digit
            if not (text[i] == " " or text[i].isdigit()):
                return None
            i += 1

    lang = "en"
    if len_zh_char > len_en_word:
        lang = "zh"

    return lang


def _get_token(phoneme: str, tone: Optional[str], vocab_type: str, vocab: dict) -> Tuple[str, int]:
    """
    :return: (token, token_id)
    """
    if vocab_type == "phoneme":
        return phoneme, vocab[phoneme]
    elif vocab_type == "phoneme+tone":
        _tk = _get_phone_tone_token(phoneme, tone)
        return _tk, vocab[_tk]
    raise ValueError(f"Unsupported vocab_type {vocab_type}.")


def _insert_token(
    phone: str, tone: str, tokens: List[str], token_ids: List[int], vocab_type: str, vocab: dict
) -> None:
    if phone == "sp":  # patch: skip sp
        return
    token, token_id = _get_token(phone, tone, vocab_type, vocab)
    tokens.append(token)
    token_ids.append(token_id)


def convert_labels_to_text_id_zh(
    tacolab: List[str], vocab_type: str, vocab: dict
) -> Tuple[List[str], List[int]]:
    assert len(tacolab[0].split("\t")) == 7, (len(tacolab[0].split("\t")), tacolab[0])
    if tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit":
        tacolab = tacolab[1:]

    tokens, token_ids = [], []
    insert_token = partial(_insert_token, vocab_type=vocab_type, vocab=vocab)

    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        x_split = x.split("\t")
        phone, tone, ws, pw, stype, word, unit = x_split
        assert phone in ALL_PHONES_LEGACY, f"{phone} not in phone set"
        assert tone in ALL_TONES, f"{tone} not in tone set"

        insert_token(phone, tone, tokens, token_ids)

        if phone[:2] == "C0" and unit in ["S", "E"]:
            insert_token("syl_sep", None, tokens, token_ids)
            if ws in ["S", "E"]:
                insert_token("zh_word_sep", None, tokens, token_ids)
        elif phone[:2] == "E0" and pw != "0":
            insert_token("en_word_sep", None, tokens, token_ids)

    return tokens, token_ids


def convert_labels_to_text_id_zh_en(
    tacolab: List[str], vocab_type: str, vocab: dict,
) -> Tuple[List[str], List[int]]:
    assert len(tacolab[0].split("\t")) == 7 or len(tacolab[0].split("\t")) == 6, (
        len(tacolab[0].split("\t")),
        tacolab[0],
    )
    if (
        tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
        or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
    ):
        tacolab = tacolab[1:]

    tokens, token_ids = [], []
    insert_token = partial(_insert_token, vocab_type=vocab_type, vocab=vocab)

    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        x_split = x.split("\t")
        if len(x_split) == 7:
            phone, tone, ws, pw, stype, word, unit = x_split
        elif len(x_split) == 6:
            phone, tone, ws, pw, stype, word = x_split
            unit = None
        else:
            print("Wrong tacolab", x_split)
            return None

        assert phone in ALL_PHONES_LEGACY, f"{phone} not in phone set"
        assert tone in ALL_TONES, f"{tone} not in tone set"

        insert_token(phone, tone, tokens, token_ids)

        if phone[:2] == "C0" and unit in ["S", "E"]:
            insert_token("syl_sep", None, tokens, token_ids)
            if ws in ["S", "E"]:
                insert_token("zh_word_sep", None, tokens, token_ids)
        elif phone[:2] == "E0" and pw != "0":
            insert_token("en_word_sep", None, tokens, token_ids)

    return tokens, token_ids


def convert_labels_to_text_id_en(
    tacolab: List[str], vocab_type: str, vocab: dict
) -> Tuple[List[str], List[int]]:
    if len(tacolab[0].split("\t")) != 5:
        tacolab = convert_v3_to_v1(tacolab)

    tokens, token_ids = [], []
    insert_token = partial(_insert_token, vocab_type=vocab_type, vocab=vocab)

    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        phone, tone, _, ws, pw = x.split("\t")
        assert phone in ALL_PHONES_LEGACY, f"{phone} not in phone set"
        assert tone in ALL_TONES, f"{tone} not in tone set"

        insert_token(phone, tone, tokens, token_ids)

        if pw != "0":
            insert_token("en_word_sep", None, tokens, token_ids)

    return tokens, token_ids


def convert_labels_to_text_id_zh_cant_en(
    tacolab: List[str], vocab_type: str, vocab: dict
) -> Tuple[List[str], List[int]]:
    assert len(tacolab[0].split("\t")) == 7 or len(tacolab[0].split("\t")) == 6, (
        len(tacolab[0].split("\t")),
        tacolab[0],
    )
    if (
        tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
        or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
    ):
        tacolab = tacolab[1:]

    tokens, token_ids = [], []
    insert_token = partial(_insert_token, vocab_type=vocab_type, vocab=vocab)

    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        x_split = x.split("\t")
        if len(x_split) == 7:
            phone, tone, ws, pw, stype, word, unit = x_split
        elif len(x_split) == 6:
            phone, tone, ws, pw, stype, word = x_split
            unit = None
        else:
            print("Wrong tacolab", x_split)
            return None

        assert phone in ALL_PHONES_LEGACY, f"{phone} not in phone set"
        assert tone in ALL_TONES, f"{tone} not in tone set"

        insert_token(phone, tone, tokens, token_ids)

        if phone[:2] == "C0" and unit in ["S", "E"]:
            insert_token("syl_sep", None, tokens, token_ids)
            if ws in ["S", "E"]:
                insert_token("zh_word_sep", None, tokens, token_ids)
        elif phone[:2] == "C1" and unit in ["S", "E"]:
            insert_token("syl_sep", None, tokens, token_ids)
            if ws in ["S", "E"]:
                insert_token("cant_word_sep", None, tokens, token_ids)
        elif phone[:2] == "E0" and pw != "0":
            insert_token("en_word_sep", None, tokens, token_ids)

    return tokens, token_ids

def convert_labels_to_text_id_ja_en(
    tacolab: List[str], vocab_type: str, vocab: dict
) -> Tuple[List[str], List[int]]:
    assert (
        len(tacolab[0].split("\t")) == 7 or len(tacolab[0].split("\t")) == 6
    ), (len(tacolab[0].split("\t")), tacolab[0])
    if (
        tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
        or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
    ):
        tacolab = tacolab[1:]

    tokens, token_ids = [], []
    insert_token = partial(_insert_token, vocab_type=vocab_type, vocab=vocab)

    for i in range(len(tacolab)):
        x = tacolab[i]

        if i != 0 and x.split("\t")[0] == "sil":
            continue

        x_split = x.split("\t")
        if len(x_split) == 7:
            phone, tone, ws, pw, stype, word, unit = x_split
        elif len(x_split) == 6:
            phone, tone, ws, pw, stype, word = x_split
            unit = None
        else:
            print("Wrong tacolab", x_split)
            return None

        assert phone in ALL_PHONExTONES_V4, f"{phone} not in phone set"
        assert tone in ALL_TONES, f"{tone} not in tone set"

        insert_token(phone, tone, tokens, token_ids)

        if phone[:6] == "JP_ML0" and unit in ["S", "E"]:
            insert_token("syl_sep", None, tokens, token_ids)
            if ws in ["S", "E"]:
                insert_token("ja_word_sep", None, tokens, token_ids)
        elif phone[:2] == "E0" and pw != "0":
            insert_token("en_word_sep", None, tokens, token_ids)

    return tokens, token_ids


_label_conversion_fns = {
    "zh": convert_labels_to_text_id_zh,
    "zh_en": convert_labels_to_text_id_zh_en,
    "en": convert_labels_to_text_id_en,
    "zh_cant_en": convert_labels_to_text_id_zh_cant_en,
    "ja": convert_labels_to_text_id_ja_en, 
}


def convert_labels_to_text_id(
    tacolab: Optional[List[str]],
    vocab_type: str,
    vocab: dict,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert tacolab into token ids.
    :param vocab_type: "phonenme" (use phoneme only) or "phoneme+tone" (merge phoneme and tone into one token)
    :param prefix_tags: Special tags that will be prepended to the beginning of the token sequence.
    :return: (np.array([token_ids, token_ids]), tokens, tokens)

    NOTE: Always only use the first row of the numpy array, and the second return value token (list of token strings).
    Do NOT use any other information.

    The return value does not make sense on its own. This format is only kept for backward compatibility.
    Historically, the return values are (np.array([phone_ids, tone_ids]), phones, tones). However, it is not flexible
    enough to support the new token types.

    If you wish to use the tone alone, or any other token type, add your vocab and modity the `_get_token` function.

    All of the exisitng projects (04/03/2024) use the first row of the token id array (phones) and the phone strings.
    """
    if vocab_type not in VOCAB_TYPES:
        raise ValueError(f"Invalid vocab_type: {vocab_type}")
    try:
        if tacolab is not None:
            lang = get_lang(tacolab)
            tokens, token_ids = _label_conversion_fns[lang](tacolab, vocab_type, vocab)
        else:
            tokens, token_ids = [], []
        return np.stack([token_ids, token_ids]), tokens, tokens

    except Exception as e:
        raise ValueError(f"Error converting labels to text id: {e}")
