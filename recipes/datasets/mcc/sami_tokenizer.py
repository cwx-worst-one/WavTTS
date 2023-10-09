import string

import torch
from zhon.hanzi import punctuation

punctuation_all = punctuation + string.punctuation
import os
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

try:
    from sami_tts_api.engine import TtsEngine, generate_tts_config
except Exception as e:
    print(f"[Warning] Failed loading sami_tts_api: {e}")

import contextlib
import json

import euler

euler.install_thrift_import_hook()
from recipes.datasets.mcc.server.base_thrift import Base
from recipes.datasets.mcc.server.sami_thrift import SAMI, InvokeRequest

GATEWAYS = ["sd://lab.sami.gateway", "sd://lab.sami.gateway.service.hl"]
_client = None
_base = None

### phones
# silence symbol
sil_symbols = ["sil", "sp", "pau"]

# punc
punctuation_all = list(punctuation + string.punctuation)
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
]  # 24

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
]  # 18

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

all_phones = (
    sil_punc_symbols + EN_consonant + EN_vowel + ZH_consonant + ZH_vowel + sep_strs
)


### tones
all_tones = [str(i) for i in range(15)] + sep_strs

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
            print("Wrong tacolab", x_split)
            return None
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
    if "C0" in prefix_phn_list:
        if "E0" in prefix_phn_list:
            lang = "zh_en"
        else:
            lang = "zh"
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
        if x in punctuation_all:  # punc
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


def convert_labels_to_text_id(tacolab):
    try:
        lang = get_lang(tacolab)
        phone_ids = []
        tone_ids = []
        phones = []
        tones = []
        if lang == "zh":
            assert len(tacolab[0].split("\t")) == 7, (
                len(tacolab[0].split("\t")),
                tacolab[0],
            )
            if tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit":
                tacolab = tacolab[1:]
            for i in range(len(tacolab)):
                x = tacolab[i]
                if i != 0 and x.split("\t")[0] == "sil":
                    continue
                x_split = x.split("\t")
                phone, tone, ws, pw, stype, word, unit = x_split
                assert phone in phone_to_int, f"{phone} not in phone set"
                assert tone in tone_to_int, f"{tone} not in tone set"

                phone_ids.append(phone_to_int[phone])
                tone_ids.append(tone_to_int[tone])
                phones.append(phone)
                tones.append(tone)
                if phone[:2] == "C0":
                    if unit in ["S", "E"]:
                        phone_ids.append(phone_to_int["syl_sep"])
                        tone_ids.append(tone_to_int["syl_sep"])
                        phones.append("syl_sep")
                        tones.append("syl_sep")
                        if ws in ["S", "E"]:
                            phone_ids.append(phone_to_int["zh_word_sep"])
                            tone_ids.append(tone_to_int["zh_word_sep"])
                            phones.append("zh_word_sep")
                            tones.append("zh_word_sep")
                elif phone[:2] == "E0":
                    if pw != "0":
                        phone_ids.append(phone_to_int["en_word_sep"])
                        tone_ids.append(tone_to_int["en_word_sep"])
                        phones.append("en_word_sep")
                        tones.append("en_word_sep")
        elif lang == "zh_en":
            assert (
                len(tacolab[0].split("\t")) == 7 or len(tacolab[0].split("\t")) == 6
            ), (len(tacolab[0].split("\t")), tacolab[0])
            if (
                tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
                or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
            ):
                tacolab = tacolab[1:]
            for i in range(len(tacolab)):
                x = tacolab[i]
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

                assert phone in phone_to_int, f"{phone} not in phone set"
                assert tone in tone_to_int, f"{tone} not in tone set"

                phone_ids.append(phone_to_int[phone])
                tone_ids.append(tone_to_int[tone])
                phones.append(phone)
                tones.append(tone)
                if phone[:2] == "C0":
                    if unit in ["S", "E"]:
                        phone_ids.append(phone_to_int["syl_sep"])
                        tone_ids.append(tone_to_int["syl_sep"])
                        phones.append("syl_sep")
                        tones.append("syl_sep")
                        if ws in ["S", "E"]:
                            phone_ids.append(phone_to_int["zh_word_sep"])
                            tone_ids.append(tone_to_int["zh_word_sep"])
                            phones.append("zh_word_sep")
                            tones.append("zh_word_sep")
                elif phone[:2] == "E0":
                    if pw != "0":
                        phone_ids.append(phone_to_int["en_word_sep"])
                        tone_ids.append(tone_to_int["en_word_sep"])
                        phones.append("en_word_sep")
                        tones.append("en_word_sep")
        elif lang == "en":
            if len(tacolab[0].split("\t")) != 5:
                tacolab = convert_v3_to_v1(tacolab)
            for i in range(len(tacolab)):
                x = tacolab[i]
                if i != 0 and x.split("\t")[0] == "sil":
                    continue
                x_split = x.split("\t")
                phone, tone, _, ws, pw = x.split("\t")
                assert phone in phone_to_int, f"{phone} not in phone set"
                assert tone in tone_to_int, f"{tone} not in tone set"
                phone_ids.append(phone_to_int[phone])
                tone_ids.append(tone_to_int[tone])
                phones.append(phone)
                tones.append(tone)
                if pw != "0":
                    phone_ids.append(phone_to_int["en_word_sep"])
                    tone_ids.append(tone_to_int["en_word_sep"])
                    phones.append("en_word_sep")
                    tones.append("en_word_sep")
        phone_ids, tone_ids = np.array(phone_ids), np.array(tone_ids)
        return np.stack([phone_ids, tone_ids]), phones, tones
    except Exception as e:
        print(e)
        return None


def InvokeServer(file_id, text, speaker):
    payload_obj = {
        "audio_info": {
            "format": "wav",
            "sample_rate": 24000,
            "pitch_rate": 0,
            "speech_rate": 0,
            "speaker": speaker,
            "need_alignment": True,
            "silence_duration": 0,
        },
        # 'text': '??',
        # 'text': '��������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������á�?'
        # 'text': ''''1. You're listening to Faith Radio Online-Simply to Relax, I'm Faith. When you're faced with so many negative and draining situations, realize how minuscule problems will seem when you view your life as a whole--and remember the positive things.''',
        # 'text': '''Although you'll need a warm coat weather this time of year hardly ever dips below freezing For warmer weather without throngs of tourists and the sweltering humidity come in May or September High average temperatures flit between the mid-70s and the lower 80s''',
        "text": text,
    }
    payload_str = json.dumps(payload_obj)

    global _base
    if _base is None:
        _base = Base()
    req = InvokeRequest(
        Base=_base, access_key="flKJmCtkYc", method="TTS", payload=payload_str
    )

    global _client
    if _client is None:
        for gateway in GATEWAYS:
            _client = euler.Client(
                SAMI, gateway + "?cluster=release_thrift", timeout=1200
            )
            result = _client.Invoke(req)
            if result.BaseResp.StatusMessage == "ServerFailedInvoke":
                continue
            else:
                break
    result = _client.Invoke(req)
    return result.data, file_id, result.BaseResp.StatusMessage


def InvokeServerPunc(file_id, text, speaker):
    payload_obj = {
        "audio_info": {
            "format": "wav",
            "sample_rate": 24000,
            "pitch_rate": 0,
            "speech_rate": 0,
            "speaker": speaker,
            "need_alignment": True,
            "silence_duration": 0,
        },
        "internal": {"lab_version": "V3", "enable_recover_puncts": True},
        # 'text': '??',
        # 'text': '��������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������á�?'
        # 'text': ''''1. You're listening to Faith Radio Online-Simply to Relax, I'm Faith. When you're faced with so many negative and draining situations, realize how minuscule problems will seem when you view your life as a whole--and remember the positive things.''',
        # 'text': '''Although you'll need a warm coat weather this time of year hardly ever dips below freezing For warmer weather without throngs of tourists and the sweltering humidity come in May or September High average temperatures flit between the mid-70s and the lower 80s''',
        "text": text,
    }
    payload_str = json.dumps(payload_obj)

    global _base
    if _base is None:
        _base = Base()
    req = InvokeRequest(
        Base=_base, access_key="flKJmCtkYc", method="TTS", payload=payload_str
    )

    global _client
    if _client is None:
        for gateway in GATEWAYS:
            _client = euler.Client(
                SAMI, gateway + "?cluster=release_thrift", timeout=1200
            )
            result = _client.Invoke(req)
            if result.BaseResp.StatusMessage == "ServerFailedInvoke":
                continue
            else:
                break
    result = _client.Invoke(req)
    return result.data, file_id, result.BaseResp.StatusMessage


def parse_raw_text(text_filepath):
    text_dict = OrderedDict()
    f = open(text_filepath)
    lines = f.readlines()
    print("lines: ", len(lines))
    for index, line in enumerate(lines):
        metas = line.strip().split("\t")
        if len(metas) == 2:
            text_dict[metas[0]] = metas[1]
        else:
            text_dict[f"{index:08}"] = metas[0]
    return text_dict


def generate_tacolabels_from_textstr(text: str, language="Chinese"):
    if language == "Chinese" or language == "English":
        speaker = "front_end"
    elif language == "Chinese_new":
        speaker = "front_end_zh"
    elif language == "English_new":
        speaker = "front_end_en"
    elif language == "Japanese":
        speaker = "front_end_jp"
    elif language == "BrazilPortuguese":
        speaker = "front_end_bp"
    elif language == "SouthKorean":
        speaker = "front_end_kr"
    else:
        raise ValueError("language error : {}".format(language))

    lab_data, file_id, invoke_response = InvokeServer(None, text, speaker)
    if lab_data is None:
        print(
            f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
        )
    return lab_data


def generate_tacolabels_from_text(text_filepath, lab_output_dir, language="Chinese"):
    os.makedirs(lab_output_dir, exist_ok=True)
    text_dict = parse_raw_text(text_filepath)
    if language == "Chinese" or language == "English":
        speaker = "front_end"
    elif language == "Japanese":
        speaker = "front_end_jp"
    elif language == "BrazilPortuguese":
        speaker = "front_end_bp"
    elif language == "SouthKorean":
        speaker = "front_end_kr"
    else:
        raise ValueError("language error : {}".format(language))
    sucess_labs = []
    for file_id, text in tqdm(text_dict.items()):
        output_path = os.path.join(lab_output_dir, f"{file_id}.lab")
        if os.path.exists(output_path):
            sucess_labs.append(os.path.abspath(output_path))
            continue
        lab_data, file_id, invoke_response = InvokeServer(file_id, text, speaker)
        if lab_data is None:
            print(
                f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
            )
            continue
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(lab_data)
        sucess_labs.append(os.path.abspath(output_path))
    return sucess_labs


def generate_tacolabels_from_text_by_split(
    text_filepath, utt2split, lab_output_dir_prefix, language="Chinese"
):
    # mkdir_or_exist(lab_output_dir)
    text_dict = parse_raw_text(text_filepath)
    if language == "Chinese" or language == "English":
        speaker = "front_end"
    elif language == "Japanese":
        speaker = "front_end_jp"
    elif language == "BrazilPortuguese":
        speaker = "front_end_bp"
    elif language == "SouthKorean":
        speaker = "front_end_kr"
    else:
        raise ValueError("language error : {}".format(language))
    sucess_labs = []
    print("text_dict.items(): ", len(text_dict.items()))
    for file_id, text in tqdm(text_dict.items()):
        if file_id not in utt2split.keys():
            print(f"file_id {file_id} not in utt2split, skip.")
            continue
        split = utt2split[file_id]
        lab_output_dir = os.path.join(lab_output_dir_prefix, split)
        os.makedirs(lab_output_dir, exist_ok=True)
        output_path = os.path.join(lab_output_dir, f"{file_id}.lab")
        if os.path.exists(output_path):
            continue
        lab_data, file_id, invoke_response = InvokeServer(file_id, text, speaker)
        if lab_data is None:
            print(
                f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
            )
            continue
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(lab_data)
        sucess_labs.append(os.path.abspath(output_path))
    return sucess_labs


def generate_tacolabels_from_textstr_punc(text: str, language="Chinese_v3_punc"):
    if language == "Chinese_v3_punc":
        speaker = "front_end_zh"
    elif language == "English_v3_punc":
        speaker = "front_end_en"
    else:
        raise ValueError("language error : {}".format(language))

    lab_data, file_id, invoke_response = InvokeServerPunc(None, text, speaker)
    if lab_data is None:
        print(
            f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
        )
    return lab_data


def generate_tacolabels_engine(text_str):
    lang_key = get_lang_by_text(text_str)
    if lang_key == None:
        print(f"{text_str}: Wrong lang_key")
        return None
    tacolab = None
    if lang_key in ["zh", "zh_en"]:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "Chinese_v3_punc")
        # tacolab = 'phn\ttone\tws\tpwpp\tsentype\tword\tunit' + '\n' + tacolab.decode()
    elif lang_key in ["en"]:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "English_v3_punc")
        # tacolab = 'phn\ttone\tws\tpwpp\tsentype\tword' + '\n' + tacolab.decode()

    return tacolab


class SamiTokenizer:
    def __init__(
        self,
        lib_path="/opt/tiger/sami_engine_cleaned/libs/libsami.so",
        fe="/opt/tiger/sami_tts_api/models/tts_chinese_frontend_model__42.0.model",
        fe_task="tts_chinese_frontend_model",
    ) -> None:
        self.cfg = generate_tts_config()
        self.engine = TtsEngine(lib_path=lib_path, fe=fe)
        self.ex = self.engine.create_fe_executor(task_type=fe_task)

    def __call__(self, text_batch, **kwds):
        text_ids = []
        if isinstance(text_batch, str):
            text_batch = [text_batch]
        for text in text_batch:
            if text.strip() == "":
                text_ids.append(torch.zeros(0).long())
            else:
                with contextlib.redirect_stdout(None):
                    labels, tn = self.ex.run(text, config=self.cfg)
                labels = list(filter(lambda x: x != "", labels.split("\n")))
                text_id, _, _ = convert_labels_to_text_id(labels)
                text_ids.append(torch.from_numpy(text_id[0]).long())
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                text_ids, batch_first=True, padding_value=0
            )
        }


class SamiTokenizerOnline:
    def __call__(self, text_batch, **kwds):
        text_ids = []
        if isinstance(text_batch, str):
            text_batch = [text_batch]
        for text in text_batch:
            if text.strip() == "":
                text_ids.append(torch.zeros(0).long())
            else:
                labels = generate_tacolabels_engine(text).decode()
                labels = list(filter(lambda x: x != "", labels.split("\n")))
                text_id, _, _ = convert_labels_to_text_id(labels)
                text_ids.append(torch.from_numpy(text_id[0]).long())
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                text_ids, batch_first=True, padding_value=0
            )
        }
