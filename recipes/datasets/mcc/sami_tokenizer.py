import string

import torch
from zhon.hanzi import punctuation

punctuation_all = punctuation + string.punctuation
import numpy as np

try:
    from sami_tts_api.engine import TtsEngine, generate_tts_config
except Exception as e:
    print(f"[Warning] Failed loading sami_tts_api: {e}")

import contextlib

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
