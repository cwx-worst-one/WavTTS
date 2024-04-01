import contextlib

import numpy as np
import torch

from samantha.dataio.lite.utils.frontend import phone_to_int, tone_to_int

try:
    from sami_tts_api.engine import TtsEngine, generate_tts_config
except Exception as e:
    print(f"[Warning] Failed loading sami_tts_api: {e}")


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
