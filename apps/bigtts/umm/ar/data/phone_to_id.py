import logging

import numpy as np

from samantha.dataio.lite.utils.frontend import (
    phone_to_int,
    tone_to_int,
    wordseg_to_int,
)
from samantha.utils.common import is_float

logger = logging.getLogger(__name__)


class PhoneToId:
    def __init__(self) -> None:
        self.phone_to_int = phone_to_int
        self.tone_to_int = tone_to_int
        self.wordseg_to_int = wordseg_to_int

    def convert_v3_to_v1(self, tacolab):
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
            tacolab_v1.append("\t".join([phone, tone, "0.0 0.0 0.0 1.0", ws, pw, word]))
        return tacolab_v1

    def get_lang(self, tacolab):
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
        elif "JP" in prefix_phn_list:
            lang = "jp"
        else:
            lang = "en"
        return lang

    def convert_tacolab_to_text_id(self, tacolab):
        try:
            lang = self.get_lang(tacolab)
            phone_ids = []
            tone_ids = []
            phones = []
            tones = []
            wordseg_ids = []
            wordsegs = []
            alignments = []

            head_type_dict = {
                "phn\ttone\tws\tpwpp\tsentype\tword": 0,
                "phn\ttone\tws\tpwpp\tsentype\tword\tunit": 1,
                "phn\ttone\tws\tpwpp\tsentype\tword\talignment": 2,
                "phn\ttone\tws\tpwpp\tsentype\tword\tunit\talignment": 3,
            }

            if lang == "zh" or lang == "jp":
                # assert tacolab[0] in head_type_dict, (len(tacolab[0].split('\t')), tacolab[0])
                # head_type = head_type_dict[tacolab[0]]
                # tacolab = tacolab[1:]
                if "phn\ttone\tws\tpwpp\tsentype\tword" in tacolab[0]:
                    assert tacolab[0] in head_type_dict, (
                        len(tacolab[0].split("\t")),
                        tacolab[0],
                    )
                    head_type = head_type_dict[tacolab[0]]
                    tacolab = tacolab[1:]
                else:
                    if len(tacolab[0].split("\t")) == 6:
                        head_type = 0
                    elif len(tacolab[0].split("\t")) == 8:
                        head_type = 3
                    elif is_float(tacolab[0].split("\t")[-1]):
                        head_type = 2
                    else:
                        head_type = 1

                assert head_type in [1, 3]

                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split("\t")[0] == "sil":
                        continue
                    x_split = x.split("\t")
                    if head_type == 0:
                        phone, tone, ws, pw, stype, word = x_split
                    elif head_type == 1:
                        phone, tone, ws, pw, stype, word, unit = x_split
                    elif head_type == 2:
                        phone, tone, ws, pw, stype, word, alignment = x_split
                    elif head_type == 3:
                        phone, tone, ws, pw, stype, word, unit, alignment = x_split

                    if phone.startswith("JP_"):
                        phone_lang, phone = phone.split("_")
                    elif lang == "jp" and (phone == "pau" or phone == "sp"):
                        phone = "jp_sp"

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)
                    wordseg_ids.append(self.wordseg_to_int[ws])
                    if head_type == 2 or head_type == 3:
                        alignments.append(float(alignment))
                    # if phone[:2] == "C0":
                    #     if unit in ['S', 'E']:
                    #         phone_ids.append(self.phone_to_int["syl_sep"])
                    #         tone_ids.append(self.tone_to_int["syl_sep"])
                    #         phones.append("syl_sep")
                    #         tones.append("syl_sep")
                    #         if ws in ["S", "E"]:
                    #             phone_ids.append(self.phone_to_int["zh_word_sep"])
                    #             tone_ids.append(self.tone_to_int["zh_word_sep"])
                    #             phones.append("zh_word_sep")
                    #             tones.append("zh_word_sep")
                    # elif phone[:2] == "E0":
                    #     if pw != "0":
                    #         phone_ids.append(self.phone_to_int["en_word_sep"])
                    #         tone_ids.append(self.tone_to_int["en_word_sep"])
                    #         phones.append("en_word_sep")
                    #         tones.append("en_word_sep")
            elif lang == "en" or lang == "zh_en":
                # import pdb
                # pdb.set_trace()
                # print('*'*50, tacolab[0], '*'*50)

                # assert tacolab[0] in head_type_dict, (len(tacolab[0].split('\t')), tacolab[0])

                if "phn\ttone\tws\tpwpp\tsentype\tword" in tacolab[0]:
                    assert tacolab[0] in head_type_dict, (
                        len(tacolab[0].split("\t")),
                        tacolab[0],
                    )
                    head_type = head_type_dict[tacolab[0]]
                    tacolab = tacolab[1:]
                else:
                    if len(tacolab[0].split("\t")) == 6:
                        head_type = 0
                    elif len(tacolab[0].split("\t")) == 8:
                        head_type = 3
                    elif is_float(tacolab[0].split("\t")[-1]):
                        head_type = 2
                    else:
                        head_type = 1

                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split("\t")[0] == "sil":
                        continue
                    x_split = x.split("\t")
                    if head_type == 0:
                        phone, tone, ws, pw, stype, word = x_split
                    elif head_type == 1:
                        phone, tone, ws, pw, stype, word, unit = x_split
                    elif head_type == 2:
                        phone, tone, ws, pw, stype, word, alignment = x_split
                    elif head_type == 3:
                        phone, tone, ws, pw, stype, word, unit, alignment = x_split

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)
                    if head_type == 2 or head_type == 3:
                        alignments.append(float(alignment))

                    if tone == "0":
                        wordsegs.append("S")
                    else:
                        if word != "":
                            if pw == "1":
                                wordsegs.append("S")
                            else:
                                wordsegs.append("B")
                        else:
                            if pw == "0":
                                wordsegs.append("M")
                            else:
                                wordsegs.append("E")
                    wordseg_ids.append(self.wordseg_to_int[wordsegs[-1]])

            phone_ids, tone_ids = np.array(phone_ids), np.array(tone_ids)
            wordseg_ids = np.array(wordseg_ids)
            alignments = np.array(alignments)

            return (
                np.stack([phone_ids, tone_ids, wordseg_ids]),
                phones,
                tones,
                wordsegs,
                alignments,
            )

        except Exception as e:
            print(e)
            # import pdb
            # pdb.set_trace()
            return None

    def convert_tacolab_to_text_id_infer(self, tacolab):
        try:
            lang = self.get_lang(tacolab)
            phone_ids = []
            tone_ids = []
            phones = []
            tones = []
            wordseg_ids = []
            wordsegs = []
            alignments = []

            if lang == "zh" or lang == "jp":
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

                    if phone.startswith("JP_"):
                        phone_lang, phone = phone.split("_")
                    elif lang == "jp" and (phone == "pau" or phone == "sp"):
                        phone = "jp_sp"

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)
                    wordseg_ids.append(self.wordseg_to_int[ws])
                    # if phone[:2] == "C0":
                    #     if unit in ['S', 'E']:
                    #         phone_ids.append(self.phone_to_int["syl_sep"])
                    #         tone_ids.append(self.tone_to_int["syl_sep"])
                    #         phones.append("syl_sep")
                    #         tones.append("syl_sep")
                    #         if ws in ["S", "E"]:
                    #             phone_ids.append(self.phone_to_int["zh_word_sep"])
                    #             tone_ids.append(self.tone_to_int["zh_word_sep"])
                    #             phones.append("zh_word_sep")
                    #             tones.append("zh_word_sep")
                    # elif phone[:2] == "E0":
                    #     if pw != "0":
                    #         phone_ids.append(self.phone_to_int["en_word_sep"])
                    #         tone_ids.append(self.tone_to_int["en_word_sep"])
                    #         phones.append("en_word_sep")
                    #         tones.append("en_word_sep")
            elif lang == "en" or lang == "zh_en":
                if "phn\ttone\tws\tpwpp\tsentype\tword" in tacolab[0]:
                    assert tacolab[0] in head_type_dict, (
                        len(tacolab[0].split("\t")),
                        tacolab[0],
                    )
                    head_type = head_type_dict[tacolab[0]]
                    tacolab = tacolab[1:]
                else:
                    if len(tacolab[0].split("\t")) == 6:
                        head_type = 0
                    elif len(tacolab[0].split("\t")) == 8:
                        head_type = 3
                    elif is_float(tacolab[0].split("\t")[-1]):
                        head_type = 2
                    else:
                        head_type = 1

                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split("\t")[0] == "sil":
                        continue
                    x_split = x.split("\t")
                    if head_type == 0:
                        phone, tone, ws, pw, stype, word = x_split
                    elif head_type == 1:
                        phone, tone, ws, pw, stype, word, unit = x_split
                    elif head_type == 2:
                        phone, tone, ws, pw, stype, word, alignment = x_split
                    elif head_type == 3:
                        phone, tone, ws, pw, stype, word, unit, alignment = x_split

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)
                    if head_type == 2 or head_type == 3:
                        alignments.append(float(alignment))

                    if tone == "0":
                        wordsegs.append("S")
                    else:
                        if word != "":
                            if pw == "1":
                                wordsegs.append("S")
                            else:
                                wordsegs.append("B")
                        else:
                            if pw == "0":
                                wordsegs.append("M")
                            else:
                                wordsegs.append("E")
                    wordseg_ids.append(self.wordseg_to_int[wordsegs[-1]])

            phone_ids, tone_ids = np.array(phone_ids), np.array(tone_ids)
            wordseg_ids = np.array(wordseg_ids)
            alignments = np.array(alignments)

            return (
                np.stack([phone_ids, tone_ids, wordseg_ids]),
                phones,
                tones,
                wordsegs,
                alignments,
            )

        except Exception as exc:
            logger.warning("error on convert_tacolab_to_text_id_infer", exc_info=exc)
            return None


def get_duration_frames_wds(alignments, phonemes, hop_ms):
    start_duration = 0
    accum_frames = 0
    phoneme_frames = []

    if len(alignments) != len(phonemes):
        return None

    i = 0
    for j, phoneme in enumerate(phonemes):
        accum_frames_temp = int(round((alignments[i] - start_duration) / hop_ms))
        frames = accum_frames_temp - accum_frames
        accum_frames = accum_frames_temp
        phoneme_frames.append(frames)
        i += 1
    return phoneme_frames
