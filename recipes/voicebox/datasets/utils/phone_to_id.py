import logging

import numpy as np

from samantha.utils.common import is_float

from .frontend import lang_to_int, phone_to_int, tone_to_int, wordseg_to_int

logger = logging.getLogger(__name__)


class PhoneToId:
    def __init__(self) -> None:
        self.phone_to_int = phone_to_int
        self.tone_to_int = tone_to_int
        self.wordseg_to_int = wordseg_to_int
        self.lang_to_int = lang_to_int

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
            elif len(x_split) == 8:
                phone, tone, ws, pw, stype, word, _, _ = x_split
            else:
                logger.error(f"convert_v3_to_v1: Wrong tacolab {x_split}")
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

        prefix_phn_list = [x.split('\t')[0][:2] for x in tacolab]
        # 排序逻辑，取主语言
        result = max(set(prefix_phn_list), key=prefix_phn_list.count)
        if result == 'C0':
            return 'zh'
        elif result == 'E0':
            return 'en'
        elif result == 'JP':
            return 'jp'
        elif result == 'MX':
            return 'mx'
        elif result == 'BR':
            return 'br'
        elif result == 'ID':
            return 'id'
        elif result == 'DE':
            return 'de'
        elif result == 'FR':
            return 'fr'
        elif result == 'KO':
            return 'ko'

    def get_lang_split(self, tacolab):
        exist_lang = ['JP', 'MX', 'C0', 'E0', 'ID', 'BR', 'DE', 'FR', 'KO']
        lang2tacolab = []
        temp_list = []
        lang_flag = ''

        prefix_phn_list = [x.split('\t')[0][:2] for x in tacolab]

        for i, prefix_phn in enumerate(prefix_phn_list):
            if lang_flag == '':
                temp_list.append(tacolab[i])
                if prefix_phn in exist_lang:
                    lang_flag = prefix_phn
            elif prefix_phn == lang_flag or prefix_phn not in exist_lang:
                temp_list.append(tacolab[i])
            else:
                lang2tacolab.append([lang_flag, temp_list])
                temp_list = []
                lang_flag = prefix_phn
                temp_list.append(tacolab[i])
        lang2tacolab.append([lang_flag, temp_list])

        return lang2tacolab

    def convert_tacolab_to_text_id(self, tacolab, return_words=False):
        try:
            lang2tacolab = self.get_lang_split(tacolab)
            phone_ids = []
            tone_ids = []
            phones = []
            tones = []
            wordseg_ids = []
            wordsegs = []
            alignments = []
            words = []

            head_type_dict = {
                'phn\ttone\tws\tpwpp\tsentype\tword': 0,
                'phn\ttone\tws\tpwpp\tsentype\tword\tunit': 1,
                'phn\ttone\tws\tpwpp\tsentype\tword\talignment': 2,
                'phn\ttone\tws\tpwpp\tsentype\tword\tunit\talignment': 3,
            }
            for sub_tacolab in lang2tacolab:
                lang, tacolab = sub_tacolab[0], sub_tacolab[1]

                if lang in ['C0', 'JP', 'MX', 'ID', 'BR', 'DE', 'FR', 'KO']:
                    # assert tacolab[0] in head_type_dict, (len(tacolab[0].split('\t')), tacolab[0])
                    # head_type = head_type_dict[tacolab[0]]
                    # tacolab = tacolab[1:]
                    if 'phn\ttone\tws\tpwpp\tsentype\tword' in tacolab[0]:
                        assert tacolab[0] in head_type_dict, (len(tacolab[0].split('\t')), tacolab[0])
                        head_type = head_type_dict[tacolab[0]]
                        tacolab = tacolab[1:]
                    else:
                        if len(tacolab[0].split('\t')) == 6:
                            head_type = 0
                        elif len(tacolab[0].split('\t')) == 8:
                            head_type = 3
                        elif is_float(tacolab[0].split('\t')[-1]):
                            head_type = 2
                        else:
                            head_type = 1

                    assert head_type in [1, 3]

                    for i in range(len(tacolab)):
                        x = tacolab[i]
                        if i != 0 and x.split('\t')[0] == 'sil':
                            continue
                        x_split = x.split('\t')
                        if head_type == 0:
                            phone, tone, ws, pw, stype, word = x_split
                        elif head_type == 1:
                            phone, tone, ws, pw, stype, word, unit = x_split
                        elif head_type == 2:
                            phone, tone, ws, pw, stype, word, alignment = x_split
                        elif head_type == 3:
                            phone, tone, ws, pw, stype, word, unit, alignment = x_split

                        if phone.startswith('JP_') or phone.startswith('MX_') or phone.startswith(
                                'ID_') or phone.startswith('BR_') or phone.startswith('DE_') or phone.startswith(
                            'FR_') or phone.startswith('KO_'):
                            phone_lang, phone = phone.split('_')

                        elif lang != 'C0' and (phone == 'pau' or phone == 'sp'):
                            # ['C0', 'JP', 'MX', 'ID', 'BR']
                            # ['id_sp', 'br_sp']
                            if lang == 'JP':
                                phone = 'jp_sp'
                            elif lang in ['MX']:
                                phone = 'mx_sp'
                            elif lang in ['ID']:
                                phone = 'id_sp'
                            elif lang in ['BR']:
                                phone = 'br_sp'
                            elif lang in ['DE']:
                                phone = 'de_sp'
                            elif lang in ['FR']:
                                phone = 'fr_sp'
                            elif lang in ['KO']:
                                phone = 'ko_sp'

                        assert phone in self.phone_to_int, f"{phone} not in phone set"
                        assert tone in self.tone_to_int, f"{tone} not in tone set"

                        if phone.startswith('E0'):
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
                        else:
                            wordseg_ids.append(self.wordseg_to_int[ws])

                        phone_ids.append(self.phone_to_int[phone])
                        tone_ids.append(self.tone_to_int[tone])
                        phones.append(phone)
                        tones.append(tone)
                        words.append(word)
                        if head_type == 2 or head_type == 3:
                            alignments.append(float(alignment))
                elif lang == 'E0':
                    if 'phn\ttone\tws\tpwpp\tsentype\tword' in tacolab[0]:
                        assert tacolab[0] in head_type_dict, (len(tacolab[0].split('\t')), tacolab[0])
                        head_type = head_type_dict[tacolab[0]]
                        tacolab = tacolab[1:]
                    else:
                        if len(tacolab[0].split('\t')) == 6:
                            head_type = 0
                        elif len(tacolab[0].split('\t')) == 8:
                            head_type = 3
                        elif is_float(tacolab[0].split('\t')[-1]):
                            head_type = 2
                        else:
                            head_type = 1

                    for i in range(len(tacolab)):
                        x = tacolab[i]
                        if i != 0 and x.split('\t')[0] == 'sil':
                            continue
                        x_split = x.split('\t')
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
                        words.append(word)
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

            if return_words:
                return (
                    np.stack([phone_ids, tone_ids, wordseg_ids]),
                    phones,
                    tones,
                    wordsegs,
                    alignments,
                    words
                )

            return (
                np.stack([phone_ids, tone_ids, wordseg_ids]),
                phones,
                tones,
                wordsegs,
                alignments,
            )

        except Exception as e:
            logger.error(f"convert_tacolab_to_text_id exception: {e}")
            return None

    def convert_tacolab_to_text_id_infer(self, tacolab):
        # try:
        # print("###########")
        # print(tacolab)
        lang2tacolab = self.get_lang_split(tacolab)
        phone_ids = []
        tone_ids = []
        phones = []
        tones = []
        wordseg_ids = []
        wordsegs = []
        alignments = []

        head_type_dict = {
            'phn\ttone\tws\tpwpp\tsentype\tword': 0,
            'phn\ttone\tws\tpwpp\tsentype\tword\tunit': 1,
            'phn\ttone\tws\tpwpp\tsentype\tword\talignment': 2,
            'phn\ttone\tws\tpwpp\tsentype\tword\tunit\talignment': 3,
        }

        for sub_tacolab in lang2tacolab:
            lang, tacolab = sub_tacolab[0], sub_tacolab[1]

            if lang in ['C0', 'JP', 'MX', 'ID', 'BR', 'DE', 'FR', 'KO']:
                assert len(tacolab[0].split('\t')) == 7, (len(tacolab[0].split('\t')), tacolab[0])
                if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                    tacolab = tacolab[1:]
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, ws, pw, stype, word, unit = x_split

                    if phone.startswith('JP_') or phone.startswith('MX_') or phone.startswith(
                            'ID_') or phone.startswith('BR_') or phone.startswith('DE_') or phone.startswith(
                            'FR_') or phone.startswith('KO_'):
                        phone_lang, phone = phone.split('_')
                    # elif lang != 'C0' and (phone == 'pau' or phone == 'sp'):
                    #     phone = 'jp_sp'

                    elif lang != 'C0' and (phone == 'pau' or phone == 'sp'):
                        # ['C0', 'JP', 'MX', 'ID', 'BR']
                        # ['id_sp', 'br_sp']
                        if lang == 'JP':
                            phone = 'jp_sp'
                        elif lang in ['MX']:
                            phone = 'mx_sp'
                        elif lang in ['ID']:
                            phone = 'id_sp'
                        elif lang in ['BR']:
                            phone = 'br_sp'
                        elif lang in ['DE']:
                            phone = 'de_sp'
                        elif lang in ['FR']:
                            phone = 'fr_sp'
                        elif lang in ['KO']:
                            phone = 'ko_sp'

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"

                    if phone.startswith('E0'):
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
                    else:
                        wordseg_ids.append(self.wordseg_to_int[ws])

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)

            elif lang == 'E0':
                if 'phn\ttone\tws\tpwpp\tsentype\tword' in tacolab[0]:
                    assert tacolab[0] in head_type_dict, (len(tacolab[0].split('\t')), tacolab[0])
                    head_type = head_type_dict[tacolab[0]]
                    tacolab = tacolab[1:]
                else:
                    if len(tacolab[0].split('\t')) == 6:
                        head_type = 0
                    elif len(tacolab[0].split('\t')) == 8:
                        head_type = 3
                    elif is_float(tacolab[0].split('\t')[-1]):
                        head_type = 2
                    else:
                        head_type = 1

                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
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
        # print("========>")
        # print(phone_ids)
        # print(tone_ids)
        return (
            np.stack([phone_ids, tone_ids, wordseg_ids]),
            phones,
            tones,
            wordsegs,
            alignments
        )


def get_duration_frames(durations, phonemes, hop_ms):
    start_duration = 0
    accum_frames = 0
    phoneme_frames = []

    # remove sil in durations
    format_durations = []
    for i, item in enumerate(durations):
        if (i != 0 and i != len(durations) - 1) and item[0] == "sil":
            continue
        format_durations.append(item)

    if len(format_durations) != len(phonemes):
        return None

    i = 0
    for j, phoneme in enumerate(phonemes):
        accum_frames_temp = int(
            round((format_durations[i][1] - start_duration) / hop_ms)
        )
        frames = accum_frames_temp - accum_frames
        accum_frames = accum_frames_temp
        phoneme_frames.append(frames)
        i += 1
    return phoneme_frames


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
