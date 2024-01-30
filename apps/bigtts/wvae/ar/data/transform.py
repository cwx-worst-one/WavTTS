import logging
import pickle
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from apps.bigtts.wvae.ar.data.frontend import (
    phone_to_int,
    phonetone_to_int,
    tone_to_int,
)
from samantha.dataio.lite.transform import ItemTransformBase
from samantha.dataio.lite.utils.lang import is_chinese_char, is_english_spanish_char
from samantha.dataio.lite.utils.parquet import get_meta_obj
from samantha.dataio.lite.utils.punctuation import punctuation_all
from samantha.dataio.remote_io import load_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ContinuousTTSLangSpkSerTransform(ItemTransformBase):
    def __init__(
        self,
        use_lang_id=False,
        use_spk_id=False,
        spk2id=None,
        lang2id=None,
        use_code_switch_data=True,
        get_lang_by_tacolab=False,
        en_foreigner_list=None,
        bpe_dir=None,
        max_length=4096,
        use_bpe=False,
        use_extra_tag=False,
        spk2tag=None,
        tokenizer_type="llama",
        use_foreigner_data=True,
        use_lang_cfg=False,
        input_type="2dim",
        spk_cfg_rate=0.0,
        lang_cfg_rate=0.0,
        use_sp=True,
        refenc_cfg_rate=0.0,
    ):
        super().__init__()
        self.max_length = max_length
        self.use_bpe = use_bpe
        self.use_extra_tag = use_extra_tag

        logger.info(f"dataset/use_extra_tag: {use_extra_tag}")

        # lang
        self.use_lang_id = use_lang_id
        if self.use_lang_id:
            self.lang2id = load_json(lang2id)
            logger.info(f"Loaded lang2id from {lang2id}")
        else:
            self.lang2id = None
        logger.info(f"{self.lang2id=}")

        # spk
        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            self.spk2id = load_json(spk2id)
            logger.info(f"Loaded spk2id from {spk2id}")
        else:
            self.spk2id = None
        # print("self.spk2id: ", self.spk2id)

        # phone/tone
        self.phone_to_int = phone_to_int
        self.tone_to_int = tone_to_int
        self.phonetone_to_int = phonetone_to_int

        logger.info(f"{self.phone_to_int=}")
        logger.info(f"{self.tone_to_int=}")
        logger.info(f"{self.phonetone_to_int=}")

        # for cross-lingual
        self.use_code_switch_data = use_code_switch_data
        self.get_lang_by_tacolab = get_lang_by_tacolab

        self.en_foreigner_list = None
        if en_foreigner_list:
            self.en_foreigner_list = [
                x.strip() for x in open(en_foreigner_list).readlines()
            ]
        self.use_foreigner_data = use_foreigner_data
        if not self.use_foreigner_data:
            assert self.en_foreigner_list != None
        self.use_lang_cfg = use_lang_cfg

        if self.use_extra_tag:
            self.tag_dict = load_json(spk2tag)

        self.input_type = input_type

        # cfg
        self.spk_cfg_rate = spk_cfg_rate
        assert 0 <= self.spk_cfg_rate <= 1
        self.lang_cfg_rate = lang_cfg_rate
        assert 0 <= self.lang_cfg_rate <= 1

        # use_sp
        self.use_sp = use_sp

        self.refenc_cfg_rate = refenc_cfg_rate
        assert 0 <= self.refenc_cfg_rate <= 1

    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for fn in self.transforms:
            item = fn(item)
        return item

    @property
    def transforms(self) -> List[Callable]:
        return [self.process_meta, self.get_text_wavid]

    def process_meta(self, sample):
        meta_obj = get_meta_obj(sample)
        item = {}
        item["labels"] = str(meta_obj.get("labels", ""))
        item["speaker_name"] = str(meta_obj.get("speaker_id", ""))
        item["snr"] = str(meta_obj.get("snr", "10.0"))
        item["mos"] = str(meta_obj.get("mos", "5.0"))
        item["rms_stats_rms_max"] = "-1"
        item["speaker_similarity_min"] = "1.0"
        sample.update(item)
        return sample

    def get_text_wavid(self, sample):
        bn = pickle.loads(sample["bns"])
        text = sample["text"]
        labels = sample["labels"]
        utt_id = sample["__key__"]
        url = sample["__data_url__"]
        wav = np.frombuffer(sample["wav"][44:], dtype=np.int16) / 32768.0
        wav = wav.astype(np.float32)

        data_dict = dict()

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split("\n")))

        if not self.use_sp:
            labels = self.remove_replace_sp(labels, utt_id)
            if labels is None:
                return None

        # text_id, [text_len]
        text_id_phones_tones = self.convert_tacolab_to_text_id(labels, url)
        if text_id_phones_tones is None:
            logger.warning(
                f"{url} {utt_id}: convert_tacolab_to_text_id failed, {labels}"
            )
            return None
        else:
            if self.input_type == "2dim":
                text_id, phones, tones = text_id_phones_tones
            elif self.input_type == "1dim":
                text_id, phones, tones, phonetone_ids, phonetones = text_id_phones_tones
                # print("phonetone_ids: ", phonetone_ids)
            else:
                raise NotImplementedError
        if self.use_extra_tag:
            dataset_name = sample.get("dataset_name")
            speaker_name = sample.get("speaker_name")
            if dataset_name:
                if dataset_name in self.tag_dict:
                    spk_key = dataset_name
                else:
                    if speaker_name:
                        spk_key = "/".join([dataset_name, speaker_name])
                    else:
                        spk_key = dataset_name
            else:
                if speaker_name:
                    spk_key = speaker_name
                else:
                    logger.warning(
                        f"{url} {utt_id}: No speaker name or No dataset name, use 'default' as spk_key"
                    )
                    spk_key = "default"
            if spk_key not in self.tag_dict:
                logger.warning(
                    f"{utt_id} {spk_key}: spk_key not in spk2tag, use 'default' as spk_key"
                )
                spk_key = "default"

            tag_id = self.tag_dict[spk_key]
        else:
            tag_id = None

        # pad eos
        text_id = np.concatenate([text_id, np.ones([text_id.shape[0], 1])], axis=-1)

        data_dict["phone"] = text_id[0, :]
        data_dict["tone"] = text_id[1, :]

        data_dict["phonetone"] = None
        if self.input_type == "1dim":
            data_dict["phonetone"] = np.concatenate(
                [phonetone_ids, np.ones([1])], axis=0
            )

        # lang seq
        lang_seq = None
        if self.use_lang_id:
            if self.get_lang_by_tacolab:
                lang_key = self.get_lang(labels)
            else:
                lang_key = self.get_lang_by_text(text.encode("utf-8"))
            if self.en_foreigner_list:
                speaker_name = sample.get("speaker_name")
                if speaker_name:
                    if (
                        speaker_name.decode() in self.en_foreigner_list
                        and lang_key == "en"
                    ):
                        lang_key = "en_foreigner"
                        if not self.use_foreigner_data:
                            print(f"{utt_id}, {text}: Wrong data, Foreigner data.")
                            return None
            if lang_key == None or lang_key not in self.lang2id.keys():
                print(f"{utt_id}, {text}, {lang_key}: Wrong lang_key")
                return None
            lang_id = self.lang2id[lang_key]

            if self.lang_cfg_rate > 0:
                if np.random.rand() < self.lang_cfg_rate:
                    lang_id = self.lang2id["default"]

            lang_id += 1
            lang_seq = np.asarray([lang_id] * bn.shape[0])

        # spk_seq
        spk_seq = None
        if self.use_spk_id:
            dataset_name = sample.get("dataset_name")
            speaker_name = sample.get("speaker_name")
            if dataset_name:
                if dataset_name in self.spk2id:
                    spk_key = dataset_name
                else:
                    if speaker_name:
                        spk_key = "/".join([dataset_name, speaker_name])
                    else:
                        spk_key = dataset_name
            else:
                if speaker_name:
                    spk_key = speaker_name
                else:
                    logger.warning(
                        f"{url} {utt_id}: No speaker name or No dataset name, use 'default' as spk_key"
                    )
                    spk_key = "default"
            if spk_key not in self.spk2id:
                logger.warning(
                    f"{utt_id} {spk_key}: spk_key not in spk2id, use 'default' as spk_key"
                )
                spk_key = "default"

            # spk_cfg
            if spk_key != "default":
                if self.spk_cfg_rate > 0:
                    if np.random.rand() < self.spk_cfg_rate:
                        spk_key = "default"

            spk_id = self.spk2id[spk_key]
            spk_id += 1
            spk_seq = np.asarray([spk_id] * bn.shape[0])

        spk_embd_mask = 1
        if self.refenc_cfg_rate > 0:
            if spk_key == "default":
                spk_embd_mask = 0
            else:
                if np.random.rand() < self.refenc_cfg_rate:
                    spk_embd_mask = 0

        # stop_token
        stop_token = np.zeros([text_id.shape[1] + bn.shape[0]])
        stop_token[-1] = 1

        # bpe_id
        if self.use_bpe:
            byt5 = pickle.loads(sample.get("byt5"))
            # bpe_seq = np.asarray(self.bpe_tokenizer(
            #     text, truncation=True, max_length=self.max_length,
            # ).input_ids)
        else:
            # bpe_seq = None
            byt5 = None

        data_dict.update(
            {
                "stop_token": stop_token,
                "bn": bn,
                "utt_id": utt_id,
                "lang_seq": lang_seq,
                "spk_seq": spk_seq,
                "byt5": byt5,
                "tag_id": tag_id,
                "wav": wav,
                "spk_embd_mask": spk_embd_mask,
            }
        )
        return data_dict

    def convert_v3_to_v1(self, tacolab, url):
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
                logger.warning(f"Wrong tacolab {x_split} {url=}")
                return None
            tacolab_v1.append("\t".join([phone, tone, "0.0 0.0 0.0 1.0", ws, pw]))
        return tacolab_v1

    def get_lang_by_text(self, text):
        text = text.replace("'", "")
        # en, zh
        en_word_cnt = 0
        zh_char_cnt = 0
        i = 0
        while i < len(text):
            x = text[i]
            if x in punctuation_all:  # punc
                i += 1
                continue
            elif is_chinese_char(x):  # zh
                zh_char_cnt += 1
                i += 1
            elif is_english_spanish_char(x):  # en with little spanish
                i += 1
                if i >= len(text):
                    en_word_cnt += 1
                    break
                while is_english_spanish_char(text[i]):
                    i += 1
                    if i >= len(text):
                        break
                en_word_cnt += 1
                continue
            else:  # blank
                if not (text[i] == " " or text[i].isdigit()):
                    print("text[i]: ", text[i])
                    return None
                i += 1

        lang = "en"
        if zh_char_cnt > en_word_cnt:
            lang = "zh"

        if not self.use_code_switch_data:
            if not (zh_char_cnt == 0 or en_word_cnt == 0):
                if self.use_lang_cfg:
                    return "default"
                else:
                    return None
        return lang

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
        else:
            lang = "en"
        return lang

    def remove_replace_sp(self, labels, utt_id):
        new_labels = []
        for label in labels:
            if len(label.split("\t")) == 7:
                phone, tone, ws, pw, stype, word, unit = label.split("\t")
                if phone == "sp":
                    if word == "":
                        pass
                    else:
                        assert word in punctuation_all, word
                        phone = word
                new_label = "\t".join([phone, tone, ws, pw, stype, word, unit])
            elif len(label.split("\t")) == 6:
                phone, tone, ws, pw, stype, word = label.split("\t")
                if phone == "sp":
                    if word == "":
                        pass
                    else:
                        assert word in punctuation_all, word
                        phone = word
                new_label = "\t".join([phone, tone, ws, pw, stype, word])
            elif len(label.split("\t")) == 5:
                # if label.split('\t')[0] == 'sp':
                print(f"labels: {utt_id} \n{labels}")
                new_label = label
            else:
                logger.warning(f"{utt_id} labels wrong format, {labels}, skip")
                return None
            new_labels.append(new_label)

        return new_labels

    def convert_tacolab_to_text_id(self, tacolab, url):
        try:
            lang = self.get_lang(tacolab)
            phone_ids = []
            tone_ids = []
            phonetone_ids = []
            phones = []
            tones = []
            phonetones = []

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
                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    assert (
                        phone + "_" + tone in self.phonetone_to_int
                    ), f"{phone + '_' + tone} not in phonetone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phonetone_ids.append(self.phonetone_to_int[phone + "_" + tone])
                    phones.append(phone)
                    tones.append(tone)
                    phonetones.append(phone + "_" + tone)
                    if phone[:2] == "C0":
                        if unit in ["S", "E"]:
                            phone_ids.append(self.phone_to_int["syl_sep"])
                            tone_ids.append(self.tone_to_int["syl_sep"])
                            phonetone_ids.append(
                                self.phonetone_to_int["syl_sep_syl_sep"]
                            )
                            phones.append("syl_sep")
                            tones.append("syl_sep")
                            phonetones.append("syl_sep_syl_sep")
                            if ws in ["S", "E"]:
                                phone_ids.append(self.phone_to_int["zh_word_sep"])
                                tone_ids.append(self.tone_to_int["zh_word_sep"])
                                phonetone_ids.append(
                                    self.phonetone_to_int["zh_word_sep_zh_word_sep"]
                                )
                                phones.append("zh_word_sep")
                                tones.append("zh_word_sep")
                                phonetones.append("zh_word_sep_zh_word_sep")
                    elif phone[:2] == "E0":
                        if pw != "0":
                            phone_ids.append(self.phone_to_int["en_word_sep"])
                            tone_ids.append(self.tone_to_int["en_word_sep"])
                            phonetone_ids.append(
                                self.phonetone_to_int["en_word_sep_en_word_sep"]
                            )
                            phones.append("en_word_sep")
                            tones.append("en_word_sep")
                            phonetones.append("en_word_sep_en_word_sep")
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

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    assert (
                        phone + "_" + tone in self.phonetone_to_int
                    ), f"{phone + '_' + tone} not in phonetone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phonetone_ids.append(self.phonetone_to_int[phone + "_" + tone])
                    phones.append(phone)
                    tones.append(tone)
                    phonetones.append(phone + "_" + tone)
                    if phone[:2] == "C0":
                        if unit in ["S", "E"]:
                            phone_ids.append(self.phone_to_int["syl_sep"])
                            tone_ids.append(self.tone_to_int["syl_sep"])
                            phonetone_ids.append(
                                self.phonetone_to_int["syl_sep_syl_sep"]
                            )
                            phones.append("syl_sep")
                            tones.append("syl_sep")
                            phonetones.append("syl_sep_syl_sep")
                            if ws in ["S", "E"]:
                                phone_ids.append(self.phone_to_int["zh_word_sep"])
                                tone_ids.append(self.tone_to_int["zh_word_sep"])
                                phonetone_ids.append(
                                    self.phonetone_to_int["zh_word_sep_zh_word_sep"]
                                )
                                phones.append("zh_word_sep")
                                tones.append("zh_word_sep")
                                phonetones.append("zh_word_sep_zh_word_sep")
                    elif phone[:2] == "E0":
                        if pw != "0":
                            phone_ids.append(self.phone_to_int["en_word_sep"])
                            tone_ids.append(self.tone_to_int["en_word_sep"])
                            phonetone_ids.append(
                                self.phonetone_to_int["en_word_sep_en_word_sep"]
                            )
                            phones.append("en_word_sep")
                            tones.append("en_word_sep")
                            phonetones.append("en_word_sep_en_word_sep")
            elif lang == "en":
                if len(tacolab[0].split("\t")) != 5:
                    tacolab = self.convert_v3_to_v1(tacolab, url)
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split("\t")[0] == "sil":
                        continue
                    x_split = x.split("\t")
                    phone, tone, _, ws, pw = x.split("\t")
                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    assert (
                        phone + "_" + tone in self.phonetone_to_int
                    ), f"{phone + '_' + tone} not in phonetone set"
                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phonetone_ids.append(self.phonetone_to_int[phone + "_" + tone])
                    phones.append(phone)
                    tones.append(tone)
                    phonetones.append(phone + "_" + tone)
                    if pw != "0":
                        phone_ids.append(self.phone_to_int["en_word_sep"])
                        tone_ids.append(self.tone_to_int["en_word_sep"])
                        phonetone_ids.append(
                            self.phonetone_to_int["en_word_sep_en_word_sep"]
                        )
                        phones.append("en_word_sep")
                        tones.append("en_word_sep")
                        phonetones.append("en_word_sep_en_word_sep")
            phone_ids, tone_ids = np.array(phone_ids), np.array(tone_ids)
            phonetone_ids = np.asarray(phonetone_ids)
            if self.input_type == "2dim":
                return np.stack([phone_ids, tone_ids]), phones, tones
            elif self.input_type == "1dim":
                return (
                    np.stack([phone_ids, tone_ids]),
                    phones,
                    tones,
                    phonetone_ids,
                    phonetones,
                )
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            return None
