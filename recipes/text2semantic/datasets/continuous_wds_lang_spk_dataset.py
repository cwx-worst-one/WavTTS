import logging
import math
import pickle
import sys
import os 
import random

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import LlamaTokenizer, T5Tokenizer, AutoTokenizer

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset.ra_wds import WebDataset
from samantha.utils.hparams import DotDict

from recipes.text2semantic.utils.remote_io import load_json
from transformers import LlamaTokenizer
from zhon.hanzi import punctuation
import string
punctuation_all = punctuation + string.punctuation

import traceback

from recipes.text2semantic.datasets.frontend import phone_to_int, tone_to_int
from collections import defaultdict

logger = logging.getLogger(__name__)



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ContinuousTTSLangSpkDataset(IterableDataset):
    def __init__(self,
        wds_urls,
        drop_last=False,
        batcher_config=None,
        use_lang_id=False,
        use_spk_id=False,
        spk2id=None,
        lang2id=None,
        use_code_switch_data=True,
        get_lang_by_tacolab=False,
        bpe_dir=None, 
        max_length=4096,
        use_bpe=False,
        use_extra_tag=False,
        wds2tag=None,
        tokenizer_type="llama",
        ):

        self.wds = (
            WebDataset(
                urls=wds_urls,
                resampled=True,
                skip_instance_cache=True,
            )
            .decode()
            .shuffle(2048)
            .map(self.get_text_wavid)
        )

        self.max_length = max_length
        self.use_bpe = use_bpe
        self.use_extra_tag = use_extra_tag

        print(f"dataset/use_extra_tag: {use_extra_tag}")

        self.drop_last = drop_last
        self.batcher = BucketBatcher(**batcher_config)

        # lang
        self.use_lang_id = use_lang_id
        if self.use_lang_id:
            self.lang2id = load_json(lang2id)
            print(f"Loaded lang2id from {lang2id}")
        else:
            self.lang2id = None
        print("self.lang2id: ", self.lang2id)

        # spk
        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            self.spk2id = load_json(spk2id)
            print(f"Loaded spk2id from {spk2id}")
        else:
            self.spk2id = None
        # print("self.spk2id: ", self.spk2id)

        # phone/tone
        self.phone_to_int = phone_to_int
        self.tone_to_int = tone_to_int
            
        logger.info(f"{self.phone_to_int=}, {self.tone_to_int=}")

        # for cross-lingual
        self.use_code_switch_data = use_code_switch_data
        self.get_lang_by_tacolab = get_lang_by_tacolab

    
        if self.use_extra_tag:
            self.tag_dict = load_json(wds2tag)

        if self.use_bpe:
            print(f'##### Using BPE #####')
            if tokenizer_type == "flan-T5-large":
                self.bpe_tokenizer = T5Tokenizer.from_pretrained(bpe_dir)
            elif tokenizer_type == "byte-T5-base":
                self.bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_dir)
            elif tokenizer_type == "llama":
                self.bpe_tokenizer = LlamaTokenizer.from_pretrained(bpe_dir)
            else:
                raise NotImplementedError
        else:
            self.bpe_tokenizer = None

    def get_text_wavid(self, sample):

        bn = pickle.loads(sample["bns"])
        text = sample["text"]
        lab = sample["labels"]
        utt_id = sample["__key__"]
        url = sample["__url__"]

        data_dict = dict()

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        labels = lab.decode()
        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split('\n')))

        # text_id, [text_len]
        text_id_phones_tones = self.convert_tacolab_to_text_id(labels)
        if text_id_phones_tones is None:
            logger.warning(f"{utt_id} convert_tacolab_to_text_id failed ...")
            return None
        else:
            text_id, phones, tones = text_id_phones_tones
        
        if self.use_extra_tag:
            tag_id = int(self.tag_dict.get(url, 0))
            if tag_id == 0:
                print(f"Warning: no tag id found for {url}")
        else:
            tag_id = None

        # pad eos
        text_id = np.concatenate([text_id, np.ones([text_id.shape[0], 1])], axis=-1)

        data_dict["phone"] = text_id[0, :]
        data_dict["tone"] = text_id[1, :]

        # lang seq
        lang_seq = None        
        if self.use_lang_id:
            if self.get_lang_by_tacolab:
                lang_key = self.get_lang(labels)
            else:
                lang_key = self.get_lang_by_text(text)
            if lang_key == None or lang_key not in ['zh', 'en']:
                print(f"{text}: Wrong lang_key")
                return None
            lang_id = self.lang2id[lang_key]
            lang_id += 1
            lang_seq = np.asarray([lang_id] * bn.shape[0])

        # spk_seq
        spk_seq = None        
        if self.use_spk_id:
            dataset_name = sample.get("dataset_name")
            speaker_name = sample.get("speaker_name")
            if not dataset_name or not speaker_name:
                # print(f"{utt_id}: No speaker name")
                spk_id = self.spk2id["default"]
            else:
                dataset_name = dataset_name.decode()
                speaker_name = speaker_name.decode()
                spk_key = '/'.join([dataset_name, speaker_name])
                if spk_key in self.spk2id:
                    spk_id = self.spk2id[spk_key]
                else:
                    print(f"{utt_id}: speaker {spk_key} not in dict, will use default spkID")
                    spk_id = self.spk2id["default"]
            spk_id += 1
            spk_seq = np.asarray([spk_id] * bn.shape[0])

        # stop_token
        stop_token = np.zeros([text_id.shape[1] + bn.shape[0]])
        stop_token[-1] = 1
    
        # bpe_id
        if self.use_bpe:
            bpe_seq = np.asarray(self.bpe_tokenizer(
                text, truncation=True, max_length=self.max_length,
            ).input_ids)
        else:
            bpe_seq = None

        data_dict.update({
            "stop_token": stop_token,
            "bn": bn,
            "utt_id": utt_id,
            "lang_seq": lang_seq,
            "spk_seq": spk_seq,
            "bpe_seq": bpe_seq,
            "tag_id": tag_id
            })
        return data_dict

    def __iter__(self):
        for item in self.wds:
            batch = self.batcher.collate_batch(item)
            if batch:
                yield batch
        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield batch

    def convert_v3_to_v1(self, tacolab):
        tacolab_v1 = []
        # en, zh
        if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
            tacolab = tacolab[1:]
        for x in tacolab:
            x_split = x.split('\t')
            if len(x_split) == 7:
                phone, tone, ws, pw, stype, word, _ = x_split
            elif len(x_split) == 6:
                phone, tone, ws, pw, stype, word = x_split
            else:
                print("Wrong tacolab", x_split)
                return None
            tacolab_v1.append('\t'.join([phone, tone, '0.0 0.0 0.0 1.0', ws, pw]))
        return tacolab_v1

    def is_english_char(self, char):
        if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a'):
            return True
        else:
            return False

    def is_english_spanish_char(self, char):
        special_Spanish_chars_list = ['á', 'é', 'í', 'ó', 'ú', 'Á', 'É', 'Í', 'Ó', 'Ú', 'ñ', 'Ñ', '¡', '¿', 'ü', 'Ü']
        if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a') or char in special_Spanish_chars_list:
            return True
        else:
            return False

    def get_lang_by_text(self, text):
        text = text.replace('\'', '')
        # en, zh
        len_en_word = 0
        len_zh_char = 0
        i = 0
        while i < len(text):
            x = text[i]
            if x in punctuation_all: # punc
                i += 1
                continue
            elif u'\u4e00' <= x <= u'\u9fff': # zh
                len_zh_char += 1
                i += 1
            elif self.is_english_spanish_char(x): # en
                i += 1
                if i >= len(text):
                    len_en_word += 1
                    break
                while self.is_english_spanish_char(text[i]):
                    i += 1
                    if i >= len(text):
                        break
                len_en_word += 1
                continue
            else: # blank
                if not (text[i] == " " or text[i].isdigit()):
                    print("text[i]: ", text[i])
                    return None
                i += 1

        lang = 'en'
        if len_zh_char > len_en_word:
            lang = 'zh'

        if not self.use_code_switch_data:
            if not (len_zh_char == 0 or len_en_word == 0):
                return None

        return lang

    def get_lang(self, tacolab):
        if len(tacolab[0].split('\t')) != 5:
            if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                tacolab = tacolab[1:]
        prefix_phn_list = [x.split('\t')[0][:2] for x in tacolab]
        if 'C0' in prefix_phn_list:
            if 'E0' in prefix_phn_list:
                lang = 'zh_en'
            else:
                lang = 'zh'
        else:
            lang = 'en'
        return lang

    def convert_tacolab_to_text_id(self, tacolab):
        try:
            lang = self.get_lang(tacolab)
            phone_ids = []
            tone_ids = []
            phones = []
            tones = []
            if lang == 'zh':
                assert len(tacolab[0].split('\t')) == 7, (len(tacolab[0].split('\t')), tacolab[0])
                if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                    tacolab = tacolab[1:]
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, ws, pw, stype, word, unit = x_split
                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)
                    if phone[:2] == "C0":
                        if unit in ['S', 'E']:
                            phone_ids.append(self.phone_to_int["syl_sep"])
                            tone_ids.append(self.tone_to_int["syl_sep"])
                            phones.append("syl_sep")
                            tones.append("syl_sep")
                            if ws in ["S", "E"]:
                                phone_ids.append(self.phone_to_int["zh_word_sep"])
                                tone_ids.append(self.tone_to_int["zh_word_sep"])
                                phones.append("zh_word_sep")
                                tones.append("zh_word_sep")
                    elif phone[:2] == "E0":
                        if pw != "0":
                            phone_ids.append(self.phone_to_int["en_word_sep"])
                            tone_ids.append(self.tone_to_int["en_word_sep"])
                            phones.append("en_word_sep")
                            tones.append("en_word_sep")
            elif lang == 'zh_en':
                assert len(tacolab[0].split('\t')) == 7 or len(tacolab[0].split('\t')) == 6, (len(tacolab[0].split('\t')), tacolab[0])
                if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword':
                    tacolab = tacolab[1:]
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    if len(x_split) == 7:
                        phone, tone, ws, pw, stype, word, unit = x_split
                    elif len(x_split) == 6:
                        phone, tone, ws, pw, stype, word = x_split
                    else:
                        print("Wrong tacolab", x_split)
                        return None

                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"

                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)
                    if phone[:2] == "C0":
                        if unit in ['S', 'E']:
                            phone_ids.append(self.phone_to_int["syl_sep"])
                            tone_ids.append(self.tone_to_int["syl_sep"])
                            phones.append("syl_sep")
                            tones.append("syl_sep")
                            if ws in ["S", "E"]:
                                phone_ids.append(self.phone_to_int["zh_word_sep"])
                                tone_ids.append(self.tone_to_int["zh_word_sep"])
                                phones.append("zh_word_sep")
                                tones.append("zh_word_sep")
                    elif phone[:2] == "E0":
                        if pw != "0":
                            phone_ids.append(self.phone_to_int["en_word_sep"])
                            tone_ids.append(self.tone_to_int["en_word_sep"])
                            phones.append("en_word_sep")
                            tones.append("en_word_sep")
            elif lang == 'en':
                if len(tacolab[0].split('\t')) != 5:
                    tacolab = self.convert_v3_to_v1(tacolab)
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    x_split = x.split('\t')
                    phone, tone, _, ws, pw = x.split('\t')
                    assert phone in self.phone_to_int, f"{phone} not in phone set"
                    assert tone in self.tone_to_int, f"{tone} not in tone set"
                    phone_ids.append(self.phone_to_int[phone])
                    tone_ids.append(self.tone_to_int[tone])
                    phones.append(phone)
                    tones.append(tone)
                    if pw != "0":
                        phone_ids.append(self.phone_to_int["en_word_sep"])
                        tone_ids.append(self.tone_to_int["en_word_sep"])
                        phones.append("en_word_sep")
                        tones.append("en_word_sep")
            phone_ids, tone_ids = np.array(phone_ids), np.array(tone_ids)
            return np.stack([phone_ids, tone_ids]), phones, tones
        except Exception as e:
            print(e)
            return None


class ContinuousCollator(object):
    def __init__(self, tokenizer_pad, block_sparse=False):
        self.pad = tokenizer_pad
        self.block_sparse = block_sparse

    def __call__(self, batches):
        results = []
        for item in batches:
            if item is not None:
                results.append(item)

        if len(results) == 0:
            return None
        ret_dict = defaultdict(list)

        text_lens = []
        bn_lens = []
        bpe_lens = []
        # utt_ids = []
        # tag_ids = []
        for x in results:
            text_lens.append(x["phone"].shape[0])
            bn_lens.append(x["bn"].shape[0])
            bpe_lens.append(x["bpe_seq"].shape[0])
            # utt_ids.append(x["utt_id"])
            # if x["tag_id"] is not None:
            #     tag_ids.append(x["tag_id"])

        max_text_len = max(text_lens)
        max_bn_len = max(bn_lens)
        max_bpe_len = max(bpe_lens)

        text_lens = torch.from_numpy(np.asarray(text_lens))
        bn_lens = torch.from_numpy(np.asarray(bn_lens))
        bpe_lens = torch.from_numpy(np.asarray(bpe_lens))

        max_seq_len = max(text_lens + bn_lens)

        ret_dict["text_lens"] = text_lens
        ret_dict["bn_lens"] = bn_lens
        ret_dict["bpe_lens"] = bpe_lens
        # ret_dict["utt_id"] = utt_ids
        # if len(tag_ids) > 0:
        #     ret_dict["tag_id"] = np.asarray(tag_ids)

        # length padding
        for x in results:
            for k, v in x.items():
                if v is None:
                    ret_dict[k] = None
                    continue
                if k == "__key__":
                    continue
                if k == "bn":
                    v = np.pad(
                        v,
                        ((0, max_bn_len - v.shape[0]), (0, 0)),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "stop_token":
                    v = np.pad(
                        v,
                        (0, max_seq_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "lang_seq":
                    v = np.pad(
                        v,
                        (0, max_bn_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "spk_seq":
                    v = np.pad(
                        v,
                        (0, max_bn_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "bpe_seq":
                    v = np.pad(
                        v,
                        (0, max_bpe_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k not in ["utt_id", "tag_id"]:
                    v = np.pad(
                        v,
                        (0, max_text_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                ret_dict[k].append(v)

        for k in ret_dict.keys():
            if k == "bn":
                ret_dict[k] = np.stack(ret_dict[k], axis=0) 
            elif k != "utt_id":
                if ret_dict[k] is not None:
                    ret_dict[k] = np.asarray(ret_dict[k])
            else:
                continue
            if ret_dict[k] is not None:
                try:
                    ret_dict[k] = torch.from_numpy(ret_dict[k])
                except Exception:
                    print("ret_dict[k]: ", ret_dict[k])

        to_long_list = ["phone", "tone", "stop_token", "lang_seq", "spk_seq", "bpe_seq", "tag_id"]
        for x in to_long_list:
            if x in ret_dict:
                if ret_dict[x] is not None:
                    ret_dict[x] = ret_dict[x].long()
        return ret_dict


if __name__ == "__main__":
    # pass
    # import torch
    # import torch.distributed as dist
    # import torch.utils.data

    # dist.init_process_group(backend="nccl")
    # urls = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/11labs/*/chunk*/*.tar hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/duibiao/*/chunk*/*.tar"
    # urls = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/chenyuanzhe/WFVAE_v2_fixtrim/librilight/chunk*/*.tar"
    # urls = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc_mix/BigSpeech_MIX-TTS/*/chunk*/*.tar"
    urls = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/duibiao/Tim_normal/chunk*/*.tar"

    from samantha.dataio.utils import parse_data_urls
    wds_urls = parse_data_urls(data_urls=urls)
    batcher_config = {
        "buckets": list(range(0, 6000, 100)),  # [0, 100, 200 ... 4000] 4000以上的可以先丢掉
        "dynamic_batch": True,
        "maximum_bucket_size": 200,
        "length_fn": "lambda x: x[\"stop_token\"].shape[0]" # seq.shape
    }
    dataset = ContinuousTTSLangSpkDataset(wds_urls, 
                                batcher_config=batcher_config,
                                drop_last=False,
                                use_lang_id=True,
                                lang2id="recipes/text2semantic/datasets/dict/lang2id.json",
                                use_spk_id=True,
                                spk2id="recipes/text2semantic/datasets/dict/spk2id.en_all_20230905.json")

    collector = ContinuousCollator(tokenizer_pad=0)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=None, collate_fn=collector)
    import tqdm

    # for item in tqdm.tqdm(dataset):
    for item in tqdm.tqdm(dataloader):
        # print(item)
        # lengths = item[-1]
        # batch_size = len(lengths)
        # print(batch_size, max(lengths), batch_size * max(lengths))
        exit()
