'''
single test of char2phone
'''
# pylint: disable='line-too-long', 'import-error', 'invalid-name'
import re
import pickle
import random
import numpy as np
import torch

try:
    from g2p_en import G2p
except ImportError:
    G2p = None
from pypinyin import lazy_pinyin, Style
from dataloader import FalconReader
from core.utils import dist_hdfs_get
from core.dataset.cuda import to_cuda
from core.dataset.preprocess import PhoneCollate, Char2Phone, PhonePredict, FbankCollate


def get_list(num=10):
    '''get single item list'''
    path = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/train_sub0'
    reader = FalconReader(path)
    _keys = reader.list_keys()
    vals = reader.read_many(list(range(num)))
    vals = sum(vals, [])
    item_list = []
    for val in vals:
        item = pickle.loads(val)
        item_list.append({'label': item['label']})
    return item_list


class BaseChar2Phone:
    '''Baseline char2phone'''

    def __init__(
        self,
        in_key='label',
        out_key='phone',
        no_need_key='fbank',
        phn_repeat=4,
        acoustic_ds=4,
        mask_rate=0.0,
        repeat_diter=0,
        enchar2phone_file='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liboyu.622/saved_data/resources/enchar_to_phone.txt',
        syllable2phone_file='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/resources/pinyin_to_phone.txt',
        one_mode=True,
    ):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        out_key: output key which contains syllable
        no_need_key: if no_need_key in item, skip Char2Syllable
        '''
        self.in_key = in_key
        self.out_key = out_key
        self.no_need_key = no_need_key
        self.one_mode = one_mode
        self.syllabel_map_dict = {
            'n1': 'en1',
            'n2': 'en2',
            'n3': 'en3',
            'n4': 'en4',
            'n5': 'en5',
            'lo5': 'lou5',
            'eng1': 'en1',
            'eng2': 'en2',
            'm2': 'me2',
        }
        self.phn_repeat = phn_repeat
        self.acoustic_ds = acoustic_ds
        self.phn2id = {}
        self.sy2phn = {}
        self.phn2id['<eps>'] = 0
        self.phn_cnt = 1
        self.mask_rate = mask_rate
        self.repeat_diter = repeat_diter
        assert repeat_diter < phn_repeat
        local_path = dist_hdfs_get(syllable2phone_file, '/tmp/', 'syllable2phone_file')
        # pylint:disable=consider-using-with
        for line in open(local_path, encoding='utf-8'):
            ls = line.strip().lower().split("\t")
            syllable = ls[0]
            phones = ls[1].split(" ")
            assert len(phones) == 2  # shengmu & yunmu
            sm = phones[0]
            if sm not in self.phn2id:
                self.phn2id[sm] = self.phn_cnt
                self.phn_cnt += 1
            ym = phones[1]
            for tone in '12345':
                # syllable with tone
                sy_t = syllable + tone
                ym_t = ym + tone
                if ym_t not in self.phn2id:
                    self.phn2id[ym_t] = self.phn_cnt
                    self.phn_cnt += 1
                self.sy2phn[sy_t] = [self.phn2id[p] for p in [sm, ym_t]]
        local_path = dist_hdfs_get(enchar2phone_file, '/tmp/', 'enchar2phone_file')
        for line in open(local_path, encoding='utf-8'):
            phoneme = line.strip()
            if phoneme not in self.phn2id:
                self.phn2id[phoneme] = self.phn_cnt
                self.phn_cnt += 1
        self.enphoneset = set(self.phn2id)
        self.eng2p = None

    def split_zh_en(self, zh_en_str, mark):
        '''split zh and en'''
        zh_en_group = []
        zh_gather = ""
        en_gather = ""
        zh_status = False
        for c in zh_en_str:
            if not zh_status and self.is_zh(c):
                zh_status = True
                if en_gather != "":
                    zh_en_group.append([mark["en"], en_gather])
                    en_gather = ""
            elif not self.is_zh(c) and zh_status:
                zh_status = False
                if zh_gather != "":
                    zh_en_group.append([mark["zh"], zh_gather])
            if zh_status:
                zh_gather += c
            else:
                en_gather += c
                zh_gather = ""

        if en_gather != "":
            zh_en_group.append([mark["en"], en_gather])
        elif zh_gather != "":
            zh_en_group.append([mark["zh"], zh_gather])

        return zh_en_group

    # pylint: disable='too-many-return-statements'
    def is_zh(self, c):
        '''check is Chinese'''
        x = ord(c)
        # Punct & Radicals
        if 0x2E80 <= x <= 0x33FF:
            return True
        # Fullwidth Latin Characters
        if 0xFF00 <= x <= 0xFFEF:
            return True
        # CJK Unified Ideographs &
        # CJK Unified Ideographs Extension A
        if 0x4E00 <= x <= 0x9FBB:
            return True
        # CJK Compatibility Ideographs
        if 0xF900 <= x <= 0xFAD9:
            return True
        # CJK Unified Ideographs Extension B
        if 0x20000 <= x <= 0x2A6D6:
            return True
        # CJK Compatibility Supplement
        if 0x2F800 <= x <= 0x2FA1D:
            return True
        return False

    def chn_syllable2phone(self, syllables):
        '''chn_syllable2phone'''
        phone_list = []
        for syllable in syllables:
            if syllable not in self.sy2phn:
                return phone_list
            if len(self.sy2phn[syllable]) == 2:
                # chinese syllable
                sm, ym = self.sy2phn[syllable]
                phone_list.append(sm)
                phone_list.append(ym)
        return phone_list

    def mask_and_repeat_phone(self, phn, phn_repeat):
        '''mask phn to zero randomly and repeat'''
        if random.uniform(0, 1) < self.mask_rate:
            phones = [0] * phn_repeat
        else:
            phones = [phn] * phn_repeat
        return phones

    def list2str(self, strlist):
        '''list2str'''
        s = ""
        preen = 0
        for t in strlist:
            if re.match(r'^[0-9a-zA-Z\']', t):
                if preen == 1:
                    s += " "
                    s += t
                else:
                    s += t
            else:
                s += t
            if re.match(r'[0-9a-zA-Z\']+$', t):
                preen = 1
            else:
                preen = 0
        return s

    # pylint: disable='too-many-branches'
    def __call__(self, item, **_kwargs):
        '''do convert label: eg.
        Args:
            item_data(dict): input data
        Returns:
            dict: raw data has phone
        '''
        if item is None or self.in_key not in item:
            return item
        if self.no_need_key in item and self.one_mode:
            return item

        if isinstance(item[self.in_key], str):
            line = item[self.in_key]
            text = line.replace("@@ ", "")
        else:
            text = self.list2str(item[self.in_key])
        mark = {"en": 1, "zh": 2}
        text_splited = self.split_zh_en(text, mark)

        if self.eng2p is None:
            self.eng2p = G2p()
        phone_ids = []
        for ch in text_splited:
            sub_text = ch[1].strip()
            if sub_text == "":
                continue
            if ch[0] == mark["en"]:
                en_phones = self.eng2p(sub_text)
                phone_ids += [
                    self.phn2id[sy]
                    for sy in en_phones
                    if sy.strip() != "" and sy.strip() in self.enphoneset
                ]
            else:
                syllable = lazy_pinyin(sub_text, style=Style.TONE3, neutral_tone_with_five=True)
                syllable = [sy for sy in syllable if sy.strip() != ""]
                syllable = [
                    self.syllabel_map_dict[sy] if sy in self.syllabel_map_dict else sy
                    for sy in syllable
                ]
                phone_ids += self.chn_syllable2phone(syllable)
        phn_repeat = self.phn_repeat
        if self.repeat_diter > 0:
            phn_repeat = random.randint(
                phn_repeat - self.repeat_diter, phn_repeat + self.repeat_diter
            )
        phones = [0] * phn_repeat
        for phoneme in phone_ids:
            phn_repeat = self.phn_repeat
            if self.repeat_diter > 0:
                phn_repeat = random.randint(
                    phn_repeat - self.repeat_diter, phn_repeat + self.repeat_diter
                )
            phones += self.mask_and_repeat_phone(phoneme, phn_repeat)
        phones += [0] * self.phn_repeat
        phones_np = np.asarray(phones).reshape(-1, 1)
        phone_length = len(phones) * self.acoustic_ds
        if self.one_mode:
            item['length'] = phone_length
        item[self.out_key] = phones_np
        return item


def old_base(item_list):
    '''baseline'''
    # do
    # python3 -m nltk.downloader "averaged_perceptron_tagger" "cmudict"
    # unzip /home/tiger/nltk_data/corpora/cmudict.zip -d /home/tiger/nltk_data/corpora/
    char_2_phone = BaseChar2Phone()
    phone_collate = FbankCollate(
        fbank_dim=1, key='phone', mask_key='phone_mask', out_key='phone_src', strict_mode=False
    )
    batch_list = []
    for item in item_list:
        new_item = char_2_phone(item)
        batch_list.append(new_item)
    batch = dict()
    phone_collate(batch_list, batch)
    return batch


def new_base(item_list):
    '''new base'''
    char_2_phone = Char2Phone()
    phone_collate = PhoneCollate()
    phone_predict = PhonePredict()
    batch_list = []
    for item in item_list:
        new_item = char_2_phone(item)
        batch_list.append(new_item)
    batch_out = dict()
    phone_collate(batch_list, batch_out)
    batch_out = to_cuda(batch_out)
    out_batch = phone_predict(batch_out)
    return out_batch


def _single_test_g2p(num=5):
    '''single test'''
    item_list = get_list(num)
    new_batch = new_base(item_list)
    base_batch = old_base(item_list)
    assert torch.allclose(base_batch['phone_src'], new_batch['phone_src'].cpu())
    assert torch.allclose(base_batch['phone_mask'], new_batch['phone_mask'].cpu())
