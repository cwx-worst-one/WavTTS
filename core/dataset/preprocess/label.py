# encoding=utf-8
'''
preprocess about label.
'''
# pylint:disable=too-many-branches,too-many-lines,too-many-return-statements
import io
import os
import re
import pickle
import random
import uuid
from collections import Counter, defaultdict
import numpy as np
import jieba
import torch
from pypinyin import lazy_pinyin, Style
from dataloader import FalconReader
import subword_nmt.apply_bpe
from core.utils import hdfs_get
from core.utils.dist_util import get_local_rank, get_local_size
from core.utils import dist_hdfs_get
from .preprocess import PREPROCESS
from .g2p_en import PreG2p, G2pPredict


def cut_words(label):
    '''cut words using jieba'''
    text = ''
    for word in label:
        if is_cjk_word(word):
            text += word
        else:
            text += ' '
            text += word
            text += ' '
    text = text.replace("  ", " ").strip().lower()
    words = set(jieba.cut(text, cut_all=False))
    return words


def split_labels(line):
    '''split labels.
    if label is CJK character,spilt it by character
    if label is English , spilt it by word.

    Args:
        line: type str, like '播放 歌曲 i am a robot'.
    Return:
        labels are list of string.
        like: ['播', '放', '歌', '曲', 'i', 'am', 'a', 'robot']
    '''
    line = line.strip()
    label_list = []
    start = 0
    while start < len(line):
        end = start + 1
        while end < len(line) and line[end] != ' ':
            end += 1
        for idx in range(start, end):
            is_ch = is_cjk_word(line[idx])
            if is_ch:
                if idx > start:
                    label_list.append(line[start:idx])
                label_list.append(line[idx])
                start = idx + 1
        if end > start:
            label_list.append(line[start:end])
        start = end + 1
    return label_list


def read_words(word_path, is_distractors=False):
    '''read words'''
    common_words = {}
    distractors_words = []
    tmp_file = '/tmp/' + str(uuid.uuid4())
    hdfs_get(word_path, tmp_file)
    with open(tmp_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() not in common_words:
                if is_distractors:
                    distractors_words.append(line.strip())
                else:
                    common_words[line.strip()] = True
    if is_distractors:
        return distractors_words
    return common_words


extra_cjk_blocks = [
    {'from': '\u2e80', 'to': '\u2eff'},  # CJK Radicals Supplement
    {'from': '\u3000', 'to': '\u303f'},  # CJK Symbols and Punctuation
    {'from': '\u31c0', 'to': '\u31ef'},  # CJK Strokes
    {'from': '\u3200', 'to': '\u32ff'},  # Enclosed CJK Letters and Months
    {'from': '\u3300', 'to': '\u33ff'},  # CJK Compatibility
    {'from': '\u3400', 'to': '\u4dbf'},  # CJK Unified Ideographs Extension A
    {'from': '\u4e00', 'to': '\u9fff'},  # CJK Unified Ideographs
    {'from': '\uf900', 'to': '\ufaff'},  # CJK Compatibility Ideographs
    {'from': '\ufe30', 'to': '\ufe4f'},  # CJK Compatibility Forms
    # CJK Unified Ideographs Extension B
    {'from': '\U00020000', 'to': '\U0002a6df'},
    # CJK Unified Ideographs Extension C
    {'from': '\U0002a700', 'to': '\U0002b73f'},
    # CJK Unified Ideographs Extension D
    {'from': '\U0002b740', 'to': '\U0002b81f'},
    # CJK Unified Ideographs Extension E
    {'from': '\U0002b820', 'to': '\U0002ceaf'},
    # CJK Unified Ideographs Extension F
    {'from': '\U0002ceb0', 'to': '\U0002ebef'},
    # CJK Compatibility Ideographs Supplement
    {'from': '\U0002f800', 'to': '\U0002fa1f'},
    # CJK Unified Ideographs Extension G
    {'from': '\U00030000', 'to': '\U0003134f'},
]


def is_cjk_word(string):
    '''
    return True if char is a chinese or japanese
    extra_cjk_blocks: https://www.compart.com/en/unicode/search?q=cjk#blocks
    '''
    # unicode for chinese
    if '\u4e00' <= string[0] <= '\u9fa5':
        return True
    # unicode for japanese
    if '\u3000' <= string[0] <= '\u30ff':
        return True
    # unicode for extra_cjk_blocks
    if any(block['from'] <= string[0] <= block['to'] for block in extra_cjk_blocks):
        return True
    return False


def add_space_op(match_obj):
    '''add space at the head and the tail of the regex matched seq.'''
    return " {} ".format(match_obj.group())


@PREPROCESS.register_module()
class SubtituteWordInMap:
    '''
    replace word in map_file
    For example, in Vietnamese, replace the string "ô" ̀” with "ồ"
    '''

    # pylint: disable='line-too-long'
    def __init__(
        self,
        in_key='label',
        out_key='label',
        enable_punct_filter=True,
        map_file='hdfs://harunava/home/byte_arnold_va_speech_asr/user/liboyu/dataset/tt_vi_VN/file_train_needed/vi_VN_str_map.dict',
    ):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        '''
        self.in_key = in_key
        self.out_key = out_key
        local_path = dist_hdfs_get(map_file, '/tmp/', 'map_file')
        self.subtitute_word_map = {}
        self.enable_punct_filter = enable_punct_filter
        with open(local_path, encoding='utf-8') as f:
            for line in f.readlines():
                key, val = line.strip().split(":")
                if '\\u' in key:
                    key = key.encode('utf-8').decode('unicode_escape')
                if '\\u' in val:
                    val = val.encode('utf-8').decode('unicode_escape')
                self.subtitute_word_map[key] = val
        self.space_filter = ''
        if self.enable_punct_filter:
            self.space_filter = Counter(
                '.,?，。？！’  ‘  ”  “  ·  ．；、“  〜~～…  ⋯：｜（）—  ❤」$-()[]!-+=*&%#@?<>,.;:{}`|/\\^《》「【】'
            )

    def __call__(self, item, **_kwargs):
        '''
        Args:
            item(dict): input data

        Returns:
            dict: data after label subtituted
            None, otherwise.
        '''
        if item is None or self.in_key not in item:
            return item

        if not isinstance(item[self.in_key], str):
            line = ' '.join(item[self.in_key]).strip().lower()
        else:
            line = item[self.in_key].strip().lower()
        for key, val in self.subtitute_word_map.items():
            line = line.replace(key, val)
        str_in = line
        if self.enable_punct_filter:
            str_in = ''
            for c in line:
                if self.space_filter[c] > 0:
                    c = ' '
                str_in += c
        item[self.out_key] = str_in
        return item


@PREPROCESS.register_module()
class LabelParser:
    '''parse in_key raw data to label'''

    def __init__(self, in_key='transcript', out_key='label'):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        '''
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, item, **_kwargs):
        '''do split label
        Args:
            item_data(dict): input data

        Returns:
            dict: raw in_key data trans to label
            None, otherwise.
        '''
        if item is None or self.in_key not in item:
            return item
        if not isinstance(item[self.in_key], str):
            if isinstance(item[self.in_key], list):
                item[self.out_key] = item[self.in_key]
            return item
        raw_data = item.pop(self.in_key)
        if '@@' in raw_data:
            raw_data = raw_data.replace("@@ ", "").replace("@@", "")
        label = [l for l in split_labels(raw_data) if l != '']
        item[self.out_key] = label
        return item


@PREPROCESS.register_module()
class ExtractRareWord(LabelParser):
    '''extarct rare word'''

    def __init__(
        self,
        in_key='label',
        out_key='rare_words',
        common_word_path=None,
        dropout=0.0,
    ):
        '''init'''
        super().__init__(in_key, out_key)
        self.common_words = {}
        if common_word_path is not None:
            self.common_words = read_words(common_word_path)
        self.dropout = dropout
        jieba.initialize()

    def __call__(self, item_data, **_kwargs):
        '''call'''
        if item_data is None:
            return None
        if self.in_key not in item_data:
            return item_data
        words = cut_words(item_data[self.in_key])
        rare_words = []
        for word in words:
            if word.strip() == '' or word in self.common_words:
                continue
            if self.dropout > 0.0 and random.random() < self.dropout:
                continue
            rare_words.append(word)
        item_data[self.out_key] = rare_words
        return item_data


@PREPROCESS.register_module()
class LangParser:
    '''parse lang from label'''

    def __init__(self, in_key='label', out_key='lang', tag_list=None):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        out_key: target key
        tag_list(list): supported tag list
        '''
        self.in_key = in_key
        self.out_key = out_key
        self.tag2id = {}
        # id 0 is for default
        if tag_list is not None:
            for i, tag in enumerate(tag_list):
                self.tag2id[tag] = i + 1

    def __call__(self, item, **_kwargs):
        '''do split label
        Args:
            item_data(dict): input data

        Returns:
            dict: raw in_key data trans to label
            None, otherwise.
        '''
        if item is None or self.in_key not in item:
            return item
        label = item[self.in_key]
        if len(label) < 1:
            return None
        lang_tag = label[0]
        lang_id = 0
        if lang_tag in self.tag2id:
            lang_id = self.tag2id[lang_tag]
            label.pop(0)
        item[self.out_key] = lang_id
        return item


@PREPROCESS.register_module()
class Char2Syllable:
    '''Chinese char to syllable'''

    def __init__(self, in_key='label', out_key='syllable', no_need_key='fbank'):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        out_key: output key which contains syllable
        no_need_key: if no_need_key in item, skip Char2Syllable
        '''
        self.in_key = in_key
        self.out_key = out_key
        self.no_need_key = no_need_key
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

    def __call__(self, item, **_kwargs):
        '''do convert label: eg.
            ['播', '放', '歌', '曲', 'i', 'am', 'a', 'robot']
            ->
            ['bo1', 'fang4', 'ge1', 'qu3', 'I', 'A', 'M', 'A', 'R', 'O', 'B', 'O', 'T']
            english words will show in uppercase letter
        Args:
            item_data(dict): input data
        Returns:
            dict: raw data has syllable
        '''
        if item is None or self.in_key not in item:
            return item
        if self.no_need_key in item:
            return item
        if isinstance(item[self.in_key], str):
            line = item[self.in_key]
            line = line.replace("@@ ", "")
            text = "".join(line.split(" "))
        else:
            text = "".join(item[self.in_key])
        syllable = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        syllable = [sy for sy in syllable if sy.strip() != ""]
        syllable = [
            self.syllabel_map_dict[sy] if sy in self.syllabel_map_dict else sy for sy in syllable
        ]
        new_syllable = []
        for sy in syllable:
            if sy[-1] not in '12345':
                # english word
                new_syllable += sy.upper()
            else:
                new_syllable.append(sy)
        item[self.out_key] = new_syllable
        return item


@PREPROCESS.register_module()
class Syllable2Phone:
    '''Chinese syllable to phone, where enlish word is converted to chars'''

    # pylint: disable=line-too-long
    def __init__(
        self,
        in_key='syllable',
        out_key='phone',
        phn_repeat=4,
        acoustic_ds=4,
        mask_rate=0.0,
        repeat_diter=0,
        syllable2phone_file='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/resources/pinyin_to_phone.txt',
    ):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        '''
        self.in_key = in_key
        self.out_key = out_key
        self.phn_repeat = phn_repeat
        self.acoustic_ds = acoustic_ds
        self.sy2phn = dict()
        self.phn2id = dict()
        self.phn2id['<eps>'] = 0
        self.phn_cnt = 1
        self.mask_rate = mask_rate
        self.repeat_diter = repeat_diter
        assert repeat_diter < phn_repeat
        # chinese syllable to shengmu & yunmu
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
        # english chars
        for phn in "QWERTYUIOPASDFGHJKLZXCVBNM'":
            self.phn2id[phn] = self.phn_cnt
            self.phn_cnt += 1
            self.sy2phn[phn] = [self.phn2id[phn]]

    def mask_and_repeat_phone(self, phn, phn_repeat):
        '''mask phn to zero randomly and repeat'''
        if random.uniform(0, 1) < self.mask_rate:
            phones = [0] * phn_repeat
        else:
            phones = [phn] * phn_repeat
        return phones

    def __call__(self, item, **_kwargs):
        '''do convert label
            ['bo1', 'fang4', 'ge1', 'qu3', 'I', 'A', 'M', 'A', 'R', 'O', 'B', 'O', 'T']
            ->
            ['b', 'o1', 'f', 'ang4', 'g', 'e1', 'q', 'v3',
             'I', 'A', 'M', 'A', 'R', 'O', 'B', 'O', 'T']
        Args:
            item_data(dict): input data
        '''
        if item is None or self.in_key not in item:
            return item
        phn_repeat = self.phn_repeat
        if self.repeat_diter > 0:
            phn_repeat = random.randint(
                phn_repeat - self.repeat_diter, phn_repeat + self.repeat_diter
            )
        phones = [0] * phn_repeat
        for syllable in item[self.in_key]:
            if syllable not in self.sy2phn:
                return None
            phn_repeat = self.phn_repeat
            if self.repeat_diter > 0:
                phn_repeat = random.randint(
                    phn_repeat - self.repeat_diter, phn_repeat + self.repeat_diter
                )
            if len(self.sy2phn[syllable]) == 2:
                # chinese syllable
                sm, ym = self.sy2phn[syllable]
                phones += self.mask_and_repeat_phone(sm, phn_repeat)
                phones += self.mask_and_repeat_phone(ym, phn_repeat)
            else:
                # english char
                en_char = self.sy2phn[syllable][0]
                phones += self.mask_and_repeat_phone(en_char, phn_repeat)
        phones += [0] * self.phn_repeat
        phones_np = np.asarray(phones).reshape(-1, 1)
        item['length'] = len(phones) * self.acoustic_ds
        item[self.out_key] = phones_np
        return item


@PREPROCESS.register_module()
class Text2Char:
    '''text to char'''

    def __init__(
        self,
        in_key='label',
        out_key='phone',
        no_need_key='fbank',
        phn_repeat=4,
        acoustic_ds=4,
        mask_rate=0.0,
        one_mode=True,
        max_length=10000,
        chars=None,
    ):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        out_key: output key which contains syllable
        no_need_key: if no_need_key in item, skip Text2Char
        '''
        self.in_key = in_key
        self.out_key = out_key
        self.no_need_key = no_need_key
        self.phn_repeat = phn_repeat
        self.acoustic_ds = acoustic_ds
        self.mask_rate = mask_rate
        self.one_mode = one_mode
        char_cnt = 1
        self.char2id = dict()
        self.char2id[' '] = 0
        if chars is None:
            chars = "yangikudtmjlbshproecvfwzx2.135q46708'9-"
        for c in chars:
            self.char2id[c] = char_cnt
            char_cnt += 1
        self.max_length = max_length

    def mask_and_repeat_phone(self, phn):
        '''mask phn to zero randomly and repeat'''
        if random.uniform(0, 1) < self.mask_rate:
            phones = [0] * self.phn_repeat
        else:
            phones = [phn] * self.phn_repeat
        return phones

    def __call__(self, item, **_kwargs):
        '''do convert label: text to char, for English, Indonesia, etc
        Args:
            item_data(dict): input data
        Returns:
            dict: raw data has syllable
        '''
        if item is None or self.in_key not in item:
            return item if self.one_mode else None
        if self.no_need_key in item and self.one_mode:
            return item
        if isinstance(item[self.in_key], str):
            line = item[self.in_key]
            line = line.replace("@@ ", "")
            text = " ".join(line.split(" "))
        else:
            text = " ".join(item[self.in_key])
        phones = [0] * self.phn_repeat
        for c in text.lower():
            if c not in self.char2id:
                phones += [0] * self.phn_repeat
            else:
                phones += self.mask_and_repeat_phone(self.char2id[c])
        phones += [0] * self.phn_repeat
        if len(phones) * self.acoustic_ds > self.max_length:
            return None
        phones_np = np.asarray(phones).reshape(-1, 1)
        if self.one_mode:
            item['length'] = len(phones) * self.acoustic_ds
        item[self.out_key] = phones_np
        return item


@PREPROCESS.register_module()
class Char2Phone:
    '''trans Chinese and English word to phone'''

    # pylint: disable='line-too-long'
    def __init__(
        self,
        in_key='label',
        out_key='phone',
        undone_key='undone_phone',
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
        undone_key: incomplete key which contains encode_phones, insert_pos, repeat_num
        no_need_key: if no_need_key in item, skip Char2Syllable
        '''
        self.in_key = in_key
        self.out_key = out_key
        self.undone_key = undone_key
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
        self.eng2p = PreG2p()

    def split_zh_en(self, zh_en_str, mark):
        '''split Chinese and English'''
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
        '''trans list to str'''
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
        phone_ids = []
        for ch in text_splited:
            sub_text = ch[1].strip()
            if sub_text == "":
                continue
            if ch[0] == mark["en"]:
                en_phones = self.eng2p(sub_text)
                for sy in en_phones:
                    if isinstance(sy, list):
                        phone_ids.append(sy)
                    elif sy.strip() in self.enphoneset:
                        phone_ids.append(self.phn2id[sy])

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
        undone_phones = []  # encode(list), pos(int), repeat(int)
        undone_phone_length = 0
        for phoneme in phone_ids:
            phn_repeat = self.phn_repeat
            if self.repeat_diter > 0:
                phn_repeat = random.randint(
                    phn_repeat - self.repeat_diter, phn_repeat + self.repeat_diter
                )
            if isinstance(phoneme, list):  # g2p encode word
                # encode_list, steps, pos, phone_repeat
                if random.uniform(0, 1) < self.mask_rate:
                    phones += [0] * phn_repeat
                else:
                    undone_phones.append([phoneme[:-1], phoneme[-1], len(phones), phn_repeat])
                    undone_phone_length = phn_repeat * len(phoneme[:-1])
            else:
                phones += self.mask_and_repeat_phone(phoneme, phn_repeat)
        phones += [0] * self.phn_repeat
        phones_np = np.array(phones)
        phone_length = len(phones) * self.acoustic_ds
        undone_phone_length *= self.acoustic_ds
        if self.one_mode:
            item['length'] = phone_length + undone_phone_length
        item[self.out_key] = phones_np
        item[self.undone_key] = undone_phones
        return item


@PREPROCESS.register_module()
class TimestampParser:
    '''parse timestamp'''

    def __init__(self, in_key='timestamp', out_key='timestamp'):
        '''init.
        Args:
        in_key: raw data key that will be parsed to label
        '''
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, item, **_kwargs):
        '''do split label
        Args:
            item_data(dict): input data

        Returns:
            dict: raw in_key data trans to timestamp
            None, otherwise.
        '''
        if item is None or self.in_key not in item:
            item[self.out_key] = []
            return item
        timestamp = [(word['start_time'], word['end_time']) for word in item[self.in_key]]
        item[self.out_key] = timestamp
        return item


@PREPROCESS.register_module()
class PunctuationFilter:
    '''
    remove punctuations from the speech transcripts.
    suitable for both chinese and english text.
    '''

    def __init__(
        self,
        key='label',
        invalid_tokens_pattern=None,
        keep_empty_label=False,
    ):
        '''init.
        Args:
            invalid_tokens_pattern: regex pattern for invalid tokens removal
        '''
        self.key = key
        if invalid_tokens_pattern is None:
            self.invalid_tokens_pattern = re.compile(r"[^ 0-9a-zA-Z\u4e00-\u9fa5']")
        else:
            self.invalid_tokens_pattern = re.compile(r"{}".format(invalid_tokens_pattern))
        self.special_tokens_pattern = re.compile(r'</?[a-zA-Z]+/?>')
        self.keep_empty_label = keep_empty_label

    def __call__(self, item_data, **_kwargs):
        '''remove punctuations from the label sequence.
        Args:
            item_data(dict): input data.

        Returns:
            dict: item_data with clean label.
            None, otherwise.
        '''
        if item_data is None:
            return None

        label = item_data.get(self.key, None)
        if label is None:
            return None

        clean_label = []
        if isinstance(label, str):
            label = self.special_tokens_pattern.sub(add_space_op, label).strip().split()
        for token in label:
            if not self.special_tokens_pattern.match(token):
                token = self.invalid_tokens_pattern.sub(' ', token).strip()
            if token != '':
                clean_label.append(token)

        if not clean_label and not self.keep_empty_label:
            return None
        item_data[self.key] = clean_label
        return item_data


@PREPROCESS.register_module()
class LabelLength:
    '''
    LabelLength calculator
    '''

    def __init__(self, key='char', out_key='char_length'):
        '''get in_out_ratio.
        Args:
            in_out_ratio(int): in out ratio condition.
        '''
        self.key = key
        self.out_key = out_key

    def __call__(self, item_data, **_kwargs):
        '''do the filter.

        Args:
            item_data(dict): must contains fbank and label.

        Returns
            dict: item_data if the item_data meet conditions,
                  None, otherwise.
        '''
        if item_data is None or self.key not in item_data:
            return None

        label_len = len(item_data[self.key])
        item_data[self.out_key] = label_len

        return item_data


@PREPROCESS.register_module()
class SplitChar:
    '''
    do char split.
    '''

    def __init__(
        self,
        key='label',
        skip_number=True,
    ):
        '''init.
        Args:
            skip_number(bool): skip numbers.
        '''
        self.in_key = key
        self.skip_number = skip_number

    def __call__(self, item_data, **_kwargs):
        '''do char split.
        Args:
            item_data(dict): input data
        Returns:
            dict: item data with bpe chars if OK.
            None, otherwise.
        '''
        if item_data is None or self.in_key not in item_data:
            return None

        if not isinstance(item_data[self.in_key], str):
            line = ' '.join(item_data[self.in_key]).strip().lower()
        else:
            line = item_data[self.in_key].strip().lower()

        if line == '':
            return None

        line_str = self.split_sent_to_char(line.lower())

        if line_str is None:
            return None

        item_data[self.in_key] = line_str

        return item_data

    def split_sent_to_char(self, lab):
        '''
        split sent to char type
        code -> c@@ o@@ d@@ e
        '''
        del_list = [
            '「',
            '」',
            '。',
            '・',
            '?',
            '、',
            '！',
            '『',
            '』',
            '……',
            '——',
            '.',
            '《',
            '》',
            '!',
            '-',
            ' ',
            '−',
            "’",
            '(',
            ')',
            '[',
            ']',
        ]

        en_flag = False
        char_list = []

        for char in lab:
            if u'\u0030' <= char <= u'\u0039' and self.skip_number:
                return None
            if (
                (u'\u0041' <= char <= u'\u005a')
                or (u'\u0061' <= char <= u'\u007a')
                or (char == "'")
            ):
                en_flag = True
                char_list.append(char + '@@')
            else:
                if en_flag:
                    char_list[-1] = char_list[-1].replace('@@', '')
                if char not in del_list:
                    char_list.append(char)
                en_flag = False

        if en_flag:
            char_list[-1] = char_list[-1].replace('@@', '')

        if len(char_list) < 1:
            return None

        return ' '.join(char_list)


@PREPROCESS.register_module()
class BPE:
    '''
    char bpe(byte-pair encoding).
    do char bpe.
    '''

    # text_key: because lm training may not use 'label' to indicate 'text input'
    def __init__(
        self,
        bpe_fn=None,
        tgt_dict=None,
        reorder_tgt_dict=None,
        use_eos=False,
        key='label',
        out_key=None,
        skip_list=None,
        skip_bpe=False,
        keep_bpe_words=False,
        bpe_code=None,
        bpe_dropout=0.0,
        keep_empty_label=False,
        convert_bpe_to_text=False,
    ):
        '''init.
        Args:
            bpe_fn(callable): bpe functions.
            tgt_dict(ScpDictionary): tgt dict.
            reorder_tgt_dict(ScpDictionary): reorder tgt dict.
            use_eos(bool): whether use eos.
            skip_list(list): special tokens (<space>, </fil>, etc.) list to skip bpe processing.
            keep_bpe_words(bool): keep bpe words in data.
        '''
        if bpe_fn:
            self.bpe_fn = bpe_fn
        elif bpe_code:
            self.bpe_fn = subword_nmt.apply_bpe.BPE(io.StringIO(bpe_code))
        else:
            raise Exception('bpe_code is None!')
        self.tgt_dict = tgt_dict
        self.reorder_tgt_dict = reorder_tgt_dict
        self.use_eos = use_eos
        self.in_key = key
        if not out_key:
            self.out_key = key.replace('label', 'char')
        else:
            self.out_key = out_key
        self.skip_list = skip_list
        self.skip_bpe = skip_bpe
        self.keep_bpe_words = keep_bpe_words
        self.keep_empty_label = keep_empty_label
        self.convert_bpe_to_text = convert_bpe_to_text
        self.bpe_dropout = bpe_dropout

    def __call__(self, item_data, **_kwargs):
        '''do char bpe.
        Args:
            item_data(dict): input data

        Returns:
            dict: item data with bpe chars if OK.
            None, otherwise.
        '''
        if item_data is None or self.in_key not in item_data:
            return None

        if not isinstance(item_data[self.in_key], str):
            line = ' '.join(item_data[self.in_key]).strip().lower()
        else:
            line = item_data[self.in_key].strip().lower()

        if self.convert_bpe_to_text:
            line = line.strip().replace('@@ ', '')

        if line == '' and not self.keep_empty_label:
            return None

        if not self.skip_bpe:
            if self.skip_list is not None:
                line = line.split()
                line = [
                    self.bpe_fn.process_line(c, self.bpe_dropout) if c not in self.skip_list else c
                    for c in line
                ]
                line_str = ' '.join(line)
            else:
                line_str = self.bpe_fn.process_line(line, self.bpe_dropout)
            if self.keep_bpe_words:
                item_data[self.in_key + '_bpe'] = line_str.split(' ')
        else:
            line_str = line

        tgt_dict = self.reorder_tgt_dict if self.reorder_tgt_dict is not None else self.tgt_dict
        char_bpes = tgt_dict.encode_line(line_str, append_eos=self.use_eos, add_if_not_exist=False)
        if len(char_bpes) == 0 and not self.keep_empty_label:
            return None
        item_data[self.out_key] = char_bpes
        return item_data


@PREPROCESS.register_module()
class ConvertToChar:
    '''
    convert english word to characters
    '''

    # text_key: because lm training may not use 'label' to indicate 'text input'
    def __init__(
        self,
        tgt_dict=None,
        reorder_tgt_dict=None,
        use_eos=False,
        key='label',
        out_key=None,
        skip_list=None,
        convert_bpe_to_text=None,
    ):
        '''init.
        Args:
            tgt_dict(ScpDictionary): tgt dict.
            reorder_tgt_dict(ScpDictionary): reorder tgt dict.
            use_eos(bool): whether use eos.
            skip_list(list): special tokens (<space>, </fil>, etc.) list to skip bpe processing.
        '''
        self.tgt_dict = tgt_dict
        self.reorder_tgt_dict = reorder_tgt_dict
        self.use_eos = use_eos
        self.in_key = key
        if not out_key:
            self.out_key = key.replace('label', 'char')
        else:
            self.out_key = out_key
        self.skip_list = skip_list
        self.convert_bpe_to_text = convert_bpe_to_text

    def __call__(self, item_data, **_kwargs):
        '''do converting char.
        Args:
            item_data(dict): input data

        Returns:
            dict: item data with chars if OK.
            None, otherwise.
        '''
        # pylint:disable=too-many-branches
        if item_data is None or self.in_key not in item_data:
            return None

        if not isinstance(item_data[self.in_key], str):
            line = ' '.join(item_data[self.in_key]).strip().lower()
        else:
            line = item_data[self.in_key].strip().lower()

        if self.convert_bpe_to_text:
            line = line.strip().replace('@@ ', '')

        if line == '':
            return None

        line_str = ''
        nwords = 0
        for c in line:
            if c == ' ':
                line_str += '<space>'
            else:
                line_str += c
            line_str += ' '
            nwords += 1
        line_char_list = line_str.strip().split(' ')

        nwords = nwords + 1 if self.use_eos else nwords
        ids = np.zeros((nwords,), dtype='int32')

        for i, c in enumerate(line_char_list):
            if c in self.tgt_dict:
                ids[i] = self.tgt_dict.index(c)
            else:
                ids[i] = self.tgt_dict.unk_index
        if self.use_eos:
            ids[-1] = self.tgt_dict.eos_index

        if len(ids) == 0:
            return None
        item_data[self.out_key] = ids

        return item_data


@PREPROCESS.register_module()
class MaskLabelByLang:
    '''
    keep specific language tokens and mask other tokens
        example:
            ['播', '放', '歌', '曲', 'i', 'am', 'a', 'robot']
            -> ['播', '放', '歌', '曲', '<unk>', '<unk>', '<unk>', '<unk>']
    '''

    def __init__(self, keep_lang, mask='<unk>', key='label', out_key=None):
        '''
        do char bpe.
        Args:
            keep_lang: language id, 'zh' 'en'
            mask: mask token
            langs: ['zh', 'en']
        Returns:
            dict: item_data with masked label.
            None, otherwise.
        '''
        self.langs = ['zh', 'en']
        self.keep_lang = keep_lang
        self.key = key
        self.out_key = out_key
        self.mask = mask
        assert keep_lang in self.langs, 'language must in {}'.format(self.langs)

    def detect_lang(self, char):
        '''
        identify token language
        '''
        if is_cjk_word(char):
            return 'zh'
        return 'en'

    def __call__(self, item_data, **_kwargs):
        '''
        identify token language and mask
        '''
        token_list = item_data[self.key]
        assert isinstance(token_list, list), 'label_file_list must be a list'
        mask_list = []
        for token in token_list:
            if self.detect_lang(token) == self.keep_lang:
                mask_list.append(token)
            else:
                mask_list.append(self.mask)
        item_data[self.out_key] = mask_list
        return item_data


@PREPROCESS.register_module()
class SpaceAddForZhLabel:
    '''add space between zh and zh_en cases egs: 我喜欢buy汽车->我 喜 欢 buy 汽 车'''

    def __init__(self, key='text'):
        '''for init.'''
        self.key = key

    def __call__(self, item_data, **_kwargs):
        '''
        for actually-call
         add space between zh and zh_en cases
        '''
        if item_data is None:
            return None

        arr = item_data[self.key]
        if isinstance(arr, str):
            arr = item_data[self.key].strip().split(' ')

        # for train/dev/testset if "-" 英文半词标注存在，则整句不参与训练/测试
        if "-" not in " ".join(arr):
            line_final = " ".join(arr)
            line_final = line_final.replace("^ ", "")
            line_final_new = self.process_text(line_final)
            line_final_new = line_final_new.replace(" ' ", "'")
            if line_final_new == '':
                return None
            item_data[self.key] = line_final_new

        return item_data

    def process_text(self, line):
        '''
        Distinguish and split Chinese-char and English-words
        '''
        # pylint:disable=too-many-branches
        space_list = [
            '.',
            ',',
            '?',
            '，',
            '。',
            '？',
            '！',
            '．',
            '；',
            '、',
            '“',
            '〜',
            '~',
            '～',
            '--',
            ';',
            ':',
        ]
        empty_list = [
            "’",
            '”',
            '·',
            '…',
            '⋯',
            '：',
            '\u200d',
            '｜',
            '（',
            '）',
            '—',
            '❤',
            '」',
            str(chr(65039)),
            str(chr(1465)),
            '_',
            '(',
            ')',
            '[',
            ']',
            '!',
            '-',
            '+',
            '=',
            '*',
            '&',
            '%',
            '#',
            '@',
            '?',
            '<',
            '>',
            '{',
            '}',
            '`',
            '|',
            '/',
            '\\',
            '^',
            '《',
            '》',
        ]
        for i in space_list:
            line = line.replace(i, ' ')

        for i in empty_list:
            line = line.replace(i, '')

        new_line = ''
        last_ch_flag = False
        last_en_flag = False
        for char in line.strip():
            if self.is_cjk_word(char):
                if last_en_flag:
                    new_line = new_line + ' ' + char
                else:
                    new_line = new_line + char
                last_ch_flag = True
                last_en_flag = False
            elif ord(char) < 123:
                if last_ch_flag:
                    new_line = new_line + ' ' + char
                else:
                    new_line = new_line + char
                last_ch_flag = False
                last_en_flag = True

        line = new_line

        words = line.lower().strip().split(' ')
        out_word_list = []
        char_list = []

        for word in words:
            if len(word) < 1:
                continue

            if len(word) < 1:
                continue
            if ord(word[0]) < 65 or ord(word[0]) > 122:
                if len(char_list) >= 1:
                    char2word = ''.join(char_list)
                    out_word_list.extend(self.number_str_to_char(char2word))
                    char_list = []
                out_word_list.extend(list(word))
            else:
                if len(word) > 1:
                    if len(char_list) >= 1:
                        char2word = ''.join(char_list)
                        out_word_list.extend(self.number_str_to_char(char2word))
                        char_list = []
                    out_word_list.append(word)
                else:
                    char_list.append(word)
        if len(char_list) >= 1:
            char2word = ''.join(char_list)
            out_word_list.extend(self.number_str_to_char(char2word))
        new_out_word_list = []
        for word in out_word_list:
            if len(word) < 1:
                continue
            en_str = ''
            for char in word:
                if self.is_cjk_word(char):
                    if len(en_str) > 0:
                        new_out_word_list.append(en_str)
                        en_str = ''
                    new_out_word_list.append(char)
                else:
                    if ord(char) < 65 or ord(char) > 122:  # ord-48->0, ord-122->z
                        if len(en_str) >= 1:
                            new_out_word_list.append(en_str)
                        new_out_word_list.append(char)
                        en_str = ''
                    else:
                        en_str += char
            if len(en_str) >= 1:
                new_out_word_list.append(en_str)

        final_line = ' '.join(new_out_word_list)
        return final_line

    @staticmethod
    def is_cjk_word(str_word):
        '''judge Chinese char'''
        return '\u4e00' <= str_word[0] <= '\u9fa5'

    @staticmethod
    def number_str_to_char(word):
        '''number str to char'''
        if len(word) > 0:
            if word[0] >= '0' and word[0] <= '9':
                char_list = list(word)
            else:
                char_list = [word]
        else:
            char_list = []
        return char_list


@PREPROCESS.register_module()
class SpaceAdd:
    '''add <space> token between english words'''

    def __call__(self, item_data, **_kwargs):
        '''add <space> token
        Args:
            item_data(dict): input data.

        Returns:
            dict: item_data with <space> inserted into the label sequence.
            None, otherwise.
        '''
        if item_data is None:
            return None

        line = ' <space> '.join(item_data['label']).strip()

        if line == '':
            return None

        item_data['label'] = line.split()

        return item_data


@PREPROCESS.register_module()
class DictTrans:
    '''transform a value to index.'''

    def __init__(self, in_key, out_key, dicts, relabel=False, relabel_key='relabel'):
        '''init.'''
        self.in_key = in_key
        self.out_key = out_key
        self.dicts = dicts
        self.relabel = relabel
        self.relabel_key = relabel_key
        self.cls_num = len(self.dicts)

    def __call__(self, item, **_kwargs):
        '''do call.'''
        if item is None or self.in_key not in item:
            return None
        src = item[self.in_key]
        if src not in self.dicts:
            return None
        new_cls = self.dicts[src]
        if self.relabel and self.relabel_key in item:
            new_cls += item[self.relabel_key] * self.cls_num
        item[self.out_key] = new_cls
        return item


@PREPROCESS.register_module()
class CodeSwitchTagAdd:
    '''add ^ token between chinese to english'''

    def __init__(self, key='label', add_token='^', special_pattern_list=None):
        '''init.'''
        self.key = key
        self.add_token = add_token
        self.special_pattern_list = special_pattern_list if special_pattern_list else []

    def __call__(self, item_data, **_kwargs):
        '''add ^ token between chinese to english
        Args:
            item_data(dict): input data.

        Returns:
            dict: item_data with ^ into the label sequence.
            None, otherwise.
        '''
        if item_data is None or self.key not in item_data:
            return None

        label_with_code_switch_tag = []
        label_before_switch = item_data[self.key]
        if len(label_before_switch) == 0:
            return None
        meet_special = False
        is_special = False
        for i in range(len(label_before_switch) - 1):
            now_token = label_before_switch[i]
            next_token = label_before_switch[i + 1]
            if is_cjk_word(now_token):
                label_with_code_switch_tag.append(now_token)
                for pattern in self.special_pattern_list:
                    if re.match(pattern, next_token):
                        meet_special = True
                        break
                if not is_cjk_word(next_token) and not meet_special:
                    label_with_code_switch_tag.append(self.add_token)
            else:
                for pattern in self.special_pattern_list:
                    if re.match(pattern, now_token):
                        is_special = True
                        break
                if not is_special:
                    if meet_special:
                        label_with_code_switch_tag.append(self.add_token)
                        meet_special = False
                is_special = False
                label_with_code_switch_tag.append(now_token)
        label_with_code_switch_tag.append(label_before_switch[-1])
        item_data[self.key] = label_with_code_switch_tag
        return item_data


@PREPROCESS.register_module()
class MergeSplitedLabel:
    '''merge item label from another path.'''

    def __init__(
        self,
        label_path_root,
        label_file_list,
        item_key='uttid',
        global_shuffle=True,
        chunk_size=1000,
        prefetch_chunk_num=50,
    ):
        '''init.'''

        self.global_shuffle = global_shuffle
        self.root = label_path_root
        if isinstance(label_file_list, str):
            label_file_list = eval(label_file_list)
        assert isinstance(label_file_list, (tuple, list)), 'label_file_list must be a list'
        self.label_file_list = [
            os.path.join(self.root, label_file) for label_file in label_file_list
        ]
        self.item_key = item_key
        self.current_path_idx = None
        self.contents = {}
        self.chunk_size = chunk_size
        self.prefetch_chunk_num = prefetch_chunk_num
        self.reader = None

    def __call__(self, item, **kwargs):
        '''do merge.'''
        path_idx = kwargs.get('path_idx', None)
        if path_idx is None or item is None:
            return None
        if self.reader is None:
            self.reader = FalconReader(
                self.label_file_list,
                128,  # fd_cache_size
                5,  # io_thread_num
                10,  # io_retry
                'MergeSplitedLabel',  # unused
                get_local_size(),  # GPU num in one worker
                get_local_rank(),  # GPU idx in one worker
                self.chunk_size,  # chunk size
            )

        if self.global_shuffle:
            if not self.contents:
                self.read_contents()
        else:
            if path_idx != self.current_path_idx:
                self.read_content(path_idx)

        if self.item_key not in item or item[self.item_key] not in self.contents:
            return None

        labels = self.contents[item[self.item_key]]
        for k, v in labels.items():
            item[k] = v

        return item

    def read_contents(self):
        '''
        read labels contents from all paths.
        labels contents should be small(less than 300M), they will be load totally into CPU memory.
        '''
        keys = self.reader.list_keys(list(range(len(self.label_file_list))), False)
        entry_nums = len(keys)
        chunk_idxs = [i * self.chunk_size for i in range(entry_nums // self.chunk_size)]
        for i in range(0, len(chunk_idxs), self.prefetch_chunk_num):
            vals_buf = self.reader.read_many(chunk_idxs[i : i + self.prefetch_chunk_num])
            for chunk_idx, vals in enumerate(vals_buf):
                for idx, v in enumerate(vals):
                    k = keys[chunk_idxs[i + chunk_idx] + idx]
                    self.contents[k] = pickle.loads(v)

    def read_content(self, path_idx):
        '''
        read labels contents from a path.
        labels contents should be small(less than 300M), they will be load totally into CPU memory.
        '''
        keys = self.reader.list_keys([path_idx], False)
        entry_nums = len(keys)
        chunk_idxs = [i * self.chunk_size for i in range(entry_nums // self.chunk_size)]
        self.contents = {}
        for i in range(0, len(chunk_idxs), self.prefetch_chunk_num):
            vals_buf = self.reader.read_many(chunk_idxs[i : i + self.prefetch_chunk_num])
            for chunk_idx, vals in enumerate(vals_buf):
                for idx, v in enumerate(vals):
                    k = keys[chunk_idxs[i + chunk_idx] + idx]
                    self.contents[k] = pickle.loads(v)
        self.current_path_idx = path_idx


@PREPROCESS.register_module()
class LabelMap:
    '''map labels to indexes'''

    def __init__(self, tgt_dict=None, key='s2s_label', out_key=None):
        '''init.
        Args:
            tgt_dict(dict): tgt dict.
        '''
        self.tgt_dict = tgt_dict
        self.in_key = key
        if not out_key:
            self.out_key = 'char'
        else:
            self.out_key = out_key

    def __call__(self, item_data, **_kwargs):
        '''do label map based on tgt dict.
        Args:
            item_data(dict): input data.

        Returns:
            dict: item data with label indexes.
            None, otherwise.
        '''
        if item_data is None or self.in_key not in item_data:
            return None
        labels = item_data[self.in_key]
        indexes = np.array(
            [self.tgt_dict.get(str(label), self.tgt_dict['<unk>']) for label in labels]
        )
        item_data[self.out_key] = indexes
        return item_data


@PREPROCESS.register_module()
class AlignZhG2PLabel:
    '''
    Align Zh G2P labels, use blank to represent all En words and other non-prononuce symbols.
    '''

    def __init__(
        self,
        phone_dict,
        char_key='label_bpe',
        phone_key='sy_label',
        out_key='sy_label',
        align_unmatched=False,
        blank_idx=0,
    ):
        '''init.'''
        self.char_key = char_key
        self.phone_key = phone_key
        self.phone_dict = phone_dict
        self.blank_idx = blank_idx
        self.out_key = out_key
        self.align_unmatched = align_unmatched

    def __call__(self, item_data, **_kwargs):
        '''align zh g2p label'''
        if item_data is None or self.char_key not in item_data or self.phone_key not in item_data:
            return None
        char_label = item_data[self.char_key]
        phone_label = item_data[self.phone_key]
        char_len = len(char_label)
        phone_len = len(phone_label)

        # might has en words or zh er tone
        if char_len != phone_len:
            if not self.align_unmatched:
                return None
            new_phone_label = []
            new_phone_text = []
            phone_idx = 0
            for char in char_label:
                if is_cjk_word(char):
                    if phone_idx >= phone_len:
                        return None
                    phone = self.phone_dict[phone_label[phone_idx]]
                    # skip en phones, which are all in uppercase format
                    while phone != phone.lower():
                        phone_idx += 1
                        if phone_idx >= phone_len:
                            return None
                        phone = self.phone_dict[phone_label[phone_idx]]
                    new_phone_label.append(phone_label[phone_idx])
                    new_phone_text.append(self.phone_dict[phone_label[phone_idx]])
                    phone_idx += 1
                else:
                    # for en words, use blank to represent their phones
                    new_phone_label.append(self.blank_idx)
                    new_phone_text.append('blank')
            if len(new_phone_label) != len(char_label):
                return None
            item_data[self.phone_key] = np.array(new_phone_label, dtype=np.int32)
        return item_data


@PREPROCESS.register_module()
class EmotionLabelParser:
    '''change labels from string to number'''

    def __init__(self, key='label', out_key=None):
        '''init.'''

        self.in_key = key
        if not out_key:
            self.out_key = key
        else:
            self.out_key = out_key

    def __call__(self, item_data, **_kwargs):
        '''do label transform.
        Args:
            string.

        Returns:
            numpy.
        '''

        if item_data is None or self.in_key not in item_data:
            return item_data
        label = item_data[self.in_key]
        indexes = np.array([int(tmp) for tmp in label[1:-1].split(',')])
        indexes = indexes.reshape(1, -1)
        item_data[self.out_key] = indexes
        return item_data


@PREPROCESS.register_module()
class MaxLengthCut:
    '''cut feature or label to max_length'''

    def __init__(
        self, key='fbank', out_key=None, min_len=4, max_len=80, keep_last=False, need_length=False
    ):
        '''init'''
        self.in_key = key
        self.max_len = max_len
        self.min_len = min_len
        self.keep_last = keep_last
        self.need_length = need_length
        if not out_key:
            self.out_key = key
        else:
            self.out_key = out_key

    def __call__(self, item, **_kwargs):
        '''call for feature cut to max_length'''
        if item is None or self.in_key not in item:
            return item
        feature = item[self.in_key]
        if self.need_length:
            item['length'] = min(feature.shape[0], self.max_len)
        if feature.shape[-1] <= self.min_len:
            return None
        if feature.shape[-1] <= self.max_len:
            return item
        if self.keep_last:
            feature[0][self.max_len - 1] = feature[0][-1]

        item[self.out_key] = feature[:, : self.max_len]
        return item


@PREPROCESS.register_module()
class W2vPhone2charLabel:
    '''wav2vec base padding postprocess phone2char cn'''

    def __init__(
        self,
        out_key='char',
        tgt_dict=None,
        lexicon=None,
        preprocess=True,
        start_token_index=0,
    ):
        '''init.
        Args:
            out_key(str): char key in output item.
            tgt_dict(dict): tgt dict.
            lexicon(defaultdict): lexicon dictionary
            preprocess(bool): whether to preprocess text
            start_token_index(int): char start token index.
        '''
        self.out_key = out_key
        self.tgt_dict = tgt_dict
        self.lexicon = lexicon
        self.preprocess = preprocess
        self.start_token_index = start_token_index

    def __call__(self, item_data, **_kwargs):
        '''wav2vec phone 2 char parser'''
        if not isinstance(self.lexicon, defaultdict):
            tmp = self.lexicon
            self.lexicon = defaultdict(lambda: ['<UNK>'])
            self.lexicon.update(tmp)

        char = np.array([1], dtype=np.int32)
        text = item_data['text']
        char_list = []
        for char in self.preprocess_txt(text):
            phn_list = self.lexicon[char]
            if phn_list[0] == '<UNK>':
                continue
            char_list += [self.tgt_dict.index(phn) for phn in phn_list]
        char = np.array(char_list, dtype=np.int32)
        # remove start #tokens
        if self.start_token_index > 0:
            char = char[self.start_token_index :]
        item_data[self.out_key] = char
        return item_data

    def preprocess_txt(self, text):
        '''preprocess'''
        if not self.preprocess:
            return text.split()
        text = self.neaten_text(text)
        return self.word2char(text)

    @staticmethod
    def neaten_text(txt):
        '''neaten_text'''
        num_map = {
            '0': '零',
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九',
        }
        space_filter = Counter('.,?，。？！’‘”“·．；、“〜~～…⋯：｜（）—❤」$-()[]!-+=*&%#@?<>,.;:{}`|/\\^《》「【】')
        neat = ''
        for c in txt.lower():
            if space_filter[c] > 0:
                c = ' '
            elif '0' <= c <= '9':
                c = num_map[c]
            neat += c
        return neat

    @staticmethod
    def word2char(text):
        '''word2char'''
        prev = ''
        char = []
        for c in text:
            if 'A' <= c <= 'Z' or 'a' <= c <= 'z' or c == "'":
                prev += c
            else:
                if len(prev) > 0:
                    char += [prev]
                    prev = ''
                if c != ' ':
                    char += [c]
        return char


class List2Str:
    '''text list to str
    ['我', '很', '好'] => '我很好'
    '''

    def __init__(
        self,
        out_key='text',
        in_key='label',
    ):
        '''init.'''
        self.out_key = out_key
        self.in_key = in_key

    def __call__(self, item_data, **_kwargs):
        '''text list to str'''
        item_data[self.out_key] = ''.join(item_data[self.in_key])
        return item_data


@PREPROCESS.register_module()
class DomainAdd:
    '''add domain info'''

    def __init__(self, key='domain', domain_list=None):
        '''init.'''
        self.key = key
        self.domain_list = domain_list

    def __call__(self, item_data, **kwargs):
        '''add domain info
        Args:
            item_data(dict): input data.

        Returns:
            dict: item_data with domain info.
            None, otherwise.
        '''
        path_idx = kwargs.get('path_idx', None)
        if item_data is None or path_idx is None or self.domain_list is None:
            return None
        item_data[self.key] = self.domain_list[path_idx]
        return item_data


@PREPROCESS.register_module()
class Textchar2Index:
    '''bert char to index or index1 to index2,
    it keeps consistent with the text emotional task of NLP,
    and is different from the huggingface's BertTokenizer
    '''

    def __init__(
        self,
        key='text',
        out_key='text',
        vocab_dict=None,
        punc_table=None,
        char2index=False,
        is_cn=True,
        unsqueeze=True,
        default_unk_index=4,
    ):
        '''init.'''
        self.in_key = key
        self.out_key = out_key
        self.vocab_dict = vocab_dict
        self.punc_table = punc_table

        self.char2index = bool(char2index)
        self.is_cn = bool(is_cn)
        self.unsqueeze = unsqueeze
        # the default unk index for the nlp group is 4 while for hugginface is 100
        self.default_unk_index = default_unk_index

    def __call__(self, item_data, **_kwargs):
        '''bert char2index or index1 to index2'''
        if self.vocab_dict is None:
            return item_data

        unk_index = self.vocab_dict['[UNK]'] if self.char2index else self.default_unk_index
        char_list = [self.vocab_dict['[CLS]']] if self.char2index else []
        data = item_data[self.in_key] if self.char2index else item_data[self.in_key][0]
        if self.char2index and self.punc_table is not None:
            data = data.translate(self.punc_table)

        for char in data:
            char = str(char)
            if char not in self.vocab_dict:
                char_list.append(unk_index)
            else:
                char_list.append(self.vocab_dict[char])

        if self.char2index:
            char_list.append(self.vocab_dict['[SEP]'])

        char_list = np.array(char_list, dtype=np.int32)
        if self.unsqueeze:
            char_list = char_list.reshape(1, -1)
        item_data[self.out_key] = char_list
        return item_data


@PREPROCESS.register_module()
class BertTokenizer:
    '''
    using BertTokenizer from huggingface to convert char to index,
    split word into word pieces.
    '''

    def __init__(
        self,
        key='text',
        out_key=None,
        bert_tokenizer=None,
        max_chars_num=512,
    ):
        '''init.'''
        self.in_key = key
        self.out_key = out_key if out_key is not None else key
        self.bert_tokenizer = bert_tokenizer
        assert max_chars_num > 0
        self.max_chars_num = max_chars_num

    def __call__(self, item_data, **_kwargs):
        '''bert char2index or index1 to index2'''
        if self.bert_tokenizer is None:
            return item_data

        if isinstance(item_data[self.in_key], list):
            line = ' '.join(item_data[self.in_key])
        elif isinstance(item_data[self.in_key], str):
            line = item_data[self.in_key]
        else:
            return item_data
        line = line.lower().strip()

        encoded_input = self.bert_tokenizer(line)
        char_list = np.array(encoded_input['input_ids'], dtype=np.int32)
        if len(char_list) > self.max_chars_num:
            char_list = char_list[-self.max_chars_num :]
        item_data[self.out_key] = char_list
        return item_data


@PREPROCESS.register_module()
class DocBertTokenizer:
    '''
    using BertTokenizer from huggingface to convert doc_char to index,
    split word into word pieces.
    '''

    def __init__(
        self,
        key='text',
        out_key=None,
        bert_tokenizer=None,
        max_chars_num=128,
        max_sents_num=128,
        # bos_id/eos_id from huggingface bert-base vocab
        bos_id=101,
        eos_id=102,
    ):
        """
        init
        """
        self.in_key = key
        self.out_key = out_key if out_key is not None else key
        self.bert_tokenizer = bert_tokenizer
        self.max_chars_num = max_chars_num
        self.max_sents_num = max_sents_num
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, item_data, **_kwargs):
        '''bert char2index or index1 to index2'''
        if self.bert_tokenizer is None:
            return item_data

        if isinstance(item_data[self.in_key], list):
            line = ' '.join(item_data[self.in_key])
        elif isinstance(item_data[self.in_key], str):
            line = item_data[self.in_key]
        else:
            return item_data
        line = " [SEP] ".join(line.lower().strip().split("</s>"))

        encoded_input = self.bert_tokenizer(line)
        doc_list = []
        left_bound = 0
        for index, token in enumerate(encoded_input['input_ids']):
            if token == self.eos_id:
                cur_sentence = encoded_input['input_ids'][left_bound : index + 1]
                if cur_sentence and cur_sentence[0] != self.bos_id:
                    cur_sentence = [self.bos_id] + cur_sentence
                    cur_sentence = cur_sentence[: self.max_chars_num]
                doc_list.append(cur_sentence)
                left_bound = index + 1
        doc_list = doc_list[: self.max_sents_num]

        item_data[self.out_key] = doc_list
        return item_data


@PREPROCESS.register_module()
class AlignSpeakerLabel:
    """
    Speaker Selection
    Intercept or align the audio label of the target speakers num
    """

    def __init__(self, speaker_key='speakers', label_key='label', target_speakers=4):
        """
        init
        """
        self.speaker_key = speaker_key
        self.label_key = label_key
        self.target_speakers = target_speakers

    def __call__(self, item, **_kwargs):
        """
        call
        We should note that, there are voiced frames for the aligned speakers after this step.
        """
        if item is None or self.speaker_key not in item:
            return item

        speakers = item[self.speaker_key]
        speaker_label = item[self.label_key]
        speakers_num = len(speakers)
        assert speakers_num == speaker_label.shape[0]
        if speakers_num < self.target_speakers:
            pad_shape = self.target_speakers - len(speakers)
            pad_label = np.zeros((pad_shape, speaker_label.shape[1]))
            speaker_label = np.concatenate([speaker_label, pad_label])
            pad_speakers = [""] * pad_shape
            speakers.extend(pad_speakers)
        elif speakers_num > self.target_speakers:
            # Make sure there are valid speakers in the sampled list
            valid_speakers_list = np.flatnonzero(speaker_label.sum(1) > 0)
            n_valid_speakers = len(valid_speakers_list.tolist())
            nonvalid_speakers_list = np.flatnonzero(speaker_label.sum(1) == 0)
            n_nonvalid_speakers = len(nonvalid_speakers_list)
            assert n_valid_speakers + n_nonvalid_speakers == speaker_label.shape[0]

            # valid speakers select num
            n_select_valid = random.randint(
                max(1, self.target_speakers - n_nonvalid_speakers),
                min(n_valid_speakers, self.target_speakers),
            )
            n_select_nonvalid = self.target_speakers - n_select_valid
            target_speakers_label = random.sample(valid_speakers_list.tolist(), n_select_valid)
            target_speakers_label += random.sample(
                nonvalid_speakers_list.tolist(), n_select_nonvalid
            )
            assert len(target_speakers_label) == self.target_speakers
            random.shuffle(target_speakers_label)
            speaker_label = np.take(speaker_label, target_speakers_label, axis=0)
            speakers = [speakers[idx] for idx in target_speakers_label]
        item[self.speaker_key] = speakers
        item[self.label_key] = speaker_label
        return item


@PREPROCESS.register_module()
class ConcateEnLetters:
    '''Concate English letters, for example: c c t v -> cctv'''

    # TODO(dlh) mainly for aligning with tf, the performance
    #  will continue to be verified and improved.

    def __init__(self, key='label', out_key=None):
        '''init'''
        self.in_key = key
        if not out_key:
            self.out_key = key
        else:
            self.out_key = out_key

    def __call__(self, item_data, **_kwargs):
        '''Concate English letters'''
        if item_data is None or self.in_key not in item_data:
            return None
        labels = item_data[self.in_key]
        out_labels = self.concate_en_letters(labels)
        item_data[self.out_key] = out_labels
        return item_data

    @staticmethod
    def concate_en_letters(labels):
        '''concate_en_letters'''
        in_labels = labels
        if isinstance(labels, str):
            in_labels = labels.split(' ')
        out_labels = []
        en_letters = []
        for label in in_labels:
            if len(label) > 0 and (ord(label[0]) < 65 or ord(label[0]) > 122):
                if en_letters:
                    out_labels.append(''.join(en_letters))
                    en_letters = []
                out_labels.append(label)
            else:
                if len(label) == 1:
                    en_letters.append(label)
                else:
                    if en_letters:
                        out_labels.append(''.join(en_letters))
                        en_letters = []
                    out_labels.append(label)
        if en_letters:
            out_labels.append(''.join(en_letters))
        if isinstance(labels, str):
            out_labels = ' '.join(out_labels)
        return out_labels


@PREPROCESS.register_module()
class DialogHistToContext:
    '''process the multi-turn dialogue histories to dialog-context'''

    def __init__(
        self,
        key='dialogue_history',
        out_key='context_text',
        turns=1,
        max_context_len=512,
        mask_prob=0,
        mask_token='<pad>',
        perturb_prob=0.1,  # refers to WER
        edit_ops_probs='0.6,0.2,0.2',  # prob of Sub, Ins, Del
        vocab=None,  # vocab of words, not indices
    ):
        '''init'''
        self.key = key
        self.out_key = out_key
        assert turns > 0 and max_context_len > 0
        self.turns = turns
        self.max_context_len = max_context_len
        self.mask_prob = mask_prob
        self.perturb_prob = perturb_prob
        self.mask_token = mask_token
        self.vocab = vocab
        self.vocab_size = 0
        if self.perturb_prob > 0:
            assert vocab is not None
            self.vocab_size = len(vocab)
        self.edit_sub_prob = 0
        self.edit_ins_prob = 0
        self.edit_del_prob = 0
        if isinstance(edit_ops_probs, str):
            sub_p, ins_p, del_p = [float(t) for t in edit_ops_probs.split(',')]
            assert sum((sub_p, ins_p, del_p)) <= 1.0
            self.edit_sub_prob = sub_p
            self.edit_ins_prob = sub_p + ins_p
            self.edit_del_prob = sub_p + ins_p + del_p

    def __call__(self, item_data, **_kwargs):
        '''get context from dialog history'''
        if item_data is None or self.key not in item_data:
            return item_data
        context = item_data[self.key]
        if not isinstance(context, list):
            return item_data
        # for now, just concat multi-turn dialog history to one long-sentence
        context = split_labels(' '.join(context[-self.turns :]))
        context = context[-self.max_context_len :]

        # context perturbation
        perturb_idx = np.where(np.random.uniform(size=(len(context))) <= self.perturb_prob)[0]
        for i in perturb_idx:
            op_p = random.random()
            i = min(i, len(context) - 1)
            token_idx = random.randint(0, self.vocab_size - 1)
            if op_p <= self.edit_sub_prob:  # Sub
                context[i] = self.vocab[token_idx]
            elif self.edit_sub_prob < op_p <= self.edit_ins_prob:  # Ins
                context.insert(i, self.vocab[token_idx])
            elif self.edit_ins_prob < op_p <= self.edit_del_prob:  # Del
                context.pop(i)

        if not context:
            context = [self.mask_token]
        if self.mask_prob > 0:
            mask_idx = np.where(np.random.uniform(size=(len(context))) <= self.mask_prob)[0]
            for i in mask_idx:
                context[i] = self.mask_token
        item_data[self.out_key] = context
        return item_data


@PREPROCESS.register_module()
class PhonePredict:
    '''encode phone predict'''

    # pylint: disable='line-too-long'
    def __init__(
        self,
        out_key='phone_src',
        mask_key='phone_mask',
        phone_key='phone_src',
        phone_mask='phone_length',
        encode_key='encode_src',
        encode_mask='encode_mask',
        phone2idx_file='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/resources/dolphin/preprocess/label/phone2idx.txt',
    ):
        '''init'''
        self.out_key = out_key
        self.phone_key = phone_key
        self.phone_mask = phone_mask
        self.encode_key = encode_key
        self.encode_mask = encode_mask
        self.predict_g2p = G2pPredict()
        self.phn2id = {}
        self.phn2id['<eps>'] = 0
        self.mask_key = mask_key
        local_path = dist_hdfs_get(phone2idx_file, '/tmp/', 'phone2idx.txt')
        # pylint: disable='consider-using-with', 'unspecified-encoding'
        for line in open(local_path):
            pho2idx = line.strip().split(" ")
            phoneme = pho2idx[0]
            idx = int(pho2idx[1])
            self.phn2id[phoneme] = idx
        self.enphoneset = set(self.phn2id)

    def predict(self, words, steps, lengths):
        '''encode'''
        en_phones = self.predict_g2p(words, steps, lengths)
        phone_ids = []
        for en_phone in en_phones:
            phone_id = [
                self.phn2id[sy]
                for sy in en_phone
                if sy.strip() != "" and sy.strip() in self.enphoneset
            ]
            phone_ids.append(phone_id)
        return phone_ids

    def __call__(self, item, **_kwargs):
        '''call'''
        if item is None or self.phone_key not in item:
            return item
        phone_tensor = item.pop(self.phone_key)
        phone_length = item.pop(self.phone_mask)
        if self.encode_key not in item:
            bsz = phone_tensor.shape[0]
            max_phone_length = phone_tensor.shape[1]
            out_mask_tensor = torch.zeros(bsz, max_phone_length, device=phone_tensor.device)
            for bid, length in enumerate(phone_length):
                out_mask_tensor[bid, :length] = 1
            item[self.mask_key] = out_mask_tensor
            item[self.out_key] = phone_tensor
            return item
        encode_tensor = item.pop(self.encode_key)
        encode_mask = item.pop(self.encode_mask)

        offset = 0
        bsz = phone_tensor.shape[0]
        encode_list = []
        split_tensor = [[0, phone_length[bid]] for bid in range(bsz)]
        encode_lengths = [encode_info[0] for encode_info in encode_mask]
        encode_steps = [encode_info[1] for encode_info in encode_mask]
        phone_ids = self.predict(encode_tensor, encode_steps, encode_lengths)
        for encode_info, phone_id in zip(encode_mask, phone_ids):
            bid = encode_info[2]
            pos = encode_info[3]
            split_tensor[bid].insert(-1, pos)
            repeat = encode_info[4]
            phone_id = [val for val in phone_id for i in range(repeat)]
            phone_length[bid] += len(phone_id)
            encode_list.append(phone_id)

        max_phone_length = max(phone_length)

        out_phone_tensor = torch.zeros(bsz, max_phone_length, device=phone_tensor.device)
        out_mask_tensor = torch.zeros(bsz, max_phone_length, device=phone_tensor.device)
        for bid in range(bsz):
            out_mask_tensor[bid, : phone_length[bid]] = 1
        t_id = 0
        for bid, split_info in enumerate(split_tensor):
            if len(split_info) > 2:
                raw_offset = 0
                offset = 0
                for idx in split_info[1:]:
                    out_phone_tensor[bid, offset : offset + (idx - raw_offset)] = phone_tensor[
                        bid, raw_offset:idx
                    ]
                    offset += idx - raw_offset
                    if offset >= phone_length[bid]:
                        break
                    raw_offset = idx
                    lent = len(encode_list[t_id])
                    out_phone_tensor[bid, offset : offset + lent] = torch.tensor(
                        encode_list[t_id], device=out_phone_tensor.device
                    )
                    offset += lent
                    t_id += 1
            else:
                out_phone_tensor[bid, : phone_tensor[bid].shape[0]] = phone_tensor[bid, :]

        item[self.out_key] = out_phone_tensor
        item[self.mask_key] = out_mask_tensor
        return item
