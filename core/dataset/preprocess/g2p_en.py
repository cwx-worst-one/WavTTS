# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p
'''
# pylint: disable='import-error'
from builtins import str as unicode
import codecs
import re
import os
import unicodedata
import numpy as np
import torch

try:
    import inflect
    import nltk
except ImportError:
    inflect, nltk = None, None
try:
    from nltk import pos_tag
    from nltk.corpus import cmudict
    from nltk.tokenize import TweetTokenizer
except ImportError:
    pos_tag, cmudict, TweetTokenizer = None, None, None
from core.utils import dist_hdfs_get, get_local_rank


# pylint: disable='invalid-name', 'line-too-long'
class PreG2p:
    '''Pre G2p'''

    def __init__(self):
        '''init'''
        self.download_hdfs_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/resources/dolphin/preprocess/label/'
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
        except LookupError:
            dist_hdfs_get(
                os.path.join(self.download_hdfs_root, 'averaged_perceptron_tagger.pickle'),
                '/home/tiger/nltk_data/taggers/averaged_perceptron_tagger',
            )
        try:
            nltk.data.find('corpora/cmudict/cmudict')
        except LookupError:
            dist_hdfs_get(
                os.path.join(self.download_hdfs_root, 'cmudict'),
                '/home/tiger/nltk_data/corpora/cmudict',
            )
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self._inflect = inflect.engine()
        self._comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
        self._decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
        self._pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
        self._dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
        self._ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
        self._number_re = re.compile(r'[0-9]+')
        self.dirname = os.path.dirname(__file__)
        self.cmu = cmudict.dict()
        self.homograph2features = self.construct_homograph_dictionary()
        self.word_tokenize = TweetTokenizer().tokenize

    @staticmethod
    def _remove_commas(m):
        '''_remove_commas'''
        return m.group(1).replace(',', '')

    @staticmethod
    def _expand_decimal_point(m):
        '''_expand_decimal_point'''
        return m.group(1).replace('.', ' point ')

    @staticmethod
    def _expand_dollars(m):
        '''_expand_dollars'''
        match = m.group(1)
        parts = match.split('.')
        if len(parts) > 2:
            return match + ' dollars'  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            cent_unit = 'cent' if cents == 1 else 'cents'
            return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
        if dollars:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            return '%s %s' % (dollars, dollar_unit)
        if cents:
            cent_unit = 'cent' if cents == 1 else 'cents'
            return '%s %s' % (cents, cent_unit)
        return 'zero dollars'

    def _expand_ordinal(self, m):
        '''_expand_ordinal'''
        return self._inflect.number_to_words(m.group(0))

    def normalize_numbers(self, text):
        '''normalize_numbers'''
        text = re.sub(self._comma_number_re, PreG2p._remove_commas, text)
        text = re.sub(self._pounds_re, r'\1 pounds', text)
        text = re.sub(self._dollars_re, PreG2p._expand_dollars, text)
        text = re.sub(self._decimal_number_re, PreG2p._expand_decimal_point, text)
        text = re.sub(self._ordinal_re, PreG2p._expand_ordinal, text)
        text = re.sub(self._number_re, self._expand_number, text)
        return text

    def _expand_number(self, m):
        '''_expand_number'''
        num = int(m.group(0))
        if 1000 < num < 3000:
            if num == 2000:
                return 'two thousand'
            if 2000 < num < 2010:
                return 'two thousand ' + self._inflect.number_to_words(num % 100)
            if num % 100 == 0:
                return self._inflect.number_to_words(num // 100) + ' hundred'
            return self._inflect.number_to_words(num, andword='', zero='oh', group=2).replace(
                ', ', ' '
            )
        return self._inflect.number_to_words(num, andword='')

    def construct_homograph_dictionary(self):
        '''construct_homograph_dictionary'''
        dist_hdfs_get(
            os.path.join(self.download_hdfs_root, 'homographs.en'),
            self.dirname,
        )
        f = os.path.join(self.dirname, 'homographs.en')
        homograph2features = dict()
        # pylint: disable='consider-using-with'
        for line in codecs.open(f, 'r', 'utf8').read().splitlines():
            if line.startswith("#"):
                continue  # comment
            headword, pron1, pron2, pos1 = line.strip().split("|")
            homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
        return homograph2features

    def encode(self, word):
        '''encode'''
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x.append(len(word))
        return x

    def __call__(self, text):
        '''call'''
        # preprocessing
        text = unicode(text)
        text = self.normalize_numbers(text)
        text = ''.join(
            char
            for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )  # Strip accents
        text = text.lower()
        # pylint: disable='anomalous-backslash-in-string'
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = self.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else:  # predict for oov
                pron = [self.encode(word)]  # encode-list + lenword
            prons.extend(pron)
            prons.extend([" "])
        return prons[:-1]


class G2pPredict:
    '''G2p predict'''

    def __init__(self):
        '''init'''
        phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
            'AA0',
            'AA1',
            'AA2',
            'AE0',
            'AE1',
            'AE2',
            'AH0',
            'AH1',
            'AH2',
            'AO0',
            'AO1',
            'AO2',
            'AW0',
            'AW1',
            'AW2',
            'AY0',
            'AY1',
            'AY2',
            'B',
            'CH',
            'D',
            'DH',
            'EH0',
            'EH1',
            'EH2',
            'ER0',
            'ER1',
            'ER2',
            'EY0',
            'EY1',
            'EY2',
            'F',
            'G',
            'HH',
            'IH0',
            'IH1',
            'IH2',
            'IY0',
            'IY1',
            'IY2',
            'JH',
            'K',
            'L',
            'M',
            'N',
            'NG',
            'OW0',
            'OW1',
            'OW2',
            'OY0',
            'OY1',
            'OY2',
            'P',
            'R',
            'S',
            'SH',
            'T',
            'TH',
            'UH0',
            'UH1',
            'UH2',
            'UW',
            'UW0',
            'UW1',
            'UW2',
            'V',
            'W',
            'Y',
            'Z',
            'ZH',
        ]
        self.idx2p = dict(enumerate(phonemes))
        self.load_variables()

    def load_variables(self):
        '''load variables'''
        download_hdfs_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/resources/dolphin/preprocess/label/'
        dirname = os.path.dirname(__file__)
        dist_hdfs_get(
            os.path.join(download_hdfs_root, 'checkpoint20.npz'),
            dirname,
        )
        variables = np.load(os.path.join(dirname, 'checkpoint20.npz'))  # can't use torch.load
        self.enc_emb = torch.from_numpy(variables["enc_emb"]).cuda(
            get_local_rank()
        )  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = torch.from_numpy(variables["enc_w_ih"]).cuda(
            get_local_rank()
        )  # (3*128, 64)
        self.enc_w_hh = torch.from_numpy(variables["enc_w_hh"]).cuda(
            get_local_rank()
        )  # (3*128, 128)
        self.enc_b_ih = torch.from_numpy(variables["enc_b_ih"]).cuda(get_local_rank())  # (3*128,)
        self.enc_b_hh = torch.from_numpy(variables["enc_b_hh"]).cuda(get_local_rank())  # (3*128,)

        self.dec_emb = torch.from_numpy(variables["dec_emb"]).cuda(
            get_local_rank()
        )  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = torch.from_numpy(variables["dec_w_ih"]).cuda(
            get_local_rank()
        )  # (3*128, 64)
        self.dec_w_hh = torch.from_numpy(variables["dec_w_hh"]).cuda(
            get_local_rank()
        )  # (3*128, 128)
        self.dec_b_ih = torch.from_numpy(variables["dec_b_ih"]).cuda(get_local_rank())  # (3*128,)
        self.dec_b_hh = torch.from_numpy(variables["dec_b_hh"]).cuda(get_local_rank())  # (3*128,)
        self.fc_w = torch.from_numpy(variables["fc_w"]).cuda(get_local_rank())  # (74, 128)
        self.fc_b = torch.from_numpy(variables["fc_b"]).cuda(get_local_rank())  # (74,)

    def sigmoid(self, x):
        '''sigmod'''
        return 1 / (1 + torch.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        '''grucell'''
        rzn_ih = torch.matmul(x, w_ih.T) + b_ih
        rzn_hh = torch.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, : rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3 :]
        rz_hh, n_hh = rzn_hh[:, : rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3 :]

        rz = self.sigmoid(rz_ih + rz_hh)
        split_size = rz.shape[-1] // 2
        r, z = torch.split(rz, split_size, -1)

        n = torch.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        '''gru
        w_ih : 768, 256
        w_hh : 768, 256
        b_ih : 768
        b_hh : 768
        h0: 1, 256
        '''
        if h0 is None:
            h0 = torch.zeros((x.shape[0], w_hh.shape[1]), dtype=torch.float32, device=x.device)
        h = h0  # initial hidden state
        outputs = torch.zeros(
            (x.shape[0], steps, w_hh.shape[1]), dtype=torch.float32, device=x.device
        )
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, x, lengths):
        '''encode'''
        x = torch.index_select(self.enc_emb, 0, x)
        bsz = len(lengths)
        max_lengths = max(lengths)
        new_encode_tensor = torch.zeros(bsz, max_lengths, x.shape[-1], device=x.device)
        offset = 0
        for bid, length in enumerate(lengths):
            new_encode_tensor[bid][:length, :] = x[offset : offset + length, :]
            offset += length
        return new_encode_tensor

    def predict(self, enc, steps, lengths):
        '''predict'''
        assert len(steps) == len(lengths)
        enc = self.encode(enc, lengths)
        bsz = enc.shape[0]
        max_steps = max(steps)
        enc = self.gru(
            enc,
            max_steps + 1,
            self.enc_w_ih,
            self.enc_w_hh,
            self.enc_b_ih,
            self.enc_b_hh,
            h0=torch.zeros((bsz, self.enc_w_hh.shape[-1]), dtype=torch.float32, device=enc.device),
        )
        for bid, step in enumerate(steps):
            enc[bid, -1, :] = enc[bid, step, :]  # or step-1?

        last_hidden = enc[:, -1, :]
        # decoder
        dec = torch.index_select(self.dec_emb, 0, torch.tensor([2] * bsz, device=enc.device))

        h = last_hidden
        preds = torch.zeros(bsz, 20, device=enc.device)
        for idx in range(20):
            h = self.grucell(
                dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh
            )  # (b, h)
            logits = torch.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax(dim=1)
            preds[:, idx] = pred
            dec = torch.index_select(self.dec_emb, 0, pred)
        preds = preds.cpu().numpy().tolist()
        out_preds = []
        for bid in range(bsz):
            this_preds = []
            for val in preds[bid]:
                idx = int(val)
                if idx == 3:
                    break
                this_preds.append(self.idx2p.get(idx, "<unk>"))
            out_preds.append(this_preds)
        return out_preds

    def __call__(self, encode_words, encode_steps, lengths):
        '''call func'''
        prons = self.predict(encode_words, encode_steps, lengths)
        return prons
