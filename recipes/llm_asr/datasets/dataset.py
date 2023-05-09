import os, json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
from itertools import groupby

from transformers import PreTrainedTokenizerFast

from samantha.utils.hparams import DotDict
from recipes.llm_asr.utils.hdfs_tools import hdfs_open


class CharacterTokenizerWithAudioTokens():
    def __init__(self, audio_tokens_num=100, audio_token_dedup=True):
        self.audio_tokens_num = audio_tokens_num
        self.en_chars = {char: idx+audio_tokens_num+1 for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ',.?!")}
        self.vocab_size = len(self.en_chars) + audio_tokens_num + 3 + 50 # <sep> <s> </s> <pad> and 50 placeholders
        self.bos = self.vocab_size - 1
        self.eos = self.vocab_size - 1
        self.sep = self.vocab_size - 2
        self.pad = 0
        self.audio_token_dedup = audio_token_dedup

    def tokenize(self, sentence, input_key):
        if input_key == 'inputs':
            if self.audio_token_dedup:
                tokens = [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
                return [key for key, _group in groupby(tokens)]
            else:
                return [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
        elif input_key == 'targets':
            return [self.en_chars[char] for char in sentence]
        else:
            return None

    def token2text(self, tokens):
        inv_map = {v: k for k, v in self.en_chars.items()}
        return ''.join([inv_map[idx] for idx in tokens if idx in inv_map])
    
    def token2audio(self, tokens):
        return ' '.join([str(idx-1) for idx in tokens])

class BPETokenizerWithAudioTokens():
    def __init__(self, bpe_tokenizer_file, audio_tokens_num=100, audio_token_dedup=True):
        self.audio_tokens_num = audio_tokens_num
        self.bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file=bpe_tokenizer_file)
        self.vocab_size = self.bpe_tokenizer.vocab_size + audio_tokens_num + 3 + 50 # <sep> <s> </s> <pad> and 50 placeholders
        self.bos = self.vocab_size - 1
        self.eos = self.vocab_size - 1
        self.sep = self.vocab_size - 2
        self.pad = 0
        self.audio_token_dedup = audio_token_dedup

    def tokenize(self, sentence, input_key):
        if input_key == 'inputs':
            if self.audio_token_dedup:
                tokens = [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
                return [key for key, _group in groupby(tokens)]
            else:
                return [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
        elif input_key == 'targets':
            return [item + self.audio_tokens_num + 1 for item in self.bpe_tokenizer.encode(sentence)] # pad and audio tokens
        else:
            return None

    def token2text(self, tokens):
        return self.bpe_tokenizer.decode([x - self.audio_tokens_num - 1 for x in tokens]).replace(' '+self.bpe_tokenizer._tokenizer.model.continuing_subword_prefix, '')
    
    def token2audio(self, tokens):
        return ' '.join([str(idx-1) for idx in tokens])

class GPT2ASRDataset(Dataset):

    def __init__(self, path, hp=None, task='joint', inference=False):
        self.path = path
        self.dir_path = os.path.abspath(os.path.dirname(path))
        self.hp = DotDict(hp)
        self.task = task
        self.inference = inference
        self.metas = self.get_metadata(path)
        
        if self.hp.text_tokenizer == 'char':
            self.tokenizer = CharacterTokenizerWithAudioTokens(audio_tokens_num=self.hp.audio_tokens_num, audio_token_dedup=self.hp.audio_token_dedup)
        elif self.hp.text_tokenizer == 'bpe':
            self.tokenizer = BPETokenizerWithAudioTokens(self.hp.bpe_tokenizer_file, audio_tokens_num=self.hp.audio_tokens_num, audio_token_dedup=self.hp.audio_token_dedup)
        else:
            raise NotImplementedError

    def get_metadata(self, path):
        metas = []
        with hdfs_open(path, 'r') as f:
            for l in f:
                sample = json.loads(l)
                if self.inference:
                    if sample['inputs'].count('audio') > self.hp.max_token_lens - 2:
                        continue
                    else:
                        metas.append(sample)
                else:
                    # pre-filter: assume audio tokens acounst for 90% of all tokens at most
                    if sample['inputs'].count('audio') > int(self.hp.max_token_lens * 0.9):
                        continue
                    else:
                        metas.append(sample)
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        # wav_id [T, n], n代表codebook数量
        # text_id [T,]
        # text_converter: 将text_idx转化为匹配的id

        x = self.metas[idx]
        
        wav_id, text_id, uttid = np.array(self.tokenizer.tokenize(x['inputs'], 'inputs')), np.array(self.tokenizer.tokenize(x['targets'], 'targets')), x['uttid'] if 'uttid' in x else None

        if self.inference:
            if self.task == 'asr':
                seq = [self.tokenizer.bos] + list(wav_id) + [self.tokenizer.sep]
                if len(seq) <= self.hp.max_token_lens:
                    seq = np.asarray(seq)

                    # get len
                    wav_len = wav_id.shape[0]

                    # get position embedding
                    pos_id = np.asarray(list(range(wav_len + 2)))
                    seq_sen_id = np.asarray([1] * (wav_len + 2))
                    return seq, pos_id, seq_sen_id, uttid
            elif self.task == 'tts':
                seq = [self.tokenizer.bos] + list(text_id) + [self.tokenizer.sep]
                if len(seq) <= self.hp.max_token_lens:
                    seq = np.asarray(seq)

                    # get len
                    text_len = text_id.shape[0]

                    # get position embedding
                    pos_id = np.asarray(list(range(text_len + 2)))
                    seq_sen_id = np.asarray([2] * (text_len + 2))
                    return seq, pos_id, seq_sen_id, uttid
            else:
                raise NotImplementedError
        else:
            if self.task == 'asr':
                seq = [self.tokenizer.bos] + list(wav_id) + [self.tokenizer.sep] + list(text_id) + [self.tokenizer.eos]
                if len(seq) <= self.hp.max_token_lens:
                    seq = np.asarray(seq)

                    # get len
                    wav_len = wav_id.shape[0]
                    text_len = text_id.shape[0]

                    # get position embedding
                    pos_id = np.asarray(list(range(wav_len + 2)) + list(range(text_len + 1)))
                    seq_sen_id = np.asarray([1] * (wav_len + 2) + [2] * (text_len + 1))
                    return seq, pos_id, seq_sen_id, None
            elif self.task == 'tts':
                seq = [self.tokenizer.bos] + list(text_id) + [self.tokenizer.sep] + list(wav_id) + [self.tokenizer.eos]
                if len(seq) <= self.hp.max_token_lens:
                    seq = np.asarray(seq)

                    # get len
                    wav_len = wav_id.shape[0]
                    text_len = text_id.shape[0]

                    # get position embedding
                    pos_id = np.asarray(list(range(text_len + 2)) + list(range(wav_len + 1)))
                    seq_sen_id = np.asarray([2] * (text_len + 2) + [1] * (wav_len + 1))
                    return seq, pos_id, seq_sen_id, None
            elif self.task == 'joint':
                prob = random.random()
                if prob > 0.5:
                    seq = [self.tokenizer.bos] + list(wav_id) + [self.tokenizer.sep] + list(text_id) + [self.tokenizer.eos]
                    if len(seq) <= self.hp.max_token_lens:
                        seq = np.asarray(seq)

                        # get len
                        wav_len = wav_id.shape[0]
                        text_len = text_id.shape[0]

                        # get position embedding
                        pos_id = np.asarray(list(range(wav_len + 2)) + list(range(text_len + 1)))
                        seq_sen_id = np.asarray([1] * (wav_len + 2) + [2] * (text_len + 1))
                        return seq, pos_id, seq_sen_id, None
                else:
                    seq = [self.tokenizer.bos] + list(text_id) + [self.tokenizer.sep] + list(wav_id) + [self.tokenizer.eos]
                    if len(seq) <= self.hp.max_token_lens:
                        seq = np.asarray(seq)

                        # get len
                        wav_len = wav_id.shape[0]
                        text_len = text_id.shape[0]

                        # get position embedding
                        pos_id = np.asarray(list(range(text_len + 2)) + list(range(wav_len + 1)))
                        seq_sen_id = np.asarray([2] * (text_len + 2) + [1] * (wav_len + 1))
                        return seq, pos_id, seq_sen_id, None
            else:
                raise NotImplementedError

class Collator(object):
    def __init__(self, tokenizer_pad, block_sparse = False):
        self.pad = tokenizer_pad
        self.block_sparse = block_sparse

    def __call__(self, batches):
        results = batches

        # length padding
        seqs = []
        seq_lens = []
        seq_sen_ids = []
        pos_ids = []
        uttids = []

        max_seq_len = max(seq.shape[0] for seq, _, _, _ in results)
        if self.block_sparse:
            max_seq_len = (max_seq_len // 32 + 1) * 32
        for seq, pos_id, seq_sen_id, uttid in results:
            seq_lens.append(seq.shape[0])
            # position id
            pos_id = np.pad(pos_id, (0, max_seq_len - seq.shape[0]), mode='constant', constant_values=self.pad)
            # 区分是text还是wav
            seq_sen_id = np.pad(seq_sen_id,
                                (0, max_seq_len - seq.shape[0]),
                                mode='constant',
                                constant_values=self.pad)
            seq = np.pad(seq,
                            (0, max_seq_len - seq.shape[0]),
                            mode='constant',
                            constant_values=self.pad)
            seqs.append(seq)
            pos_ids.append(pos_id)
            seq_sen_ids.append(seq_sen_id)
            uttids.append(uttid)

        # to numpy
        seqs = np.asarray(seqs)
        seq_lens = np.asarray(seq_lens)
        pos_ids = np.asarray(pos_ids)
        seq_sen_ids = np.asarray(seq_sen_ids)
        # to torch
        seqs = torch.from_numpy(seqs)
        seq_lens = torch.from_numpy(seq_lens)
        pos_ids = torch.from_numpy(pos_ids)
        seq_sen_ids = torch.from_numpy(seq_sen_ids)

        return seqs, seq_lens, pos_ids, seq_sen_ids, uttids

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm
    hp = {
        'audio_tokens_num': 1000,
        'max_token_lens': 2048,
        'audio_token_dedup': True,
        'text_tokenizer': 'bpe',
        'bpe_tokenizer_file': '/mnt/bn/ming-dev/data/bpe_tokenizer/train960_bpe1000.json'
    }
    dataset = GPT2ASRDataset('/mnt/bn/ming-dev/data/Giga_LL_LS_merge.json', DotDict(hp), inference=False, task = 'joint')
    collator = Collator(0)
    dataloader = DataLoader(dataset, batch_size=16,num_workers=0,collate_fn=collator, shuffle=False, drop_last=False)
    sample_lens_min = 10000
    sample_lens_max = 0
    for batch in tqdm.tqdm(dataloader):
        assert batch[0].shape[1] == torch.max(batch[1])
        if batch[0].shape[1] > 1000:
            print(batch[0].shape[1])
        if batch[0].shape[1] < sample_lens_min:
            sample_lens_min = batch[0].shape[1]
        if batch[0].shape[1] > sample_lens_max:
            sample_lens_max = batch[0].shape[1]

    print(sample_lens_min, sample_lens_max)