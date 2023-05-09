from typing import List
import random
import torch
import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset

from recipes.llm_asr.datasets.dataset import CharacterTokenizerWithAudioTokens, BPETokenizerWithAudioTokens
from samantha.utils.hparams import DotDict
from recipes.llm_asr.utils.hdfs_tools import hdfs_open
from webdataset.filters import pipelinefilter

def sort_buf(data, buf_size=1000):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= buf_size:
            buf.sort(key=lambda x: x[0].shape[0], reverse=False)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x[0].shape[0], reverse=False)
    for x in buf:
        yield x

def token_batched(
    data,
    max_token_size=100,
    collation_fn=None,
):
    """Create batches of the given size.
    :param data: iterator
    :param batchsize: target batch size
    :param tensors: automatically batch lists of ndarrays into ndarrays
    :param partial: return partial batches
    :returns: iterator
    """
    batch = []
    longest_lens = 0
    for sample in data:
        new_lens = sample[0].shape[0]
        longest_lens = max(longest_lens, new_lens)
        len_after_padding = longest_lens * (len(batch) + 1)
        if len_after_padding > max_token_size:
            batch.sort(key=lambda x: x[0].shape[0], reverse=True)
            if len(batch) >= 1:
                if collation_fn is not None:
                    batch = collation_fn(batch)
                yield batch
            batch = [sample]
            longest_lens = new_lens
        else:
            batch.append(sample)
    if len(batch) >= 1:
        if collation_fn is not None:
            batch = collation_fn(batch)
        yield batch

token_batched = pipelinefilter(token_batched)
wds_sort = pipelinefilter(sort_buf)

class newWebDataset(wds.WebDataset):
    def __init__(self, urls, **kwargs):
        super().__init__(urls, **kwargs)

    def sort(self, size, **kw):
        if size < 1:
            return self
        else:
            return self.compose(wds_sort(size, **kw))
    
    def max_token_batched(self, batchsize, collation_fn):
        return self.compose(
            token_batched(batchsize, collation_fn=collation_fn)
        )


class TrainWebDataset(IterableDataset):
    def __init__(self, datalist,  
                       batch_size=16,
                       max_token_per_batch=0, 
                       hp=None, 
                       task='joint', 
                       inference=False, 
                       block_sparse=False, 
                       train_len=None, 
                       shuffle_buf=10000, 
                       sort_buff=10000, 
                       epoch_length=1e5,
                       semi_prob=1.0):
        """
        epoch length: should be close  to total data samples / (world_size * num_workers * batch_size), but it's not necessary if epoch number is not used during the training.
        """
        if isinstance(datalist, str):
            tar_paths = hdfs_open(datalist).read().splitlines()
        elif isinstance(datalist, list):
            tar_paths = []
            for item in datalist:
                tar_paths.extend(hdfs_open(item).read().splitlines())
        else:
            raise NotImplementedError

        assert len(tar_paths) > 0
        if tar_paths[0].startswith('hdfs'): # hdfs path
            urls = [f"pipe:hdfs dfs -cat {item}" for item in tar_paths]
        else: # local path
            urls = [f"pipe:cat {item}" for item in tar_paths]
        self.dataset = newWebDataset(urls, resampled=True).decode().map(self._process_data).shuffle(shuffle_buf)

        if sort_buff > 0:
            self.dataset = self.dataset.sort(sort_buff)
        if max_token_per_batch > 0:
            self.dataset = self.dataset.max_token_batched(max_token_per_batch, collation_fn=self._collation_fn)
        else:
            self.dataset = self.dataset.batched(batch_size, collation_fn=self._collation_fn)
        
        if epoch_length > 0:
            self.dataset = self.dataset.with_epoch(epoch_length)

        self.task = task
        self.hp = DotDict(hp)
        self.inference = inference
        self.block_sparse = block_sparse
        self.train_len = train_len
        # probability a non-parallel sample will be used for training; this can used to balance the amount of parallel and non-parallel data
        self.semi_prob = semi_prob

        self.parallel_seen = 0
        self.non_parallel_seen = 0

        if self.hp.text_tokenizer == 'char':
            self.tokenizer = CharacterTokenizerWithAudioTokens(audio_tokens_num=self.hp.audio_tokens_num, audio_token_dedup=self.hp.audio_token_dedup)
        elif self.hp.text_tokenizer == 'bpe':
            self.tokenizer = BPETokenizerWithAudioTokens(self.hp.bpe_tokenizer_file, audio_tokens_num=self.hp.audio_tokens_num, audio_token_dedup=self.hp.audio_token_dedup)
        else:
            raise NotImplementedError

    def _process_data(self, data):

        sample = data['sample.pyd']
        wav_id = np.array(self.tokenizer.tokenize(sample['inputs'], 'inputs')) if 'inputs' in sample and sample['inputs'] != '' else None
        text_id = np.array(self.tokenizer.tokenize(sample['targets'], 'targets')) if 'targets' in sample and sample['targets'] != '' else None
        uttid = sample['uttid'] if 'uttid' in sample else None

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
                if wav_id is not None and text_id is not None:
                    self.parallel_seen += 1
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
                else: # non-parallel data
                    if random.random() <= self.semi_prob:
                        self.non_parallel_seen += 1
                        if wav_id is None:
                            if text_id.shape[0] > self.hp.max_token_lens - 2:
                                start = np.random.randint(0,text_id.shape[0]+2-self.hp.max_token_lens)
                                text_id = text_id[start:start + self.hp.max_token_lens -2]
                            seq = [self.tokenizer.bos] + list(text_id) + [self.tokenizer.eos]
                            seq = np.asarray(seq)
                            # get len
                            text_len = text_id.shape[0]
                            # get position embedding
                            pos_id = list(range(text_len + 2))
                            seq_sen_id = np.asarray([2] * (text_len + 2))
                            return seq, pos_id, seq_sen_id, None
                        elif text_id is None:
                            if wav_id.shape[0] > self.hp.max_token_lens - 2:
                                start = np.random.randint(0,wav_id.shape[0]+2-self.hp.max_token_lens)
                                wav_id = wav_id[start:start + self.hp.max_token_lens -2]
                            seq = [self.tokenizer.bos] + list(wav_id) + [self.tokenizer.eos]
                            seq = np.asarray(seq)
                            # get len
                            wav_len = wav_id.shape[0]
                            # get position embedding
                            pos_id = list(range(wav_len + 2))
                            seq_sen_id = np.asarray([1] * (wav_len + 2))
                            return seq, pos_id, seq_sen_id, None
                        else:
                            raise Exception('Wav id and text id can not both be None!')
            else:
                raise NotImplementedError


    def _collation_fn(self, batch):
        print("++++++++++{} parallel samples and {} non-parallel samples have been seen!++++++++++".format(self.parallel_seen, self.non_parallel_seen))
        # length padding
        seqs = []
        seq_lens = []
        seq_sen_ids = []
        pos_ids = []
        uttids = []

        max_seq_len = max(seq.shape[0] for seq, _, _, _ in batch)
        if self.block_sparse:
            if self.train_len is not None:
                max_seq_len = self.train_len
            else:
                max_seq_len = (max_seq_len // 32 + 1) * 32
        for seq, pos_id, seq_sen_id, uttid in batch:
            seq_lens.append(seq.shape[0])
            # position id
            pos_id = np.pad(pos_id, (0, max_seq_len - seq.shape[0]), mode='constant', constant_values=self.tokenizer.pad)
            # 区分是text还是wav
            seq_sen_id = np.pad(seq_sen_id,
                                (0, max_seq_len - seq.shape[0]),
                                mode='constant',
                                constant_values=self.tokenizer.pad)
            seq = np.pad(seq,
                            (0, max_seq_len - seq.shape[0]),
                            mode='constant',
                            constant_values=self.tokenizer.pad)
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

    

    def __iter__(self):
        return iter(self.dataset)

if __name__ == "__main__":
    import tqdm
    hp = {"audio_tokens_num": 1000, "audio_token_dedup": True, "text_tokenizer": "bpe", "bpe_tokenizer_file": "/mnt/bn/ming-dev/data/bpe_tokenizer/train960_bpe1000.json", "max_token_lens": 2048}
    dataset = TrainWebDataset(
        datalist='/mnt/bn/ming-dev/data/librispeech/train960_mhubert-km1000_wds/data.lst',
        hp = hp,
        task = 'joint',
        inference=False,
        batch_size = 16,
        epoch_length=100
    )
    loader = wds.WebLoader(dataset, num_workers=1, batch_size=None)
    for batch in tqdm.tqdm(loader):
        print(batch)
        break