from typing import List
import random
import torch
import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset

from samantha.utils.hparams import DotDict
from recipes.asr_tts_qa.utils.hdfs_tools import hdfs_open
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
                       tokenizer,
                       frontend,
                       batch_size=16,
                       max_token_per_batch=0, 
                       hp=None, 
                       task='joint', 
                       inference=False, 
                       block_sparse=False, 
                       train_len=None, 
                       shuffle_buf=10000, 
                       sort_buff=10000, 
                       epoch_length=1e5):
        """
        epoch length: should be close  to total data samples / (world_size * num_workers * batch_size), but it's not necessary if epoch number is not used during the training.
        """
        tar_paths = hdfs_open(datalist).read().splitlines()
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

        self.tokenizer = tokenizer
        

    def _process_data(self, data):
        utt_id = data['__key__']
        sample = data['audio.npy']
        meta = data['meta.json']
        """
        meta:
            data_source
            text: for data without paired data,this field is None
            sr: sampling rate
            duration: length of audio
            spk_id: for data without paired data,this field is None
            other: MORE for other dataset
        for example: {'data_source': 'Librilight', 'text': None, 'sr': 16000, 'duration': 51.04, 'spk_id': '5304'}
        """

        


        # sample = data['sample.pyd']
        # wav_id, text_id, uttid = np.array(self.tokenizer.tokenize(sample['inputs'], 'inputs')), np.array(self.tokenizer.tokenize(sample['targets'], 'targets')), sample['uttid'] if 'uttid' in sample else None

        # if self.inference:
        #     if self.task == 'asr':
        #         seq = [self.tokenizer.bos] + list(wav_id) + [self.tokenizer.sep]
        #         if len(seq) <= self.hp.max_token_lens:
        #             seq = np.asarray(seq)

        #             # get len
        #             wav_len = wav_id.shape[0]

        #             # get position embedding
        #             pos_id = np.asarray(list(range(wav_len + 2)))
        #             seq_sen_id = np.asarray([1] * (wav_len + 2))
        #             return seq, pos_id, seq_sen_id, uttid
        #     elif self.task == 'tts':
        #         seq = [self.tokenizer.bos] + list(text_id) + [self.tokenizer.sep]
        #         if len(seq) <= self.hp.max_token_lens:
        #             seq = np.asarray(seq)

        #             # get len
        #             text_len = text_id.shape[0]

        #             # get position embedding
        #             pos_id = np.asarray(list(range(text_len + 2)))
        #             seq_sen_id = np.asarray([2] * (text_len + 2))
        #             return seq, pos_id, seq_sen_id, uttid
        #     else:
        #         raise NotImplementedError
        # else:
        #     if self.task == 'asr':
        #         seq = [self.tokenizer.bos] + list(wav_id) + [self.tokenizer.sep] + list(text_id) + [self.tokenizer.eos]
        #         if len(seq) <= self.hp.max_token_lens:
        #             seq = np.asarray(seq)

        #             # get len
        #             wav_len = wav_id.shape[0]
        #             text_len = text_id.shape[0]

        #             # get position embedding
        #             pos_id = np.asarray(list(range(wav_len + 2)) + list(range(text_len + 1)))
        #             seq_sen_id = np.asarray([1] * (wav_len + 2) + [2] * (text_len + 1))
        #             return seq, pos_id, seq_sen_id, None
        #     elif self.task == 'tts':
        #         seq = [self.tokenizer.bos] + list(text_id) + [self.tokenizer.sep] + list(wav_id) + [self.tokenizer.eos]
        #         if len(seq) <= self.hp.max_token_lens:
        #             seq = np.asarray(seq)

        #             # get len
        #             wav_len = wav_id.shape[0]
        #             text_len = text_id.shape[0]

        #             # get position embedding
        #             pos_id = np.asarray(list(range(text_len + 2)) + list(range(wav_len + 1)))
        #             seq_sen_id = np.asarray([2] * (text_len + 2) + [1] * (wav_len + 1))
        #             return seq, pos_id, seq_sen_id, None
        #     elif self.task == 'joint':
        #         prob = random.random()
        #         if prob > 0.5:
        #             seq = [self.tokenizer.bos] + list(wav_id) + [self.tokenizer.sep] + list(text_id) + [self.tokenizer.eos]
        #             if len(seq) <= self.hp.max_token_lens:
        #                 seq = np.asarray(seq)

        #                 # get len
        #                 wav_len = wav_id.shape[0]
        #                 text_len = text_id.shape[0]

        #                 # get position embedding
        #                 pos_id = np.asarray(list(range(wav_len + 2)) + list(range(text_len + 1)))
        #                 seq_sen_id = np.asarray([1] * (wav_len + 2) + [2] * (text_len + 1))
        #                 return seq, pos_id, seq_sen_id, None
        #         else:
        #             seq = [self.tokenizer.bos] + list(text_id) + [self.tokenizer.sep] + list(wav_id) + [self.tokenizer.eos]
        #             if len(seq) <= self.hp.max_token_lens:
        #                 seq = np.asarray(seq)

        #                 # get len
        #                 wav_len = wav_id.shape[0]
        #                 text_len = text_id.shape[0]

        #                 # get position embedding
        #                 pos_id = np.asarray(list(range(text_len + 2)) + list(range(wav_len + 1)))
        #                 seq_sen_id = np.asarray([2] * (text_len + 2) + [1] * (wav_len + 1))
        #                 return seq, pos_id, seq_sen_id, None
        #     else:
        #         raise NotImplementedError


    def _collation_fn(self, batch):
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
    from recipes.asr_tts_qa.datasets.tokenizer import CharacterTokenizerWithAudioTokens, BPETokenizerWithAudioTokens
    hp = {"audio_tokens_num": 1000, "audio_token_dedup": True, 
            "text_tokenizer": "bpe", "bpe_tokenizer_file": "/mnt/bn/ming-dev/data/bpe_tokenizer/train960_bpe1000.json", 
            "max_token_lens": 2048}
    dataset = TrainWebDataset(
        datalist="/mnt/bn/cyz-lq-nas/project/samantha/recipes/asr_tts_qa/datasets/debug_detalist.lst",
        tokenizer = BPETokenizerWithAudioTokens(hp['bpe_tokenizer_file'], audio_tokens_num=hp['audio_tokens_num'], audio_token_dedup=hp['audio_token_dedup']),
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

    # import webdataset
    # import soundfile as sf

    # def process_data(data):
    #     utt_id = data['__key__']
    #     sample = data['audio.npy']
    #     meta = data['meta.json']
    #     return utt_id,sample,meta

    # # You can access a list of tar files
    # urls = [
    #     "hdfs://haruna/home/byte_speech_sv/user/rui.xia/misc/webdataset_test/librispeech_separate-00000.tar",
    #     "hdfs://haruna/home/byte_speech_sv/user/rui.xia/misc/webdataset_test/librispeech_separate-00001.tar"
    # ]

    # # For tar files on HDFS, we need to stream them through the HDFS client
    # # Skip this line if your tar files live on local disk or ByteNAS
    # urls = [f"pipe:hdfs dfs -cat {x}" for x in urls]
    # dataset = webdataset.WebDataset(urls).decode().map(process_data)

    # # Iterate through every item in the dataset (note: don't actually run this in CLI!)
    # for i, (utt_id, sample_audio, sample_meta) in enumerate(dataset):

    #     #utt id
    #     print(utt_id)
    #     print(sample_meta)
    #     print(sample_audio)
    #     print(sample_audio.max())
        

    

