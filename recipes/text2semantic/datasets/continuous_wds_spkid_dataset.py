import logging
import math
import pickle
import sys
import os 

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset.ra_wds import WebDataset
from samantha.utils.hparams import DotDict

from .continuous_dataset import ContinuousCollator, PhoneTokenizerWithAudioTokens
from recipes.text2semantic.datasets.sami_tacolabel import enc_taco_label_no_bytes
from recipes.text2semantic.utils.remote_io import load_json

import traceback

logger = logging.getLogger(__name__)



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ContinuousTTSDataset(IterableDataset):
    def __init__(self, wds_urls, phone_tokens_num, speaker_tokens_num, speaker_converter_dict, 
        return_full_seq=False, inference=False, drop_last=False, batcher_config=None):
        
        self.wds = (
            WebDataset(
                urls=wds_urls,
                resampled=True,
                # nodesplitter=wds.shardlists.split_by_node,
            )
            .decode()
            .shuffle(2048)
            .map(self.get_text_wavid)
        )
        self.drop_last = drop_last
        self.text_converter_dict = load_json('recipes/valle/datasets/dict/metaid_to_textid.json')
        self.speaker_converter_dict = load_json(speaker_converter_dict)
        
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            phone_tokens_num, speaker_tokens_num
        )
        self.return_full_seq = return_full_seq
        self.inference = inference
        self.batcher = BucketBatcher(**batcher_config)


    def get_text_wavid(self, sample):

        bn = pickle.loads(sample["bns"])
        text = sample["text"]
        lab = sample["labels"]
        utt_id = sample["__key__"]
        dataset_name = sample["dataset_name"]
        speaker_name = sample["speaker_name"]

        dataset_name = dataset_name.decode()
        speaker_name = speaker_name.decode()
        labels = lab.decode()
        
        spk_key = '/'.join([dataset_name, speaker_name])
        spk_id = self.speaker_converter_dict[spk_key]

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        text = text
        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split('\n')))
        # text_id, [text_len]
        text_id = self.convert_tacolab_to_text_id(labels)
        if text_id is None:
            logger.warning(f"{utt_id} convert_tacolab_to_text_id failed ...")
            return None

        text_id = self.tokenizer.tokenize(text_id, "inputs")
        spk_id = self.tokenizer.tokenize(spk_id, "targets")

        # get len
        bn_T, bn_C = bn.shape[0], bn.shape[1]
        text_len = text_id.shape[0]

        seq = (
            [self.tokenizer.bos]
            + list(text_id)
            + [self.tokenizer.sep]
            + [spk_id]
            + [0] * bn_T # place holder
            + [self.tokenizer.eos]
        )

        text_id = np.array(text_id)
        bn = np.array(bn)
        seq = np.asarray(seq)
        return text_id, bn, seq, text, utt_id

    def __iter__(self):
        for item in self.wds:
            batch = self.batcher.collate_batch(item)
            if batch:
                yield batch
        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield batch


    def convert_tacolab_to_text_id(self, tacolab):
        try:
            with HiddenPrints():
                metas = enc_taco_label_no_bytes(
                    None,
                    tacolab,
                    {"use_prsdword": False, "forced_refix": True})
            text_id =  metas[0].astype(np.int64) * 1_000_000_000 + \
                        metas[1].astype(np.int64) * 1_000_000 + \
                        metas[2].astype(np.int64) * 1_000 + \
                        metas[3].astype(np.int64)
            # TODO: remove hard-coded oov number
            text_id = np.asarray([self.text_converter_dict.get(str(x), self.text_converter_dict.get("oov")) for x in text_id]).astype(np.int64)
            return text_id
        except Exception as e:
            print(e)
            return None



if __name__ == "__main__":
    pass
    # import torch
    # import torch.distributed as dist
    # import torch.utils.data

    # dist.init_process_group(backend="nccl")
    # urls = "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tmp/wds_out/{00000..00024}.tar"
    # hp = {
    #     "phone_tokens_num": 200, 
    #     "speaker_tokens_num": 1024,
    # }
    # dataset = ContinuousTTSDataset(urls, hp=hp, dynamic_batch_size=True, drop_last=False)
    # collector = ContinuousCollator(tokenizer_pad=0)
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=None, collate_fn=collector)
    # import tqdm

    # # for item in tqdm.tqdm(dataset):
    # for item in tqdm.tqdm(dataloader):
    #     # print(item)
    #     lengths = item[-1]
    #     batch_size = len(lengths)
    #     print(batch_size, max(lengths), batch_size * max(lengths))