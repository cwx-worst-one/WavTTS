import logging
import math
import pickle

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset.ra_wds import WebDataset
from samantha.utils.hparams import DotDict

from .continuous_dataset import ContinuousCollator, PhoneTokenizerWithAudioTokens

logger = logging.getLogger(__name__)


class ContinuousTTSDataset(IterableDataset):
    def __init__(self, wds_urls, hp=None, return_full_seq=False, inference=False, drop_last=False, batcher_config=None):
        
        self.wds = (
            WebDataset(
                urls=wds_urls,
                resampled=True,
                # nodesplitter=wds.shardlists.split_by_node,
            )
            .decode()
            .shuffle(100)
            .map(self.get_text_wavid)
        )
        self.hp = DotDict(hp)
        self.drop_last = drop_last
        
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            self.hp.phone_tokens_num, self.hp.audio_tokens_num
        )
        self.return_full_seq = return_full_seq
        self.inference = inference
        self.batcher = BucketBatcher(**batcher_config)

    def _norm(self, x):
        x = torch.from_numpy(x)
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        norm = torch.norm(x, dim=-1, keepdim=True) * 16**-0.5
        x = x / norm.clamp(min=1e-8)
        return x.numpy()

    def get_text_wavid(self, sample):
        
        x = pickle.loads(sample["meta"])
        bn = x["bn"]
        text_id = x["text_id"]
        prompt_text_id = x.get("prompt_text_id", None)
        uttid = sample["uttid"].decode()
        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        if self.inference:
            if prompt_text_id is not None:
                text_id = np.concatenate((prompt_text_id[:-1], np.array([2]), text_id[1:]))

                # 测试noprompt 
                # bn_type = bn.dtype
                # bn = np.random.rand(1, bn.shape[1])
                # bn = self._norm(bn).astype(bn_type)
                # text_id = np.load(text_id_path)

                # # 测试noprompt vae
                # print('无prompt模式')
                # bn_type = bn.dtype
                # bn = np.zeros([0, 16]).astype(bn_type)
                # text_id = np.load(text_id_path)

            else:
                # 无prompt模式的inference
                bn = np.zeros_like(bn)[:1, :]

        text_id = self.tokenizer.tokenize(text_id, "inputs")
        
        if text_id is None:
            logger.warning(f"{uttid} text_id is None")
            return None

        # get len
        bn_T, bn_C = bn.shape[0], bn.shape[1]
        text_len = text_id.shape[0]

        if self.inference:
            seq = (
                [self.tokenizer.bos]
                + list(text_id)
                + [self.tokenizer.sep]
                + [0] * bn_T # place holder
            )
            pos_id = np.asarray(list(range(text_len + 2)) + list(range(bn_T)))
            seq_sen_id = np.asarray([1] * (text_len + 2) + [2] * (bn_T))
        else:
            seq = (
                [self.tokenizer.bos]
                + list(text_id)
                + [self.tokenizer.sep]
                + [0] * bn_T # place holder
                + [self.tokenizer.eos]
            )
            # get position embedding
            pos_id = np.asarray(list(range(text_len + 2)) + list(range(bn_T + 1)))
            seq_sen_id = np.asarray([1] * (text_len + 2) + [2] * (bn_T + 1))

        text_id = np.array(text_id)
        bn = np.array(bn)
        seq = np.asarray(seq)

        if seq.shape[-1] >= 4096:
            logger.warning('exceed len!')
            return

        full_seq = None

        return text_id, bn, seq, pos_id, seq_sen_id, full_seq, uttid

    def __iter__(self):
        for item in self.wds:
            batch = self.batcher.collate_batch(item)
            if batch:
                yield batch
        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield batch


if __name__ == "__main__":
    import torch
    import torch.distributed as dist
    import torch.utils.data

    dist.init_process_group(backend="nccl")
    urls = "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tmp/wds_out/{00000..00024}.tar"
    hp = {
        "phone_tokens_num": 200, 
        "audio_tokens_num": 1024,
    }
    dataset = ContinuousTTSDataset(urls, hp=hp, dynamic_batch_size=True, drop_last=False)
    collector = ContinuousCollator(tokenizer_pad=0)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=None, collate_fn=collector)
    import tqdm

    # for item in tqdm.tqdm(dataset):
    for item in tqdm.tqdm(dataloader):
        # print(item)
        lengths = item[-1]
        batch_size = len(lengths)
        print(batch_size, max(lengths), batch_size * max(lengths))