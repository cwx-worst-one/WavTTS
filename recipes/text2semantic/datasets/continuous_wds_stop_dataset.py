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

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset.ra_wds import WebDataset
from samantha.utils.hparams import DotDict

from recipes.text2semantic.datasets.continuous_dataset import ContinuousCollator, PhoneTokenizerWithAudioTokens
from recipes.text2semantic.datasets.sami_tacolabel import enc_taco_label_no_bytes
from recipes.text2semantic.utils.remote_io import load_json
from transformers import LlamaTokenizer
from zhon.hanzi import punctuation
import string
punctuation_all = punctuation + string.punctuation

import traceback

logger = logging.getLogger(__name__)



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ContinuousTTSStopDataset(IterableDataset):
    def __init__(self,
        wds_urls,
        phone_tokens_num,
        speaker_tokens_num,
        metaid_to_textid,
        simulated_cycle_rate=0.0,
        return_full_seq=False,
        inference=False,
        drop_last=False,
        batcher_config=None,
        textid_version='v1',
        spk2id=None,
        bpe_dir=None, 
        bpe_tokens_num=0,
        max_length=4096,
        use_bpe=False,
        use_spkid=False,
        use_sy=True):

        self.wds = (
            WebDataset(
                urls=wds_urls,
                resampled=True,
                # nodesplitter=wds.shardlists.split_by_node,
                skip_instance_cache=True,
            )
            .decode()
            .shuffle(2048)
            .map(self.get_text_wavid)
        )
        self.drop_last = drop_last
        self.text_converter_dict = load_json(metaid_to_textid)
        print(f"Loaded text_converter_dict from {metaid_to_textid}")
        self.max_length = max_length
        self.use_bpe = use_bpe
        self.use_spkid = use_spkid
        print(f"use_spkid: {use_spkid}")

        if self.use_bpe:
            print('##### Using BPE #####')
            self.bpe_tokenizer = LlamaTokenizer.from_pretrained(bpe_dir)
            _bpe_tokens_num = len(self.bpe_tokenizer)
        else:
            self.bpe_tokenizer = None
            _bpe_tokens_num = 0
        assert _bpe_tokens_num == bpe_tokens_num

        if self.use_spkid:
            self.spk2id = load_json(spk2id)
            print(f"Loaded spk2id from {spk2id}")
        else:
            self.spk2id = None

        self.tokenizer = PhoneTokenizerWithAudioTokens(
            phone_tokens_num,
            speaker_tokens_num,
            bpe_tokens_num=bpe_tokens_num
        )
        self.return_full_seq = return_full_seq
        self.inference = inference
        self.batcher = BucketBatcher(**batcher_config)
        self.simulated_cycle_rate = simulated_cycle_rate
        self.textid_version = textid_version
        self.use_sy = use_sy


    def get_text_wavid(self, sample):

        bn = pickle.loads(sample["bns"])
        text = sample["text"]
        lab = sample["labels"]
        utt_id = sample["__key__"]

        if len(bn.shape) == 3 and bn.shape[0] == 1:
            bn = bn[0]

        text = text
        # print("text: ", text)
        # print("utt_id: ", utt_id)
        labels = lab.decode()
        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split('\n')))

        if self.use_spkid:
            dataset_name = sample.get("dataset_name")
            speaker_name = sample.get("speaker_name")
            if not dataset_name or not speaker_name:
                print(f"{utt_id}: No speaker name")
                return None

            dataset_name = dataset_name.decode()
            speaker_name = speaker_name.decode()

            spk_key = '/'.join([dataset_name, speaker_name])
            spk_id = self.spk2id[spk_key]
            spk_id = self.tokenizer.tokenize(spk_id, "spk")
            # print("spk_id: ", spk_id)

        # text_id, [text_len]
        text_id = self.convert_tacolab_to_text_id(labels)
        if text_id is None:
            logger.warning(f"{utt_id} convert_tacolab_to_text_id failed ...")
            return None

        text_id = self.tokenizer.tokenize(text_id, "inputs")

        bn_T, bn_C = bn.shape[0], bn.shape[1]
        text_len = text_id.shape[0]

        # bpe_id
        if self.use_bpe:
            bpe_id = np.asarray(self.bpe_tokenizer(
                text, truncation=True, max_length=self.max_length,
            ).input_ids)
            bpe_id = self.tokenizer.tokenize(bpe_id, "bpe")

            # text_id = np.concatenate([text_id, [self.tokenizer.sep], bpe_id])
            text_id = np.concatenate([
                bpe_id, 
                [self.tokenizer.sep], 
                text_id])

        if self.use_spkid:
            wav_id = np.concatenate([
                [spk_id], 
                [0] * bn_T])
        else:
            wav_id = np.zeros([bn_T], dtype=np.int64)

        seq = (
            [self.tokenizer.bos]
            + list(text_id)
            + [self.tokenizer.sep]
            + list(wav_id)
            + [self.tokenizer.eos]
        )

        stop_token = np.zeros([len(seq)])
        stop_token[-1] = 1

        text_id = np.array(text_id)
        bn = np.array(bn)
        seq = np.asarray(seq)
        return text_id, bn, seq, text, utt_id, stop_token

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
            if self.textid_version == 'v1':
                with HiddenPrints():
                    if len(tacolab[0].split('\t')) != 5:
                        tacolab = self.convert_v3_to_v1(tacolab)
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
                # print("text_id: ", text_id)
            elif self.textid_version == 'v2':
                if len(tacolab[0].split('\t')) != 5:
                    tacolab = self.convert_v3_to_v1(tacolab)
                labs = []
                for i in range(len(tacolab)):
                    x = tacolab[i]
                    if i != 0 and x.split('\t')[0] == 'sil':
                        continue
                    phone, tone, _, ws, pw = x.split('\t')
                    lab = '_'.join([phone, tone, ws, pw])
                    labs.append(lab)
                # print("labs: ", labs)
                text_id = []
                for x in labs:
                    if self.text_converter_dict.get(x) == None:
                        print("x: ", x)
                    cur_text_id = self.text_converter_dict.get(x, self.text_converter_dict.get("oov"))
                    # if cur_text_id is None:
                    #     print("x: ", x)
                    #     exit()
                    text_id.append(cur_text_id)
                text_id = np.asarray(text_id).astype(np.int64)
            elif self.textid_version == 'v3':
                # if len(tacolab[0].split('\t')) != 5:
                #     tacolab = self.convert_v3_to_v1(tacolab)
                lang = self.get_lang(tacolab)
                # print("lang: ", lang)
                if lang in ['zh_en', 'zh']:
                    assert len(tacolab[0].split('\t')) == 7, (len(tacolab[0].split('\t')), tacolab[0])
                    if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
                        tacolab = tacolab[1:]
                    labs = []
                    for i in range(len(tacolab)):
                        x = tacolab[i]
                        if i != 0 and x.split('\t')[0] == 'sil':
                            continue
                        x_split = x.split('\t')
                        phone, tone, ws, pw, stype, word, unit = x_split
                        lab = '_'.join([phone, tone])
                        if (phone in ['sil', 'sp'] or phone in punctuation_all or phone[0] in punctuation_all):
                            if len(labs) >= 1:
                                if labs[-1] in ["ws", "sy_zh"]:
                                    labs[-1] = lab
                            else:
                                labs.append(lab)
                            continue
                        else:
                            labs.append(lab)
                        
                        syllable_flag = False
                        word_segment = False
                        unit = unit.strip()
                        if phone[:2] == 'C0':
                            if unit in ['S', 'E']:
                                if self.use_sy:
                                    syllable_flag = True
                                if ws in ['S', 'E']:
                                    word_segment = True
                        elif phone[:2] == 'E0':
                            if pw != '0':
                                word_segment = True
                        else:
                            print("Wrong phone", phone)
                            return None

                        if word_segment:
                            labs.append("ws")
                        elif syllable_flag:
                            labs.append("sy_zh")
                elif lang == 'en':
                    if len(tacolab[0].split('\t')) != 5:
                        tacolab = self.convert_v3_to_v1(tacolab)
                    labs = []
                    for i in range(len(tacolab)):
                        x = tacolab[i]
                        if i != 0 and x.split('\t')[0] == 'sil':
                            continue
                        x_split = x.split('\t')
                        phone, tone, _, ws, pw = x.split('\t')
                        lab = '_'.join([phone, tone])
                        # print("lab: ", lab)
                        # print("phone: ", phone)
                        # print("labs: ", labs)
                        # labs.append(lab)
                        if (phone in ['sil', 'sp'] or phone in punctuation_all or phone[0] in punctuation_all):
                            if len(labs) >= 1:
                                if labs[-1] == "ws":
                                    labs[-1] = lab
                            else:
                                labs.append(lab)
                            continue
                        else:
                            labs.append(lab)

                        word_segment = False
                        if pw != '0':
                            word_segment = True

                        if word_segment:
                            labs.append("ws")

                # print("labs: ", labs)
                text_id = []
                for x in labs:
                    cur_text_id = self.text_converter_dict.get(x, self.text_converter_dict.get("oov"))
                    # if cur_text_id is None:
                    #     print("x: ", x)
                    #     exit()
                    text_id.append(cur_text_id)
                text_id = np.asarray(text_id).astype(np.int64)
            return text_id
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

        # length padding
        text_ids = []
        text_id_lens = []
        bns = []
        bn_lens = []
        seqs = []
        seq_lens = []
        seq_sen_ids = []
        utt_ids = []
        stop_tokens = []

        if len(results) == 0:
            return None

        # text_id, bn, seq, text, utt_id

        max_seq_len = max(seq.shape[0] for text_id, bn, seq, text, utt_id, stop_token in results)
        max_text_id_len = max(text_id.shape[0] for text_id, bn, seq, text, utt_id, stop_token in results)
        max_bn_len = max(bn.shape[0] for text_id, bn, seq, text, utt_id, stop_token in results)
        if self.block_sparse:
            max_seq_len = (max_seq_len // 32 + 1) * 32
        for text_id, bn, seq, text, utt_id, stop_token in results:
            # 暂时还不用text

            seq_lens.append(seq.shape[0])
            text_id_lens.append(text_id.shape[0])
            bn_lens.append(bn.shape[0])

            text_id = np.pad(
                text_id,
                (0, max_text_id_len - text_id.shape[0]),
                mode="constant",
                constant_values=self.pad,
            )

            bn = np.pad(
                bn,
                ((0, max_bn_len - bn.shape[0]), (0, 0)),
                mode="constant",
                constant_values=self.pad,
            )

            seq = np.pad(
                seq,
                (0, max_seq_len - seq.shape[0]),
                mode="constant",
                constant_values=self.pad,
            )

            stop_token = np.pad(
                    stop_token,
                    (0, max_seq_len - stop_token.shape[0]),
                    mode="constant",
                    constant_values=self.pad,
                )
            
            text_ids.append(text_id)
            bns.append(bn)
            seqs.append(seq)
            utt_ids.append(utt_id)
            stop_tokens.append(stop_token)


        # to numpy
        text_ids = np.asarray(text_ids)
        text_id_lens = np.asarray(text_id_lens)
        bns = np.stack(bns, axis=0)
        bn_lens = np.asarray(bn_lens)
        seqs = np.asarray(seqs)
        seq_lens = np.asarray(seq_lens)
        stop_tokens = np.asarray(stop_tokens)


        # to torch
        text_ids = torch.from_numpy(text_ids)
        text_id_lens = torch.from_numpy(text_id_lens)
        bns = torch.from_numpy(bns)
        bn_lens = torch.from_numpy(bn_lens)
        seqs = torch.from_numpy(seqs)
        seq_lens = torch.from_numpy(seq_lens)
        stop_tokens = torch.from_numpy(stop_tokens).long()

        return text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, utt_ids, stop_tokens



if __name__ == "__main__":
    # pass
    # import torch
    # import torch.distributed as dist
    # import torch.utils.data

    # dist.init_process_group(backend="nccl")
    # hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/11labs/*/chunk*/*.tar hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/duibiao/*/chunk*/*.tar 
    urls = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/chenyuanzhe/WFVAE_v2_fixtrim/librilight/chunk*/*.tar"

    from samantha.dataio.utils import parse_data_urls
    wds_urls = parse_data_urls(data_urls=urls)
    hp = {
        "phone_tokens_num": 7370, 
        "speaker_tokens_num": 8192,
    }
    batcher_config = {
        "buckets": list(range(0, 4100, 100)),  # [0, 100, 200 ... 4000] 4000以上的可以先丢掉
        "dynamic_batch": True,
        "maximum_bucket_size": 200,
        "length_fn": "lambda x: x[2].shape[0]" # seq.shape
    }
    dataset = ContinuousTTSDataset(wds_urls, 
                                phone_tokens_num=7370,
                                speaker_tokens_num=8192,
                                max_length=4096,
                                batcher_config=batcher_config,
                                metaid_to_textid='recipes/text2semantic/datasets/dict/metaid_to_textid.v2.en_zh.json',
                                textid_version='v2',
                                bpe_dir='recipes/text2semantic/datasets/dict/llama_7B_tokenizer',
                                bpe_tokens_num=0,
                                spk2id='recipes/text2semantic/datasets/dict/sft_spk_0812.json',
                                use_bpe=False,
                                use_spkid=False,
                                drop_last=False,
                                use_sy=True)

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
