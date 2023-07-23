import os
import torch
import numpy as np
from torch.utils.data import Dataset

from samantha.utils.hparams import DotDict
import json
import math
import random

class PhoneTokenizerWithAudioTokens:
    def __init__(self, phone_token_num, audio_token_num) -> None:
        self.audio_token_num = audio_token_num
        self.phone_token_num = phone_token_num
        self.vocab_size = (
            audio_token_num + phone_token_num + 3
        )  # <s> </s>, <sep>, <pad>
        self.pad = 0
        self.bos = self.vocab_size - 1
        self.eos = self.vocab_size - 1
        self.sep = self.vocab_size - 2

    def tokenize(self, inputs, input_key): 
        if input_key == "inputs":  # text_id
            if inputs.max() >= self.phone_token_num:
                # print(inputs, ' is OOV, ignore ...')
                return None
            return inputs + 1
        elif input_key == "targets":  # wav_id
            return inputs + 1 + self.phone_token_num
        else:
            return None


class ContinuousTTSDataset(Dataset):
    def __init__(self, path, hp=None, return_full_seq=False, inference=False, dynamic_batch_size=False):
        self.path = path
        self.hp = DotDict(hp)
        self.metas = self.get_metadata(path)
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            self.hp.phone_tokens_num, self.hp.audio_tokens_num
        )
        self.return_full_seq = return_full_seq
        self.inference = inference
        if dynamic_batch_size:
            self.seqlens = [int(x.split("|")[2]) + 2 for x in self.metas]
        else:
            self.seqlens = None

    def get_metadata(self, path):
        with open(path, "r") as f:
            metas = [l.strip() for l in f]
        return metas

    def _norm(self, x):
        x = torch.from_numpy(x)
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        norm = torch.norm(x, dim=-1, keepdim=True) * 16**-0.5
        x = x / norm.clamp(min=1e-8)
        return x.numpy()

    def get_text_wavid(self, idx):
        if self.inference:
            x = self.metas[idx].split("|")
            if len(x) == 3:
                bn_path, prompt_text_id_path, text_id_path = x[0], x[1], x[2]
                bn = np.load(bn_path)
                if len(bn.shape) == 3 and bn.shape[0] == 1:
                    bn = bn[0]
                # load text id
                prompt_text_id = np.load(prompt_text_id_path)

                text_id = np.load(text_id_path)
                text_id = np.concatenate((prompt_text_id[:-1], np.array([2]), text_id[1:]))
                uttid = os.path.splitext(os.path.basename(text_id_path))[0]

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

            elif len(x) == 2:
                # 无prompt模式的inference
                bn_path, text_id_path = x[0], x[1]
                uttid = os.path.splitext(os.path.basename(text_id_path))[0]
                bn = np.load(bn_path)
                if len(bn.shape) == 3 and bn.shape[0] == 1:
                    bn = bn[0]
                bn = np.zeros_like(bn)[:1, :]
                text_id = np.load(text_id_path)
        else:
            x = self.metas[idx].split("|")
            bn_path, text_id_path = x[0], x[1]
            uttid = os.path.splitext(os.path.basename(text_id_path))[0]
            bn = np.load(bn_path)
            if len(bn.shape) == 3 and bn.shape[0] == 1:
                bn = bn[0]
            text_id = np.load(text_id_path)

        text_id = self.tokenizer.tokenize(text_id, "inputs")

        return text_id, bn, uttid

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        text_id, bn, uttid = self.get_text_wavid(idx)
        
        if text_id is None:
            print(uttid, ' text_id is None')
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
            print('exceed len!')
            return

        full_seq = None

        return text_id, bn, seq, pos_id, seq_sen_id, full_seq, uttid


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
        pos_ids = []
        full_seqs = []
        utt_ids = []

        if len(results) == 0:
            return None

        max_seq_len = max(seq.shape[0] for _, _, seq, _, _, _, _ in results)
        max_text_id_len = max(text_id.shape[0] for text_id, _, _, _, _, _, _ in results)
        max_bn_len = max(bn.shape[0] for _, bn, _, _, _, _, _ in results)
        if self.block_sparse:
            max_seq_len = (max_seq_len // 32 + 1) * 32
        for text_id, bn, seq, pos_id, seq_sen_id, full_seq, utt_id in results:
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

            # position id
            pos_id = np.pad(
                pos_id,
                (0, max_seq_len - seq.shape[0]),
                mode="constant",
                constant_values=self.pad,
            )
            # 区分是text还是wav
            seq_sen_id = np.pad(
                seq_sen_id,
                (0, max_seq_len - seq.shape[0]),
                mode="constant",
                constant_values=self.pad,
            )
            seq = np.pad(
                seq,
                (0, max_seq_len - seq.shape[0]),
                mode="constant",
                constant_values=self.pad,
            )
            if full_seq is not None:
                full_seq = np.pad(
                    full_seq,
                    [(0, max_seq_len - full_seq.shape[0]), (0, 0)],
                    mode="constant",
                    constant_values=0,
                )  # [t, n_codebook]
                full_seqs.append(full_seq)

            text_ids.append(text_id)
            bns.append(bn)
            seqs.append(seq)
            pos_ids.append(pos_id)
            seq_sen_ids.append(seq_sen_id)
            utt_ids.append(utt_id)


        # to numpy
        text_ids = np.asarray(text_ids)
        text_id_lens = np.asarray(text_id_lens)
        bns = np.stack(bns, axis=0)
        # print('logs+=3')
        # # bns[:, :, :32] = bns[:, :, :32]/10
        # bns[:, :, 32:] = bns[:, :, 32:] + 3
        # print('logs+=2.5')
        # bns[:, :, 32:] = bns[:, :, 32:] + 2.5
        bn_lens = np.asarray(bn_lens)
        seqs = np.asarray(seqs)
        seq_lens = np.asarray(seq_lens)
        pos_ids = np.asarray(pos_ids)
        seq_sen_ids = np.asarray(seq_sen_ids)
        full_seqs = np.array(full_seqs) if len(full_seqs) > 0 else None

        # to torch
        text_ids = torch.from_numpy(text_ids)
        text_id_lens = torch.from_numpy(text_id_lens)
        bns = torch.from_numpy(bns)
        bn_lens = torch.from_numpy(bn_lens)
        seqs = torch.from_numpy(seqs)
        seq_lens = torch.from_numpy(seq_lens)
        pos_ids = torch.from_numpy(pos_ids)
        seq_sen_ids = torch.from_numpy(seq_sen_ids)
        full_seqs = torch.from_numpy(full_seqs) if full_seqs is not None else None

        return text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs, utt_ids
