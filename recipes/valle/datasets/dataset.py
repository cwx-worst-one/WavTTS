import os
import torch
import numpy as np
from torch.utils.data import Dataset

from samantha.utils.hparams import DotDict


class PhoneTokenizerWithAudioTokens:
    def __init__(self, phone_token_num, audio_token_num) -> None:
        self.audio_token_num = audio_token_num
        self.phone_token_num = phone_token_num
        self.vocab_size = (
            audio_token_num + phone_token_num + 3
        )  #  <s> </s>, <sep>, <pad>
        self.pad = 0
        self.bos = self.vocab_size - 1
        self.eos = self.vocab_size - 1
        self.sep = self.vocab_size - 2

    def tokenize(self, inputs, input_key):
        if input_key == "inputs":  # text_id
            return inputs + 1
        elif input_key == "targets":  # wav_id
            return inputs + 1 + self.phone_token_num
        else:
            return None

class GPT2TTSDataset(Dataset):
    def __init__(self, path, hp=None, return_full_seq=False, inference=False):
        self.path = path
        self.hp = DotDict(hp)
        self.metas = self.get_metadata(path)
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            self.hp.phone_tokens_num, self.hp.audio_tokens_num
        )
        self.return_full_seq = return_full_seq
        self.inference = inference

    def get_metadata(self, path):
        with open(path, "r") as f:
            metas = [l.strip() for l in f]
        return metas

    def get_text_wavid(self, idx):
        if self.inference:
            x = self.metas[idx].split("|")
            wav_id_path, prompt_text_id_path, text_id_path = x[0], x[1], x[2]
            wav_id = np.load(wav_id_path)
            # load text id
            prompt_text_id = np.load(prompt_text_id_path)
            text_id = np.load(text_id_path)
            text_id = np.concatenate((prompt_text_id[:-1], np.array([2]), text_id[1:]))
            uttid = os.path.splitext(os.path.basename(text_id_path))[0]
        else:
            x = self.metas[idx].split("|")
            wav_id_path, text_id_path = x[0], x[1]
            uttid = os.path.splitext(os.path.basename(text_id_path))[0]
            wav_id = np.load(wav_id_path)
            text_id = np.load(text_id_path)

        text_id = self.tokenizer.tokenize(text_id, "inputs")
        wav_id = self.tokenizer.tokenize(wav_id, "targets")

        return text_id, wav_id, uttid

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        text_id, wav_id, uttid = self.get_text_wavid(idx)
        # get len
        wav_len, num_rvqs = wav_id.shape[0], wav_id.shape[1]
        text_len = text_id.shape[0]

        if self.inference:
            seq = (
                [self.tokenizer.bos]
                + list(text_id)
                + [self.tokenizer.sep]
                + list(wav_id[:, 0])
            )
            pos_id = np.asarray(list(range(text_len + 2)) + list(range(wav_len)))
            seq_sen_id = np.asarray([1] * (text_len + 2) + [2] * (wav_len))
        else:
            seq = (
                [self.tokenizer.bos]
                + list(text_id)
                + [self.tokenizer.sep]
                + list(wav_id[:, 0])
                + [self.tokenizer.eos]
            )
            # get position embedding
            pos_id = np.asarray(list(range(text_len + 2)) + list(range(wav_len + 1)))
            seq_sen_id = np.asarray([1] * (text_len + 2) + [2] * (wav_len + 1))
        
        seq = np.asarray(seq)

        if self.return_full_seq:
            # concat text & wav, add EOS, full codebook
            full_text_seq = np.stack([text_id] * num_rvqs, axis=1)
            sep = np.stack([np.asarray([self.tokenizer.sep])] * num_rvqs, axis=1)
            bos = np.stack([np.asarray([self.tokenizer.bos])] * num_rvqs, axis=1)
            eos = np.stack([np.asarray([self.tokenizer.eos])] * num_rvqs, axis=1)
            if self.inference:
                full_seq = np.concatenate(
                    [bos, full_text_seq, sep, wav_id], axis=0
                )
            else:
                full_seq = np.concatenate(
                    [bos, full_text_seq, sep, wav_id, eos], axis=0
                )
        else:
            full_seq = None

        return seq, pos_id, seq_sen_id, full_seq, uttid

class ValleCollator(object):
    def __init__(self, tokenizer_pad, block_sparse=False):
        self.pad = tokenizer_pad
        self.block_sparse = block_sparse

    def __call__(self, batches):
        results = batches

        # length padding
        seqs = []
        seq_lens = []
        seq_sen_ids = []
        pos_ids = []
        full_seqs = []
        utt_ids = []

        max_seq_len = max(seq.shape[0] for seq, _, _, _, _ in results)
        if self.block_sparse:
            max_seq_len = (max_seq_len // 32 + 1) * 32
        for seq, pos_id, seq_sen_id, full_seq, utt_id in results:
            seq_lens.append(seq.shape[0])
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

            seqs.append(seq)
            pos_ids.append(pos_id)
            seq_sen_ids.append(seq_sen_id)
            utt_ids.append(utt_id)

        # to numpy
        seqs = np.asarray(seqs)
        seq_lens = np.asarray(seq_lens)
        pos_ids = np.asarray(pos_ids)
        seq_sen_ids = np.asarray(seq_sen_ids)
        full_seqs = np.array(full_seqs) if len(full_seqs) > 0 else None

        # to torch
        seqs = torch.from_numpy(seqs)
        seq_lens = torch.from_numpy(seq_lens)
        pos_ids = torch.from_numpy(pos_ids)
        seq_sen_ids = torch.from_numpy(seq_sen_ids)
        full_seqs = torch.from_numpy(full_seqs) if full_seqs is not None else None

        return seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs, utt_ids