import torch
import numpy as np
import webdataset as wds
import os
import ffmpeg
import time
import json
import sys
from functools import lru_cache
from torchdata.datapipes.iter import IterableWrapper, Mapper
from tqdm import tqdm
from scipy.signal import resample
import random

from torch.utils.data import IterableDataset
from recipes.valle.datasets.sami_tacolabel import enc_taco_label_no_bytes
from recipes.valle.datasets.dataset import PhoneTokenizerWithAudioTokens
from recipes.valle.utils.remote_io import load_json
from recipes.valle.datasets.dataset import ValleCollator
from samantha.utils.hparams import DotDict
from samantha.dataio.webdataset.ra_wds import WebDataset as RAWds
import traceback

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def tacolab_postprocess(meta):

    def is_phone(ph):
        return ph not in set(["sil", "sp", "pau", "<unk>"])
    def is_en(lab):
        return lab.startswith("E")

    try:
        short_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t1"
        mid_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t2"

        labels = meta.get("labels")
        if not labels: return None
        labels = labels.split("\n")

        if not meta.get("words") or len(meta["words"]) < 1:
            mis_align = True
        else:
            words = []
            for word in meta["words"]:
                if word.get("tag") == "<aed>":
                    continue
                words.append(word)

            # check if tacolab can align with words, if not, only insert short-sp.
            label_cnt = 0
            mis_align = False
            for label in labels:
                try:
                    data = label.split("\t")
                    if len(data) != 5:
                        continue
                    phone, tone, wordpost, wordcateg, prosody = data
                    if is_phone(phone) and prosody != "0":
                        label_cnt += 1
                except Exception as e:
                    continue
            if label_cnt != len(words):
                mis_align = True


        idx = 0
        cur_st, pre_et = 0, 0
        flag = False
        new_lab = []
        for label in labels:
            if mis_align or idx == len(words):
                word = None
            else:
                word = words[idx]
                
            data = label.split("\t")
            if len(data) != 5:
                continue

            phone, tone, wordpost, wordcateg, prosody = data
            if flag:
                flag = False
                cur_st = word["start_time"] if word else 0
                # mid sp > 50ms, short sp < 50ms
                if not mis_align and cur_st - pre_et > 0.05:
                    new_lab.append(mid_sp)
                else:
                    new_lab.append(short_sp)
            if is_phone(phone) and prosody != "0":
                idx += 1
                if is_en(phone) and prosody == "1":
                    flag = True
                    pre_et = word["end_time"] if word else 0
                    cur_lab = f"{phone}\t{tone}\t{wordpost}\t{wordcateg}\t0"
                else:
                    cur_lab = label
                new_lab.append(cur_lab)
            else:
                new_lab.append(label)
        # safe check
        if not mis_align and idx != len(words):
            return None

        return new_lab
    except Exception as e:
        # print(e, word, words, phone, new_lab, mis_align)
        traceback.print_exc()
        return None

def ffmpeg_read_audio(audio_bin, sample_rate=24000):
    st = time.time()
    seg_bin, err = ffmpeg.input("pipe:").output("pipe:", loglevel="error", format="s16le", ar=sample_rate).run(input=audio_bin, quiet=True)
    return (np.frombuffer(seg_bin, dtype="int16") / 32768.0).astype(np.float32)

def get_buffer_length(length, enable=True, max_tokens=None):
    '''
    avoid CUDA error
    '''
    if enable and length >= 1500:
        length = int(length / 1000 * length)
        if max_tokens:
            length = min(length, max_tokens)
    return length

def dynamic_bucketizer(dataset, max_token_count=1000, min_len=10, max_len=1492, buffer_size=1000, enable_buffer_length=True):
    dataset = dataset.max_token_bucketize(
        buffer_size=buffer_size,
        max_token_count=max_token_count, 
        len_fn=lambda x: get_buffer_length(x[1].shape[0], enable_buffer_length, max_token_count - 1),
        min_len=min_len,
        max_len=get_buffer_length(max_len, enable_buffer_length, max_token_count - 1),
        include_padding=True) # max_len must equal to max_token_count, or may return empty batch when there is a sample longer than max_token_count
    return dataset

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src

class WDSDataset(IterableWrapper):
    def __init__(self, wds_lst, 
                 metaid2textid, 
                 hp,
                 inference=False,
                 resampled=True):
        # self.meta_dir = meta_dir
        self.hp = DotDict(hp)
        self.inference = inference
        self.text_converter_dict = load_json(metaid2textid)
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            self.hp.phone_tokens_num, self.hp.audio_tokens_num
        )

        urls = self._load_wds_lst(wds_lst)
        self.dataset = (
            RAWds(urls=urls, 
                        resampled=resampled, 
                        shardshuffle=True,
                        nodesplitter=wds.split_by_node)
            .decode() # 根据已知的后缀名或类型进行解码
            .map(self._lab2id)
            .shuffle(5000)
        )

    
    def __getstate__(self):
        pass

    def quality_check(self, sent):
        try:
            rms_max = sent['rms_stats']['rms_max']
            snr = sent['snr']
            speaker_similarity_min = sent['speaker_similarity']['min']
            mos = sent['mos']
            if rms_max >= -13 and snr >= 7 and speaker_similarity_min >= 0.6 and mos >= 4.2:
                return True
            else:
                return False
        except Exception as e:
            return False

    def norm_pad(self, wav):
        wav = wav * 1.0 / max(0.01, np.max(np.abs(wav)))
        sil = np.zeros(2400)
        wav = np.concatenate((sil, wav, sil))
        return wav.astype(np.float32)

    def _load_wds_lst(self, fn):
        urls = []
        with open(fn, "r", encoding="utf-8") as fi:
            for line in fi:
                wds_path = line.strip()
                if not wds_path:
                    continue
                if wds_path.startswith("hdfs://"):
                    url = f"pipe: hdfs dfs -cat {wds_path}"
                else:
                    url = f"pipe: cat {wds_path}"
                urls.append(url)
        return urls

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def _lab2id(self, sample):
        utt_id = sample["__key__"]

        if self.hp.quality_check and not self.quality_check(sample):
            return None
        labels = sample.get("labels").decode()
        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split('\n')))
        # text_id, [text_len]
        text_id = self.convert_tacolab_to_text_id(labels)
        # wav_id, [1, wav_len, dim=6]
        wav_id = np.frombuffer(sample.get("wav_id"), np.int64).reshape([-1, 6])
        if wav_id is None:
            return None

        text_id = self.tokenizer.tokenize(text_id, "inputs")
        wav_id = self.tokenizer.tokenize(wav_id, "targets")

        text_len = text_id.shape[0]
        wav_len, num_rvqs = wav_id.shape

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

        if self.hp.return_full_seq:
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

        return utt_id, seq, pos_id, seq_sen_id, full_seq
        # return seq, pos_id, seq_sen_id, full_seq, utt_id
    
    def convert_tacolab_to_text_id(self, tacolab):
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

class WDSCollator(object):
    def __init__(self, tokenizer_pad, block_sparse=False):
        self.pad = tokenizer_pad
        self.block_sparse = block_sparse

    def __call__(self, batches):
        # using bucket dynmaic batch, so batches shape is:
        # [
        #    [
        #        sample1, sample2, ..., sampleN
        #    ]
        # ]
        results = batches[0]

        # length padding
        seqs = []
        seq_lens = []
        seq_sen_ids = []
        pos_ids = []
        full_seqs = []
        utt_ids = []
        max_seq_len = max(x[1].shape[0] for x in results)
        
        if self.block_sparse:
            max_seq_len = (max_seq_len // 32 + 1) * 32
        for utt_id, seq, pos_id, seq_sen_id, full_seq in results:
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

        return utt_ids, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs
    
if __name__ == "__main__":
    from tqdm import tqdm
    from recipes.soundstream.utils.utils import get_config_from_file
    
    hp = DotDict({
        "phone_tokens_num": 8000,
        "audio_tokens_num": 1024,
        "quality_check": False,
        "return_full_seq": True
    })

    dataset = WDSDataset(
        wds_lst="/mnt/bn/jeffus/data/metas/valle/demo/wds.lst",
        metaid2textid="recipes/valle/datasets/dict/metaid_to_textid.json",
        hp=hp,
        resampled=False
    )

    collate_fn = WDSCollator(tokenizer_pad=0., block_sparse=False)


    dataloader = torch.utils.data.DataLoader(
        dynamic_bucketizer(dataset, 140000, buffer_size=2000), 
        num_workers=32, 
        collate_fn=collate_fn,
        batch_size=1
        )

    # st = time.time()
    for batch in tqdm(dataloader, desc="loop dataset"):
        utt_ids, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs = batch
        print(seqs[0])
        print(seqs.max(), seq_lens.max())
        import pdb;pdb.set_trace()
        if seqs.max() >= 9027 or seqs.shape[1] >= 8000:
            print(seqs.max())
            print(utt_ids)
        # print(len(batch))
        # print(batch[0].shape)
        # for i in batch:
            # print(i[-1])

    # for batch in tqdm(dataset):
        # print(f"{time.time() - st}s")
        # print(batch)
        # exit(0)
        # st = time.time()
