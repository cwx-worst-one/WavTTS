import logging
import multiprocessing
import os
import time

import numpy as np
import torch
from cruise import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher

from recipes.valle.datasets.dataset import PhoneTokenizerWithAudioTokens
from recipes.valle.datasets.sami_tacolabel import enc_taco_label_no_bytes
from recipes.valle.datasets.wds_dataload import HiddenPrints
from recipes.valle.utils.remote_io import load_json
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.utils import file_pattern, sort_data_sources
from samantha.utils.hparams import DotDict

multiprocessing.set_start_method(method="fork", force=True)

logger = logging.getLogger(__name__)


class ValleProcessor:
    def __init__(
        self,
        batcher,
        metaid2textid="recipes/valle/datasets/dict/metaid_to_textid.json",
        inference=False,
        enable_buffer_length=True,
        tokenizer_pad=0.0,
        block_sparse=False,
    ):
        self.batcher = batcher
        self.hp = DotDict(
            {
                "phone_tokens_num": 8000,
                "audio_tokens_num": 1024,
                "quality_check": False,
                "return_full_seq": True,
            }
        )
        self.text_converter_dict = load_json(metaid2textid)
        self.tokenizer = None
        self.inference = inference
        self.enable_buffer_length = enable_buffer_length
        self.block_sparse = block_sparse
        self.pad = tokenizer_pad

    def convert_tacolab_to_text_id(self, tacolab):
        """ """
        with HiddenPrints():
            metas = enc_taco_label_no_bytes(
                None, tacolab, {"use_prsdword": False, "forced_refix": True}
            )
        text_id = (
            metas[0].astype(np.int64) * 1_000_000_000
            + metas[1].astype(np.int64) * 1_000_000
            + metas[2].astype(np.int64) * 1_000
            + metas[3].astype(np.int64)
        )
        # TODO: remove hard-coded oov number
        text_id = np.asarray(
            [
                self.text_converter_dict.get(
                    str(x), self.text_converter_dict.get("oov")
                )
                for x in text_id
            ]
        ).astype(np.int64)
        return text_id

    def quality_check(self, sent):
        try:
            rms_max = sent["rms_stats"]["rms_max"]
            snr = sent["snr"]
            speaker_similarity_min = sent["speaker_similarity"]["min"]
            mos = sent["mos"]
            if (
                rms_max >= -13
                and snr >= 7
                and speaker_similarity_min >= 0.6
                and mos >= 4.2
            ):
                return True
            else:
                return False
        except Exception as e:
            return False

    def transform(self, sample):
        utt_id = sample["__key__"]

        if self.hp.quality_check and not self.quality_check(sample):
            return None
        labels = sample.get("labels").decode()
        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split("\n")))
        # text_id, [text_len]
        text_id = self.convert_tacolab_to_text_id(labels)
        # wav_id, [1, wav_len, dim=6]
        wav_id = np.frombuffer(sample.get("wav_id"), np.int64).reshape([-1, 6])
        if wav_id is None:
            return None
        if self.tokenizer == None:
            self.tokenizer = PhoneTokenizerWithAudioTokens(
                self.hp.phone_tokens_num, self.hp.audio_tokens_num
            )
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
                full_seq = np.concatenate([bos, full_text_seq, sep, wav_id], axis=0)
            else:
                full_seq = np.concatenate(
                    [bos, full_text_seq, sep, wav_id, eos], axis=0
                )
        else:
            full_seq = None

        item = dict(
            utt_id=utt_id,
            seq=seq,
            pos_id=pos_id,
            seq_sen_id=seq_sen_id,
            full_seq=full_seq,
        )

        return item

    def batch_transform(self, batch_datas, collate_last=False):
        for item in batch_datas:
            batch_data = self.batcher.collate_batch(item)
            if batch_data is None:
                continue
            results = batch_data

            # length padding
            seqs = []
            seq_lens = []
            seq_sen_ids = []
            pos_ids = []
            full_seqs = []
            utt_ids = []
            max_seq_len = max(x["seq"].shape[0] for x in results)

            if self.block_sparse:
                max_seq_len = (max_seq_len // 32 + 1) * 32
            for item in results:
                utt_id = item["utt_id"]
                seq = item["seq"]
                pos_id = item["pos_id"]
                seq_sen_id = item["seq_sen_id"]
                full_seq = item["full_seq"]
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

            yield (utt_ids, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs)


class ValleDataModule(CruiseDataModule):
    """Valle dataset"""

    def __init__(
        self,
        train_num_workers=4,
        valid_num_workers=1,
        batch_means_tokens=True,
        drop_last=False,
        shuffle=True,
        split_path_list_by_rank=True,
        max_batch_size=20000,
        pin_memory=False,
        cuda_cache_size=2,
        prefetch_retry=3,
        meta_path="",
        train_file_list=None,
        valid_file_list=None,
        train_item_transform=None,
        valid_item_transform=None,
        batch_transform=None,
        train_batch_size=64,
        valid_batch_size=64,
        train_num_readers=32,
        valid_num_readers=32,
        gpu_prefetch=None,
        batcher_config=None,
        **_kwargs,
    ):
        super().__init__()
        self.save_hparams()

    def setup(self, stage=None):
        """setup"""
        self.train_path = []
        for cfg in self.hparams.train_file_list:
            data_root, file_prefix, file_num = (
                cfg["data_root"],
                cfg["file_prefix"],
                cfg.get("file_num", None),
            )
            self.train_path += file_pattern(data_root, file_prefix, file_num)

    def train_dataloader(self):
        data_sources, source_types = sort_data_sources(self.train_path)
        batcher = BucketBatcher(**self.hparams.batcher_config)
        falcon_config = self.hparams
        loader = DistributedCruiseDataLoader(
            data_sources=data_sources,
            source_types=source_types,
            num_workers=self.hparams.train_num_workers,
            shuffle=True,
            processor=ValleProcessor(batcher=batcher),
            batch_sizes=[self.hparams.train_batch_size] * len(data_sources),
            num_readers=[self.hparams.train_num_readers] * len(data_sources),
            # predefined_steps=train_steps,
            drop_last=True,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            decode_fn_list=None,
            transform_output_many=False,
            falcon_config=falcon_config,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader


if __name__ == "__main__":
    from hyperpyyaml import load_hyperpyyaml

    max_steps = 20000
    data_yaml_path = "tests/benchmarks/configs/cruise_valle.yaml"
    with open(data_yaml_path) as f:
        kwargs = load_hyperpyyaml(f)

    data_module = ValleDataModule(**kwargs)
    data_module.setup()
    dataloader = data_module.train_dataloader()

    worker_id = int(os.getenv("DMLC_WORKER_ID", "0"))
    worker_num = int(os.getenv("ARNOLD_WORKER_NUM", "1"))
    gpu_num = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    rank = int(
        os.getenv(
            "RANK", int(os.getenv("OMPI_COMM_WORLD_RANK", "0")) + worker_id * gpu_num
        )
    )
    world_size = int(os.getenv("WORLD_SIZE", worker_num * gpu_num))

    last_time = time.time()
    cur_time = 0
    total_cnt = 0
    first_time = 0
    total_time = 0
    total_token_size = 0
    for idx, batch in enumerate(dataloader):
        cur_time = time.time() - last_time
        utt_ids, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs = batch
        if idx == 0:
            first_time = cur_time
        else:
            total_cnt += 1
            total_time += cur_time
            total_token_size += seqs.shape[0] * seqs.shape[1]
        logger.info(
            f"rank: {rank}, world_size: {world_size}, idx: {idx}, time: {cur_time}, seq size: {seqs.shape}"
        )

        if idx > max_steps:
            break
        last_time = time.time()
    logger.info(
        f"rank: {rank}, world_size: {world_size}, read first batch use time {first_time}s"
    )
    logger.info(
        f"rank: {rank}, world_size: {world_size}, read {total_token_size} tokens, speed is {total_token_size / total_time} tokens/s"
    )
    logger.info(
        f"rank: {rank}, world_size: {world_size}, read {total_cnt} batches use time {total_time}s, speed is {total_time/total_cnt}sec per batch"
    )
