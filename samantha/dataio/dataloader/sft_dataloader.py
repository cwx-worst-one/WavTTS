"""Supervised Fine-tuning datamodule for GPT"""
import logging
import os
import tempfile
from typing import List, Union

import torch
from cruise import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities import DIST_ENV
from cruise.utilities.hdfs_io import hcopy, hglob
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer

from mariana.data.gpt.tokenization import CasterTokenizer


class RawTextProcessor:
    r"""
    Args:
        tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
        text_keys: keys that contains text as values in the input.
        max_seq_len: max length that the model accept, if data is not enough,
                        pad_token_id will be used.
        drop_last: if text length is not divisible by max_seq_len, set this
                    field to False will pad the remainder.
    """

    def __init__(
        self,
        tokenizer: str,
        text_keys: Union[str, List[str]],
        max_seq_len: int,
        drop_last: bool = False,
        tokenizer_kwargs=None,
        **kwargs,
    ):
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        self.drop_last = drop_last
        self.tokenizer_kwargs = tokenizer_kwargs
        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs

    def truncate_pair(self, tokenized_prompt, tokenized_response, max_seq_len):
        max_prompt_len = tokenized_prompt["input_ids"]
        max_response_len = tokenized_response["input_ids"]

        while max_prompt_len + max_response_len > max_seq_len:
            if max_prompt_len >= max_response_len:
                max_prompt_len -= 1
            else:
                max_response_len -= 1

        for key in list(tokenized_prompt.keys()):
            tokenized_prompt[key] = tokenized_prompt[key][:max_prompt_len]

        for key in list(tokenized_response.keys()):
            tokenized_response[key] = tokenized_prompt[key][:max_response_len]
        return tokenized_prompt, tokenized_response

    def transform_single_pair(self, pair, max_seq_len):
        prompt_key, response_key = self.text_keys
        prompt = pair[prompt_key]
        response = pair[response_key]
        if isinstance(self.tokenizer, CasterTokenizer):
            prompt = prompt.replace("\n", "●").replace("  ", "●")
            response = response.replace("\n", "●").replace("  ", "●")

        tokenized_prompt = self.tokenizer(prompt, **self.tokenizer_kwargs)
        tokenized_response = self.tokenizer(response, **self.tokenizer_kwargs)

        num_sep_token = 0
        num_eos_token = 0
        if (
            self.kwargs.get("add_sep_token", False)
            and len(tokenized_prompt["input_ids"]) > 0
        ):
            num_sep_token += 1

        if self.kwargs.get("add_eos_token"):
            num_eos_token += 1

        max_seq_len -= num_sep_token + num_eos_token

        if num_sep_token > 0:
            tokenized_prompt["input_ids"].append(self.tokenizer.sep_token_id)
            tokenized_prompt["attention_mask"].append(1)

        if num_eos_token > 0:
            tokenized_response["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_response["attention_mask"].append(1)

        return {"prompt": tokenized_prompt, "response": tokenized_response}

    def transform_dialogue(self, session):
        history_input_ids = []
        history_attention_mask = []
        loss_weight = []
        prompt_loss_weight = self.kwargs.get("prompt_loss_weight", 0.0)
        prev_response_loss_weight = self.kwargs.get("prev_response_loss_weight", 1.0)
        for pair in session[:-1]:
            tokenized_pair = self.transform_single_pair(
                pair, self.kwargs.get("max_seq_len_per_turn", self.max_seq_len)
            )
            history_input_ids.extend(
                tokenized_pair["prompt"]["input_ids"]
                + tokenized_pair["response"]["input_ids"]
            )
            history_attention_mask.extend(
                tokenized_pair["prompt"]["attention_mask"]
                + tokenized_pair["response"]["attention_mask"]
            )
            # mask prompt loss
            loss_weight += [prompt_loss_weight] * len(
                tokenized_pair["prompt"]["input_ids"]
            )
            # add history response loss
            loss_weight += [prev_response_loss_weight] * len(
                tokenized_pair["response"]["input_ids"]
            )

        last_turn = self.transform_single_pair(session[-1], self.max_seq_len)

        loss_weight += [prompt_loss_weight] * len(last_turn["prompt"]["input_ids"]) + [
            1.0
        ] * len(last_turn["response"]["input_ids"])

        input_ids = (
            history_input_ids
            + last_turn["prompt"]["input_ids"]
            + last_turn["response"]["input_ids"]
        )
        attention_mask = (
            history_attention_mask
            + last_turn["prompt"]["attention_mask"]
            + last_turn["response"]["attention_mask"]
        )

        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) == len(loss_weight)

        while len(input_ids) < self.max_seq_len:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            loss_weight.append(0.0)

        input_ids = input_ids[-self.max_seq_len :]
        attention_mask = attention_mask[-self.max_seq_len :]
        loss_weight = loss_weight[-self.max_seq_len :]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_weight,
        }

    def transform(self, session):
        session = session["session"]  # parquet
        if self.kwargs.get("split_session", False):
            dialogues = self.split_pairs(session)
        else:
            dialogues = [session]
        final_instances = []
        for dialogue in dialogues:
            instance = self.transform_dialogue(dialogue)
            final_instances.append({k: torch.as_tensor(v) for k, v in instance.items()})

        return final_instances

    def split_pairs(self, session, max_windows_size):
        pairs = []
        for i in range(len(session)):
            pairs.append(session[0 : i + 1])
        return pairs

    def batch_transform(self, batch_data):
        return default_collate(batch_data)


class SFTDataModule(CruiseDataModule):
    """Supervised Fine-Tuning dataset module.

    It supports reading from raw text dataset and process using pretrained tokenizers.
    """

    def __init__(
        self,
        train_path: str = "hdfs://haruna/home/byte_search_nlp_cr/data/alice/full_20230223_clean_baike_shuf.json",  # noqa
        val_path: str = "",
        val_lm_path: str = "",
        train_size: int = -1,
        train_batch_size: int = 4,
        train_num_workers: int = 1,
        val_batch_size: int = 4,
        val_num_workers: int = 1,
        max_seq_len: int = 1024,
        text_keys: List[str] = ["prompt", "response"],
        source_types: List[str] = ["parquet"],
        tokenizer: str = "hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase",  # noqa
        tokenizer_type: str = "caster",
        gpu_prefetch: bool = False,
        add_sep_token: bool = True,
        add_eos_token: bool = True,
        prompt_loss_weight: float = 1.0,
        split_session: bool = False,
    ):
        super().__init__()
        self.save_hparams()
        self.tokenizer = None

    def local_rank_zero_prepare(self) -> None:
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(
                tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer)
            )
            hcopy(self.hparams.tokenizer, tmp_dir)
        else:
            logging.info(
                f"Prefetching HF tokenizers {self.hparams.tokenizer} on local rank zero..."  # noqa
            )
            AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self):
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(
                tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer)
            )
            if self.hparams.tokenizer_type == "caster":
                self.tokenizer = CasterTokenizer.from_pretrained(tmp_dir, max_len=-1)
            elif self.hparams.tokenizer_type == "bbpe":
                self.tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            else:
                raise NotImplementedError
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def train_dataloader(self):
        train_steps = -1
        if self.hparams.train_size > 0:
            train_steps = self.hparams.train_size // (
                self.hparams.train_batch_size * DIST_ENV.world_size
            )
            assert (
                train_steps > 0
            ), f"train_size={self.hparams.train_size} may be too small to split to batch_size * world_size"  # noqa
        train_files = hglob(self.hparams.train_path)
        self.rank_zero_info(f"Fetched {len(train_files)} training files.")
        if self.hparams.tokenizer_type == "bbpe":
            tokenizer_kwargs = {"return_token_type_ids": False}
        else:
            tokenizer_kwargs = {}

        loader = DistributedCruiseDataLoader(
            data_sources=[train_files],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.train_num_workers,
            predefined_steps=train_steps,
            source_types=self.hparams.source_types,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=RawTextProcessor(
                tokenizer=self.tokenizer
                if self.tokenizer is not None
                else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False,
                add_sep_token=self.hparams.add_sep_token,
                add_eos_token=self.hparams.add_eos_token,
                prompt_loss_weight=self.hparams.prompt_loss_weight,
                max_seq_len_per_turn=self.hparams.max_seq_len,
                split_session=self.hparams.split_session,
                tokenizer_kwargs=tokenizer_kwargs,
            ),
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

    def val_dataloader(self):
        if not self.hparams.val_path:
            return iter([])
        val_steps = -1
        val_files = hglob(self.hparams.val_path)
        self.rank_zero_info(f"Fetched {len(val_files)} val files.")
        if self.hparams.tokenizer_type == "bbpe":
            tokenizer_kwargs = {"return_token_type_ids": False}
        else:
            tokenizer_kwargs = {}
        loader = DistributedCruiseDataLoader(
            data_sources=[val_files],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.val_num_workers,
            predefined_steps=val_steps,
            source_types=self.hparams.source_types,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=RawTextProcessor(
                tokenizer=self.tokenizer
                if self.tokenizer is not None
                else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False,
                add_sep_token=self.hparams.add_sep_token,
                add_eos_token=self.hparams.add_eos_token,
                prompt_loss_weight=self.hparams.prompt_loss_weight,
                max_seq_len_per_turn=self.hparams.max_seq_len,
                split_session=self.hparams.split_session,
                tokenizer_kwargs=tokenizer_kwargs,
            ),
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)

        if not self.hparams.val_lm_path:
            return [loader]
        val_lm_steps = -1
        val_lm_files = hglob(self.hparams.val_lm_path)
        self.rank_zero_info(f"Fetched {len(val_lm_files)} val_lm files.")
        lm_loader = DistributedCruiseDataLoader(
            data_sources=[val_lm_files],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.val_num_workers,
            predefined_steps=val_lm_steps,
            source_types=self.hparams.source_types,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=RawTextProcessor(
                tokenizer=self.tokenizer
                if self.tokenizer is not None
                else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False,
                add_sep_token=self.hparams.add_sep_token,
                add_eos_token=self.hparams.add_eos_token,
                prompt_loss_weight=self.hparams.prompt_loss_weight,
                max_seq_len_per_turn=self.hparams.max_seq_len,
                split_session=self.hparams.split_session,
                tokenizer_kwargs=tokenizer_kwargs,
            ),
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            lm_loader = GPUPrefetcher(lm_loader)

        return [loader, lm_loader]
