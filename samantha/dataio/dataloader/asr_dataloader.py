"""Supervised Fine-tuning datamodule for GPT"""
import logging
import os
import re
import tempfile
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from cruise import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities import DIST_ENV
from cruise.utilities.hdfs_io import hcopy, hglob
from transformers import AutoTokenizer

from mariana.data.gpt.tokenization import CasterTokenizer


class UtteranceTextProcessor:
    r"""
    单条句子作为样本。
    以后尝试将所有的样本拼成一个sequence再切割。
    Args:
        tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
        templates:  a template file to format input text
        max_seq_len: max length that the model accept, if data is not enough,
                    pad_token_id will be used.
        drop_last: if text length is not divisible by max_seq_len, set this
                    field to False will pad the remainder.
    """

    def __init__(
        self,
        tokenizer: str,
        template_fn: str,
        max_seq_len: int,
        drop_last: bool = False,
        tokenizer_kwargs=None,
        sep_tokens=[],
        **kwargs,
    ):
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.parse_template(template_fn)
        self.max_seq_len = max_seq_len
        self.drop_last = drop_last
        self.tokenizer_kwargs = tokenizer_kwargs
        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs
        self.sep_tokens = sep_tokens

        self.space_patten = re.compile(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])")

    def add_tokens(self, tokens):
        return self.tokenizer.add_tokens(tokens, special_tokens=True)

    def parse_template(self, fn):
        templates = []
        with open(fn, "r") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                prob, template = line.strip().split("\t")
                prob = float(prob)
            templates.append((prob, template))
        self.templates = templates

    def _get_prefix_and_affix_masks(self, tokens, token_masks):
        masks = torch.zeros_like(tokens, device=tokens.device).long()
        for sep_token in self.sep_tokens:
            masks |= (tokens == self.tokenizer.vocab[sep_token]).long()
        first_eos_postion_idxs = masks.argmax(dim=1).long()
        not_exceed = ((first_eos_postion_idxs + 1) < masks.shape[1]).long()
        affix_mask_start_position_idxs = (
            first_eos_postion_idxs + 1
        ) * not_exceed + first_eos_postion_idxs * (1 - not_exceed)
        affix_mask_start_positions = F.one_hot(
            affix_mask_start_position_idxs.long(), masks.shape[1]
        ).long()
        affix_masks = torch.cumsum(affix_mask_start_positions, dim=1) * token_masks
        prefix_masks = (1 - affix_masks) * token_masks
        return prefix_masks, affix_masks

    def _code_list_to_str(self, codelist):
        """
        convert codelist to token:
        [1,2,3] -> <1><2><3>
        """
        return "".join([f"<{i}>" for i in codelist])

    def _labellist_to_str(self, label_list):
        """
        convert label_list to text
        ['哈', '哈', '哈'] -> 哈哈哈
        """
        text = " ".join(label_list)
        # keep space between English words
        text = self.space_patten.sub(r"\1\2", text)
        text = self.space_patten.sub(r"\1\2", text)
        return text

    def pre_transform(self, data_dict):
        if "label" in data_dict:
            if isinstance(data_dict["label"], np.ndarray):
                data_dict["label"] = data_dict["label"].tolist()
            if isinstance(data_dict["label"], list):
                data_dict["label"] = self._labellist_to_str(data_dict["label"])
        if "bestrq" in data_dict:
            if isinstance(data_dict["bestrq"], np.ndarray):
                data_dict["bestrq"] = data_dict["bestrq"].tolist()
            if isinstance(data_dict["bestrq"], list) and isinstance(
                data_dict["bestrq"][0], int
            ):
                data_dict["bestrq"] = self._code_list_to_str(data_dict["bestrq"])
        return data_dict

    def transform(self, data_dict):
        # sample a template
        distribution = [t[0] for t in self.templates]
        index = np.random.multinomial(1, distribution).argmax()

        data_dict = self.pre_transform(data_dict)

        text = self.templates[index][1]
        selected_template = text
        template_keys = list(set(re.findall("\{\S+?\}", text)))  # noqa
        for key in template_keys:
            item_key = key.lstrip("{").rstrip("}")
            data = data_dict[item_key]
            text = re.sub(key, data, text)
        data_dict["used_text"] = text
        data_dict["used_template"] = selected_template

        return data_dict

    def batch_transform(self, batch_data):
        collated_batch_data = {}
        keys = batch_data[0].keys()
        for key in keys:
            collated_batch_data[key] = []
            for item in batch_data:
                collated_batch_data[key].append(item[key])
        tokenized = self.tokenizer.batch_encode_plus(
            collated_batch_data["used_text"], padding=True, return_tensors="pt"
        )
        collated_batch_data["input_ids"] = tokenized["input_ids"]
        collated_batch_data["attention_mask"] = tokenized["attention_mask"]
        prefix_masks, affix_masks = self._get_prefix_and_affix_masks(
            collated_batch_data["input_ids"], collated_batch_data["attention_mask"]
        )
        collated_batch_data["prefix_masks"] = prefix_masks
        collated_batch_data["affix_masks"] = affix_masks
        return collated_batch_data


class AsrDataModule(CruiseDataModule):
    """ASR dataset"""

    def __init__(
        self,
        train_path: str = "",
        val_path: str = "",
        train_size: int = -1,
        train_batch_size: int = 1,
        train_num_workers: int = 1,
        val_batch_size: int = 1,
        val_num_workers: int = 1,
        max_seq_len: int = 1024,
        tokenizer: str = "",
        tokenizer_type: str = "bbpe",
        tokens_to_add: List[str] = ["[SPEECH]", "[/SPEECH]"]
        + ["<{}>".format(i) for i in range(1024)],
        sep_tokens: List[str] = ["[/SPEECH]"],
        template_fn: str = "",
        val_template_fn: str = "",
        source_types: List[str] = ["parquet"],
        gpu_prefetch: bool = False,
        prompt_loss_weight: float = 1.0,
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
        logging.info(
            "Adding {} tokens to tokenizer: {}...".format(
                len(self.hparams.tokens_to_add),
                " ".join(self.hparams.tokens_to_add[:3]),
            )
        )
        self.tokenizer.add_tokens(self.hparams.tokens_to_add, special_tokens=True)

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
            processor=UtteranceTextProcessor(
                tokenizer=self.tokenizer
                if self.tokenizer is not None
                else self.hparams.tokenizer,
                template_fn=self.hparams.template_fn,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False,
                sep_tokens=self.hparams.sep_tokens,
                tokenizer_kwargs=tokenizer_kwargs,
            ),
            transform_output_many=False,
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
            processor=UtteranceTextProcessor(
                tokenizer=self.tokenizer
                if self.tokenizer is not None
                else self.hparams.tokenizer,
                template_fn=self.hparams.val_template_fn,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False,
                sep_tokens=self.hparams.sep_tokens,
                tokenizer_kwargs=tokenizer_kwargs,
            ),
            transform_output_many=False,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return [loader]
