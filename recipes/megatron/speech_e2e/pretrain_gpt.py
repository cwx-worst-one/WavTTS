# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

from functools import partial

import torch
from megatron import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.multi_indexed_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import (
    average_losses_across_data_parallel_group,
    get_packed_masks_and_position_ids,
)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ["token_ids", "field_ids", "sample_ids"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    token_ids = data_b["token_ids"].long()
    field_ids = data_b["field_ids"].long()
    sample_ids = data_b["sample_ids"].long()

    labels = token_ids[:, 1:].contiguous()
    tokens = token_ids[:, :-1].contiguous()
    field_ids_for_tokens = field_ids[:, :-1].contiguous()
    field_ids_for_labels = field_ids[:, 1:].contiguous()
    sample_ids_for_tokens = sample_ids[:, :-1].contiguous()

    # Get the attention masks and postition ids.
    attention_mask, position_ids = get_packed_masks_and_position_ids(
        tokens, sample_ids_for_tokens, field_ids_for_tokens == 0
    )

    # Get loss masks
    loss_mask = field_ids_for_labels == 1

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0("> building train, validation, and test datasets " "for GPT ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path[0],
        data_impl=args.data_impl,
        splits_string=args.split,
        fields=["inputs", "targets"],
        skip_warmup=(not args.mmap_warmup),
        max_seq_length=args.seq_length + 1,
        sep_token=tokenizer.sep,
        pad_token=tokenizer.pad,
        train_valid_test_num_samples=train_val_test_num_samples,
        seed=args.seed,
        pack=True,
    )
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
    )
