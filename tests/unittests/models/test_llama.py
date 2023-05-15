# Copyright (c) ByteDance, Lightning AI. All rights reserved.
#
# Parts of this source code is licensed under the Apache-2.0 license as found in
# https://github.com/Lightning-AI/lit-llama/blob/main/LICENSE

import os
from urllib.request import urlretrieve

import numpy as np
import pytest
import torch

from samantha.models import Llama, LlamaConfig, LlamaModel
from samantha.utils.checkpoints_utils.convert_llama import (
    calc_rotary_inv_freq,
    convert_llama_state_dict,
)
from tests.helpers.runif import RunIf
from tests.helpers.testing_utils import torch_device
from tests.unittests.models.utils import ids_tensor

files = {
    "original_model.py": "https://gist.githubusercontent.com/lantiga/fd36849fb1c498da949a0af635318a7b/raw/7dd20f51c2a1ff2886387f0e25c1750a485a08e1/llama_model.py",  # noqa
    "original_adapter.py": "https://gist.githubusercontent.com/awaelchli/546f33fcdb84cc9f1b661ca1ca18418d/raw/e81d8f35fb1fec53af1099349b0c455fc8c9fb01/original_adapter.py",  # noqa
}


def download_original_llama_model() -> None:
    for filepath, url in files.items():
        if not os.path.isfile(filepath):
            print(f"Downloading original implementation to {filepath!r}")
            urlretrieve(url=url, filename=filepath)
            print("Done")
        else:
            print("Original implementation found. Skipping download.")


@pytest.fixture()
def orig_llama():
    download_original_llama_model()
    import original_model

    return original_model


class LlamaModelTester:
    def __init__(
        self,
        batch_size=14,
        seq_len: int = 7,
        vocab_size: int = 99,
        n_head: int = 2,
        n_embd: int = 32,
        n_layer: int = 2,
        initializer_range: float = 0.02,
        rms_norm_epsilon: float = 1e-5,
        use_rotary_embeddings: bool = True,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.initializer_range = initializer_range
        self.rms_norm_epsilon = rms_norm_epsilon
        self.use_rotary_embeddings = use_rotary_embeddings

    def get_inputs(self):
        return ids_tensor(
            (self.batch_size, self.seq_len), self.vocab_size, device=torch_device
        )

    def get_config(self):
        return LlamaConfig(
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            logit_num=self.vocab_size,
            initializer_range=self.initializer_range,
            rms_norm_epsilon=self.rms_norm_epsilon,
            use_rotary_embeddings=self.use_rotary_embeddings,
            attention_kwargs={"enable_flash": False, "enable_mem_efficient": False},
        )

    def create_and_test_model(self):
        model = LlamaModel(self.get_config())
        model = model.to(torch_device)
        model.apply(model._init_weights)
        model.eval()
        return model

    def create_and_test_model_with_lm_head(self):
        model = Llama(self.get_config())
        model = model.to(torch_device)
        model.apply(model._init_weights)
        model.eval()
        return model


@pytest.fixture()
def model_tester() -> LlamaModelTester:
    yield LlamaModelTester()


def test_llama_model(model_tester: LlamaModelTester) -> None:
    inputs = model_tester.get_inputs()
    model = model_tester.create_and_test_model()
    out = model(inputs)
    assert out.shape == (
        model_tester.batch_size,
        model_tester.seq_len,
        model_tester.n_embd,
    )


@RunIf(min_torch="2.0")
def test_llama_model_compile(model_tester: LlamaModelTester) -> None:
    inputs = model_tester.get_inputs()
    model_tester.use_rotary_embeddings = False

    model = model_tester.create_and_test_model()
    model = torch.compile(model)
    out = model(inputs)
    assert out.shape == (
        model_tester.batch_size,
        model_tester.seq_len,
        model_tester.n_embd,
    )


def load_equivalent_llama(mt, model, o_llama):
    orig_llama_config = o_llama.ModelArgs(
        dim=mt.n_embd,
        n_layers=mt.n_layer,
        n_heads=mt.n_head,
        vocab_size=mt.vocab_size,
        norm_eps=mt.rms_norm_epsilon,
        max_seq_len=mt.seq_len,
    )
    orig_llama_model = o_llama.Transformer(orig_llama_config).to(torch_device).eval()
    inv_freq = calc_rotary_inv_freq(mt.n_embd, mt.n_head)

    converted_state_dict = convert_llama_state_dict(
        orig_llama_model.state_dict(), inv_freq
    )
    model.load_state_dict(converted_state_dict)
    return model, orig_llama_model


@torch.no_grad()
def test_to_orig_llama(model_tester: LlamaModelTester, orig_llama) -> None:
    model_tester.seq_len = 64
    model_tester.vocab_size = 32000
    model_tester.n_layer = 16
    model_tester.n_head = 16
    model_tester.n_embd = 32

    model = model_tester.create_and_test_model_with_lm_head()

    model, orig_llama_model = load_equivalent_llama(model_tester, model, orig_llama)

    batch_size = 3

    token_sample = torch.randint(
        0,
        model_tester.vocab_size,
        size=(batch_size, model_tester.seq_len),
        dtype=torch.int64,
        device=torch_device,
    )

    orig_llama_embed = orig_llama_model.tok_embeddings(token_sample)
    llama_embed = model.transformer.wte(token_sample)
    assert torch.allclose(orig_llama_embed, llama_embed)

    seq_len = token_sample.shape[1]
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=torch_device)
    mask = torch.triu(mask, diagonal=1)

    orig_llama_model.freqs_cis = orig_llama_model.freqs_cis.to(torch_device)
    orig_llama_block_out = orig_llama_model.layers[0](
        orig_llama_embed, 0, orig_llama_model.freqs_cis[:seq_len], mask
    )
    llama_block_out = model.transformer.h[0](llama_embed)
    np.testing.assert_allclose(
        orig_llama_block_out.cpu(), llama_block_out.cpu(), atol=1e-6
    )

    expected = orig_llama_model(token_sample, 0)
    out = model(token_sample)
    np.testing.assert_allclose(out.cpu(), expected.cpu(), atol=1e-5)


@torch.no_grad()
def test_llama_kv_cache(model_tester: LlamaModelTester, orig_llama) -> None:
    n_samples = 20
    model_tester.seq_len = 64
    model_tester.batch_size = 2
    model_tester.vocab_size = 12000
    model_tester.n_embd = 32
    model_tester.n_layer = 2
    model_tester.n_head = 2

    model = model_tester.create_and_test_model_with_lm_head()

    model, orig_llama_model = load_equivalent_llama(model_tester, model, orig_llama)

    # 1. no k/v cache
    sampled_ids = ids_tensor(
        (model_tester.batch_size, 1), model_tester.vocab_size, device=torch_device
    )
    for _ in range(n_samples):
        logits = model(sampled_ids)
        sampled = logits[:, -1].softmax(dim=-1).argmax(dim=-1, keepdim=True)
        sampled_ids = torch.cat((sampled_ids, sampled), dim=1)
    assert sampled_ids.shape == (model_tester.batch_size, n_samples + 1)

    def sample_with_kv_cache(samples, start_idx, kv_cache):
        for idx in range(start_idx, n_samples):
            logits = model(samples[:, idx : idx + 1], kv_cache=kv_cache)
            sampled = logits[:, -1].softmax(dim=-1).argmax(dim=-1, keepdim=True)
            samples = torch.cat((samples, sampled), dim=1)
        return samples

    # 2. k/v cache without prefix
    kv_cache = {}
    sampled_ids_cache = sampled_ids[:, :1]
    sampled_ids_cache = sample_with_kv_cache(sampled_ids_cache, start_idx=0, kv_cache=kv_cache)
    torch.testing.assert_close(sampled_ids, sampled_ids_cache)

    # 3. k/v cache with prefix
    prefix_len = 10
    sampled_ids_cache_prefix = sampled_ids[:, : prefix_len + 1].clone()

    kv_cache2 = {}
    model(sampled_ids_cache_prefix[:, :prefix_len], kv_cache=kv_cache2)

    for idx in range(prefix_len, n_samples):
        logits = model(sampled_ids_cache_prefix[:, idx : idx + 1], kv_cache=kv_cache2)
        sampled = logits[:, -1].softmax(dim=-1).argmax(dim=-1, keepdim=True)
        sampled_ids_cache_prefix = torch.cat((sampled_ids_cache_prefix, sampled), dim=1)
    torch.testing.assert_close(
        sampled_ids, sampled_ids_cache_prefix
    )
