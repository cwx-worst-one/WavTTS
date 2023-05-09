# Copyright (c) 2023, ByteDance, 2021, EleutherAI, Meta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CREDITS: This implementation is partially taken from GPT-NeoX and adjusted
# to be compatible with k/v caching
# https://github.com/EleutherAI/gpt-neox

from typing import Optional, Tuple

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# @torch.jit.script # NOTE: This is not compatible with the FDSP strategy
def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose,
    as it is transformative (it does not create the embedding dimension) and
    will likely be picked up (imported) on a ad-hoc basis.
    """

    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(
                x.shape[seq_dimension], device=x.device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, q_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )
        if q.shape[2] != k.shape[2] and q_len is not None:
            return (
                apply_rotary_pos_emb(
                    q,
                    self._cos_cached[..., q_len - 1 :, :],
                    self._sin_cached[..., q_len - 1 :, :],
                ),
                apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
            )
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class SeerEmbedding(RotaryEmbedding):
    def __init__(
        self, dim_model: int, n_priors: int, n_seers: int, seq_len: int, *_, **__
    ):
        super().__init__(dim_model, *_, **__)
        self.n_priors = n_priors
        self.n_seers = n_seers
        self.seq_len = seq_len

        t = torch.arange(self.n_priors + self.seq_len, dtype=torch.float32)
        t[self.n_priors :] = (
            t[self.n_priors :]
            .reshape(self.n_seers, -1)
            .transpose(1, 0)
            .reshape(self.seq_len)
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()[None, None, :, :]
        self._sin_cached = emb.sin()[None, None, :, :]

    def _update_cos_sin_tables(self, x):
        self._cos_cached = self._cos_cached.to(dtype=x.dtype)
        self._sin_cached = self._sin_cached.to(dtype=x.dtype)
        self._cos_cached = self._cos_cached.to(device=x.device)
        self._sin_cached = self._sin_cached.to(device=x.device)

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, q_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k)
        if q.shape[2] != k.shape[2] and q_len is not None:
            return (
                apply_rotary_pos_emb(
                    q,
                    self._cos_cached[..., q_len - self.n_seers :, :],
                    self._sin_cached[..., q_len - self.n_seers :, :],
                ),
                apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
            )
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
