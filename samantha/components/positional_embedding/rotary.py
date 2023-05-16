from typing import Optional, Tuple

import torch


def apply_rotary_pos_emb(
    x: torch.Tensor, rope_cache: torch.Tensor, use_complex: bool = False
) -> torch.Tensor:
    """Applies the rotary position embedding `rope_cache` on x.
    Here, we assume that `rope_cache` is a tensor of the following shape:

    [1, 1, seq_len, d_k*2]

    Here, d_k*2 represents both the cosine/sine tables that will be applied to x.
    Two methods are currently supported for calculating the resulting positional
    embeddings: a complex and real variant. The first is more precise
    (for longer sequences) and used in training the original LLaMA model.
    The latter is here to support Triton model compilation (complex types are
    not supported yet).

    Args:
        x (torch.Tensor): q/k tensor
        rope_cache (torch.Tensor): ROPE embedding table (cos/sine)
        use_complex (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
    T = x.shape[2]
    rope_cache = rope_cache[:, :, :T]

    # split the [b, seq_len, d_k, cos/sin] into [b, seq_len, d_k, (cos, sin)]
    x_cos_sin_stacked = x.float().reshape(*x.shape[:-1], -1, 2)
    if use_complex:
        rope_cache = torch.view_as_complex(rope_cache)
        x_ = torch.view_as_complex(x_cos_sin_stacked)
        x_ = torch.view_as_real(x_ * rope_cache).flatten(3)
    else:
        x_ = torch.stack(
            [
                x_cos_sin_stacked[..., 0] * rope_cache[..., 0]
                - x_cos_sin_stacked[..., 1] * rope_cache[..., 1],
                x_cos_sin_stacked[..., 1] * rope_cache[..., 0]
                + x_cos_sin_stacked[..., 0] * rope_cache[..., 1],
            ],
            -1,
        ).flatten(3)
    return x_.type_as(x)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer (Su et. al), supported for
    k/v caching. The `max_seq_len` can be used to pre-ininitialize the
    position embedding table, which speeds up inference time.
    """

    def __init__(
        self,
        dim_model: int,
        max_seq_len: Optional[int] = None,
        use_complex: bool = True,
        *_,
        **__,
    ):
        super().__init__()
        self.use_complex = use_complex
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        if max_seq_len is not None:
            self._seq_len_cached = max_seq_len
            self._cache = self.compute_cache(max_seq_len)
        else:
            self._seq_len_cached = None
            self._cache = None

    def compute_cache(self, seq_len: int, dtype=torch.float16):
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq).float()
        self._cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self._cache = self._cache[None, None, :, :]

        # this is to mimic the behaviour of complex32,
        # else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            self._cache = self._cache.float()
        return self._cache

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]
        # Reset the tables if the sequence length has changed,
        if self._seq_len_cached is None or seq_len > self._seq_len_cached:
            self._cache = self.compute_cache(seq_len, dtype=x.dtype)
        return self._cache

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, q_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cache = self._update_cos_sin_tables(k, seq_dimension=-2)
        if q.shape[2] != k.shape[2] and q_len is not None:
            return (
                apply_rotary_pos_emb(
                    q,
                    self._cache[:, :, q_len - 1 : q_len],
                    use_complex=self.use_complex,
                ),
                apply_rotary_pos_emb(k, self._cache, use_complex=self.use_complex),
            )
        return (
            apply_rotary_pos_emb(q, self._cache, use_complex=self.use_complex),
            apply_rotary_pos_emb(k, self._cache, use_complex=self.use_complex),
        )


class SeerEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_priors: int,
        n_seers: int,
        seq_len: int,
        use_complex: bool = True,
        seer_rearrange: bool = True,
        *_,
        **__,
    ):
        super().__init__()
        self.n_priors = n_priors
        self.n_seers = n_seers
        self.seq_len = seq_len
        self.seer_rearrange = seer_rearrange

        self.use_complex = use_complex
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)
        self._cache = self.compute_cache()

    def compute_cache(self, dtype=torch.float16):
        t = torch.arange(
            self.n_priors + self.seq_len,
            dtype=torch.float32,
            device=self.inv_freq.device,
        )
        if self.seer_rearrange:
            t[self.n_priors :] = (
                t[self.n_priors :]
                .reshape(self.n_seers, -1)
                .transpose(1, 0)
                .reshape(self.seq_len)
            )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq).float()
        self._cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self._cache = self._cache[None, None, :, :]

        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            self._cache = self._cache.float()
        return self._cache

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, q_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache.device != q.device:
            self._cache = self._cache.to(q.device)
        if q.shape[2] != k.shape[2] and q_len is not None:
            return (
                apply_rotary_pos_emb(
                    q,
                    self._cache[:, :, q_len - self.n_seers : q_len],
                    use_complex=self.use_complex,
                ),
                apply_rotary_pos_emb(k, self._cache, use_complex=self.use_complex),
            )
        return (
            apply_rotary_pos_emb(q, self._cache, use_complex=self.use_complex),
            apply_rotary_pos_emb(k, self._cache, use_complex=self.use_complex),
        )
