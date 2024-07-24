from typing import Optional, Tuple

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script  # NOTE: This is not compatible with the FDSP strategy
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, dim_model: int, seq_before_head_dim: bool):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, q: torch.Tensor, k: torch.Tensor, seq_dimension: int):
        seq_len = max(q.shape[seq_dimension], k.shape[seq_dimension])

        if q.dtype != k.dtype:
            raise Exception(f"q and k dtypes are not the same: {q.dtype}, {k.dtype}")

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != k.device
            or self._cos_cached.dtype != k.dtype
        ):
            self._seq_len_cached = seq_len

            t = torch.arange(seq_len, device=k.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(k.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(k.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(k.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(k.dtype)

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, q_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype

        q = q.float()
        k = k.float()
        
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(q, k, seq_dimension=self.default_seq_dim)
        if q.shape[2] != k.shape[2] and q_len is not None:
            q = apply_rotary_pos_emb(
                q,
                self._cos_cached[..., q_len - 1 :, :],
                self._sin_cached[..., q_len - 1 :, :],
            )
            k = apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)
        else:
            q = apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
            k = apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        q = q.to(q_dtype)
        k = k.to(k_dtype)
        return q, k

def llama_apply_rotary_pos_emb(
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


class LlamaRotaryEmbedding(torch.nn.Module):
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
        return self._cache.to(x.device)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, q_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cache = self._update_cos_sin_tables(k, seq_dimension=-2)
        if q.shape[2] != k.shape[2] and q_len is not None:
            return (
                llama_apply_rotary_pos_emb(
                    q,
                    self._cache[:, :, q_len - 1 : q_len],
                    use_complex=self.use_complex,
                ),
                llama_apply_rotary_pos_emb(k, self._cache, use_complex=self.use_complex),
            )
        return (
            llama_apply_rotary_pos_emb(q, self._cache, use_complex=self.use_complex),
            llama_apply_rotary_pos_emb(k, self._cache, use_complex=self.use_complex),
        )
