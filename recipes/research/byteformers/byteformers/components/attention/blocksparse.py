# Copyright (c) Facebook, Inc. and its affiliates, ByteDance. All rights reserved.
# CREDITS: Mainly taken from the xformers library, but some modifications
# to make it simpler.


import logging
import math

import torch
import torch.nn.functional as F
from einops import rearrange

from byteformers.components.attention.base import MultiHeadAttention
from byteformers.triton import is_triton_available

logger = logging.getLogger(__name__)


_is_blocksparse_available = is_triton_available()


if _is_blocksparse_available:
    from triton.ops.blocksparse import matmul as blocksparse_matmul  # type: ignore
    from triton.ops.blocksparse import softmax as blocksparse_softmax  # type: ignore


if _is_blocksparse_available:

    class BlockSparseAttention(MultiHeadAttention):
        r"""
        Thin wrap over the Triton blocksparse computations.
        The sparsity pattern is determined through the layout.

        .. warning: the layout is assumed to have dimensions [heads, seq, seq]
            If some dimensions are missing, we assume that the same layout is
            to be used across heads.

        .. warning: for now, the sequence (context) length has to be a power of
        two. This constraint could
            be relaxed in the future.

        .. warning: the block size has to be picked from [16, 32, 64]. Some
        speed is gained from bigger blocks. It is of course possible to
        reproduce coarser patterns given these primitives, as the user sees fit.

        """

        def __init__(self, layout: torch.Tensor, block_size: int = 16, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if layout.dim() == 2:
                logger.warning(
                    "The layout passed is lacking a head dimension and a batch"
                    " dimension"
                )
                logger.warning(
                    "Now assuming that the same layout is to be used across all heads"
                )
                layout = layout.unsqueeze(0).expand(self.n_heads, -1, -1)
                logger.warning(f"New layout dimensions: {layout.shape}")

            assert block_size in (
                16,
                32,
                64,
                128,
            ), "Only block sizes in [16, 32, 64, 128] are supported"

            # Pure blocksparse data
            self.layout = layout
            self.block_size = block_size

            # make sure that the head dimension is not folded down with the batch
            self.requires_head_dimension = True

            # key padding mask and attention mask must be passed in separately
            self.requires_same_k_q_dimensions = True

            # The underlying triton op does not support per element attention mask
            self.supports_attention_mask = False
            self.supports_key_padding_mask = False

        def create_triton_kernels(self, device):
            # blocksparse operators
            self.sparse_dot_sdd = blocksparse_matmul(
                self.layout,
                self.block_size,
                "sdd",
                trans_a=False,
                trans_b=True,
                device=device,
            )

            self.sparse_dot_dsd = blocksparse_matmul(
                self.layout,
                self.block_size,
                "dsd",
                trans_a=False,
                trans_b=False,
                device=device,
            )

            self.sparse_softmax = blocksparse_softmax(
                self.layout, self.block_size, device=device
            )

        def attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            return_attention: bool,
        ) -> torch.Tensor:
            r"""
            A thin wrap around the Triton blockparse attention operation

            .. note: Per element attention mask is not supported, but you can
                specify causality
            """

            # Delayed triton init, to make sure that we get the right device
            # Infer device from query
            if not hasattr(self, "sparse_dot_sdd"):
                self.create_triton_kernels(q.device)

            assert (
                q.shape[-2] == k.shape[-2]
            ), "Blocksparse requires the same dimensions for K and Q for now"

            assert (
                q.shape[-2] == self.layout.shape[-2] * self.block_size
            ), "Actual sequence size and layout are inconsistent"
            assert (
                k.shape[-2] == self.layout.shape[-2] * self.block_size
            ), "Actual sequence size and layout are inconsistent"

            assert (
                q.shape[-2] % self.block_size
            ) == 0, "Sequence length {}  must be a multiple of block size {}".format(  # noqa
                q.shape[-2], self.block_size
            )

            # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
            # When the computations are block sparse, the matrix types change
            # along the way:
            # - (sparse) attention matrix = (dense) Kt * (dense) Q
            q = q / math.sqrt(q.size(-1))

            sparse_att_mat = self.sparse_dot_sdd(q, k)

            # - softmax on the sparse attention matrix
            sparse_att_mat = self.sparse_softmax(
                sparse_att_mat, scale=self.scale, is_causal=self.is_causal
            )

            sparse_att_mat = F.dropout(sparse_att_mat, p=self.dropout_p)

            # - then (dense) attention is (sparse) attention matrix * dense (value)
            score = self.sparse_dot_dsd(sparse_att_mat, v)

            score = rearrange(score, "b n_heads s d_k -> b s (n_heads d_k)")
            if return_attention:
                return self.WO(score), sparse_att_mat
            return self.WO(score)
