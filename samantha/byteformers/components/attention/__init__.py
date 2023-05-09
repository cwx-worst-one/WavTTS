from .base import MultiHeadAttention, SeerAttention  # noqa
from .blocksparse import _is_blocksparse_available  # noqa

if _is_blocksparse_available:
    from .blocksparse import BlockSparseAttention  # noqa
