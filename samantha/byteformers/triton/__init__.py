import logging

import torch

logger = logging.getLogger(__name__)


def is_triton_available():
    if not torch.cuda.is_available():
        return False
    try:
        from .softmax import softmax as triton_softmax  # noqa

        return True
    except (ImportError, AttributeError) as e:
        logger.warning(f"Triton is not available: {e}")
        return False
