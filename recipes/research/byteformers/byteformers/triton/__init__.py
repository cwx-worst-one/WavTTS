import logging

import torch

logger = logging.getLogger(__name__)


def compute_once(func):
    value = None

    def func_wrapper():
        nonlocal value
        if value is None:
            value = func()
        return value

    return func_wrapper


@compute_once
def is_triton_available():
    if not torch.cuda.is_available():
        return False
    try:
        from byteformers.triton.softmax import softmax as triton_softmax  # noqa

        return True
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"A matching Triton is not available, some optimizations will not be enabled.\nError caught was: {e}"  # noqa
        )
        return False
