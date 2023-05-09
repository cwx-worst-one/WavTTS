# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
#

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
def _is_triton_available():
    if not torch.cuda.is_available():
        return False
    try:
        from samantha.byteformers.triton.softmax import (  # noqa
            softmax as triton_softmax,
        )

        return True
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"A matching Triton is not available, some optimizations will not be enabled.\nError caught was: {e}"  # noqa
        )
        return False
