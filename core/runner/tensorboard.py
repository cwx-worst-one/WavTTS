#!/usr/bin/env python3
# coding=utf-8
'''Self wrapped tensorboard.SummaryWriter.'''
# pylint: skip-file
import os

import tensorboard.compat
import tensorboard.lazy as _lazy


@_lazy.lazy_load("tensorboard.compat.tf")
def dolphin_tf():
    """Provide the root module of a TF-like API for use within TensorBoard.

    By default this is equivalent to `import tensorflow as tf`, but it can be used
    in combination with //tensorboard/compat:tensorflow (to fall back to a stub TF
    API implementation if the real one is not available) or with
    //tensorboard/compat:no_tensorflow (to force unconditional use of the stub).

    Returns:
        The root module of a TF-like API, if available.

    Raises:
        ImportError: if a TF-like API is not available.
    """
    try:
        from tensorboard.compat import notf  # noqa: F401
    except ImportError:
        pass
    from tensorboard.compat import tensorflow_stub

    return tensorflow_stub


tensorboard.compat.tf = dolphin_tf

import torch.utils.tensorboard

SummaryWriter = torch.utils.tensorboard.SummaryWriter
