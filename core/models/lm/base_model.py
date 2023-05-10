"""Lm base model"""
import os
import torch
from torch import nn

import openfst_python as fst

from core.utils import logging, dist_hdfs_get, load_checkpoint
from core.criterions import *


class FST:
    """Fst model, including hotword fst type and sentence fst type currently."""

    def load(self, path):
        """Load model from path. Support local and hdfs."""
        if os.path.exists(path):
            # pylint:disable=no-member
            self.fst = fst.Fst.read(path)
        else:
            file_name = os.path.basename(path)
            local_path = dist_hdfs_get(path, local_file=file_name)
            # pylint:disable=no-member
            self.fst = fst.Fst.read(local_path)

    def start(self):
        """Retrun start state of the fst."""
        start = -1
        if self.fst:
            start = self.fst.start()
        return start


class NNLM(nn.Module):
    """NNLM model."""

    def __init__(self):
        '''init.'''
        super().__init__()
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    def compatible_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        '''
        compatible for load nnlm.
        '''
        nnlm_prefix = 'nnlm_model.'
        if prefix.startswith(nnlm_prefix):
            return
        # nnlm_model prefix is not in this module, but is in state_dict.
        for k in list(state_dict.keys()):
            if not k.startswith(nnlm_prefix):
                continue
            new_k = k[len(nnlm_prefix) :]
            v = state_dict.pop(k)
            state_dict[new_k] = v

    def forward(self):
        """Forward function."""
        raise NotImplementedError

    @torch.no_grad()
    def load(self, path):
        '''resume NNLM'''
        if os.path.exists(path):
            load_checkpoint(self, path)
        else:
            file_name = os.path.basename(path)
            local_path = dist_hdfs_get(path, local_file=file_name)
            load_checkpoint(self, local_path)
        logging.info("Loading nnlm successfully.")

    @staticmethod
    def start():
        """Return start state."""
        return []
