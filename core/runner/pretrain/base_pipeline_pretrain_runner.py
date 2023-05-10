"""base pipeline pretrain runner"""

import os.path as osp
import torch
from core.runner.asr.base_rnnt_runner import BaseRNNTRunner
from core.utils import logging
from ..base_runner import RUNNERS
from ..utils import get_time


@RUNNERS.register_module()
class BasePipelinePretrainRunner(BaseRNNTRunner):
    """
    BasePipelinePretrainRunner
    Use next_train_batch and next_valid_batch in forward to adapt to pipeline training.
    """

    def __init__(self, cfg, inference=False, export_onnx=False):
        """Init."""
        super().__init__(cfg, inference=inference, export_onnx=export_onnx)
        self.end_of_valid = False

    def pre_build_dataset(self, dataset_cfg):
        """pre_build_dataset"""

    @get_time('time')
    def train_iteration(self):
        '''pipeline train iteration.'''
        try:
            self.solution_out = self.solution(self.next_train_batch)
            self.train_log_buffer.update(self.solution_out)
        except RuntimeError as e:
            # TODO(zhengyijie): Currently missing exception handling for pipeline engine
            logging.warning(e)

    def next_valid_batch(self):
        '''next valid batch'''
        batch_data = self.valid_data_loader.next()
        if batch_data is None:
            self.valid_data_loader.reset()
            batch_data = self.valid_data_loader.next()
            self.end_of_valid = True
        return batch_data

    @torch.no_grad()
    def validation(self):
        '''valid'''
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self._val_iter = 0
        if self.solution_cfg.get('valid_multi_cer', False):
            dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
        else:
            dataset_names = [None]
        for dataset_name in dataset_names:
            self.valid_log_buffer.reset()
            self.end_of_valid = False
            while not self.end_of_valid:
                self.call_hook('before_val_iter')
                try:
                    validation_out = self.solution(self.next_valid_batch)
                    self._val_iter += 1
                    self.call_hook('after_val_iter')
                    self.valid_log_buffer.update(validation_out)
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            self.log_metric(self.valid_log_buffer, name=dataset_name)
        self.call_hook('after_val_epoch')
        self.mode = 'train'
        self.solution.train()
