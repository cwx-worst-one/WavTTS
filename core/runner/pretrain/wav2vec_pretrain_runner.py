"""Wav2vecPretrainRunner"""

import os.path as osp
import torch
from core.runner.asr.base_rnnt_runner import BaseRNNTRunner
from core.runner.metric.asr_metric import Wav2vecMetric
from core.extensions import clear_cuda_error
from core.utils import logging
from core.dataset import get_meta
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class Wav2vecPretrainRunner(BaseRNNTRunner):
    """Wav2vecPretrainRunner"""

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transform cfg'''
        for cfg in dataset_cfg.batch_transform:
            if cfg.type == 'ComputeW2vMask':
                cfg['mask_prob'] = self.solution_cfg.mask_prob
                cfg['mask_length'] = self.solution_cfg.mask_length
                cfg['use_fbank'] = self.solution_cfg.wav2vec_use_fbank
                cfg['conv_feature_layers'] = self.solution_cfg.get('conv_feature_layers', '')
                cfg['mask_type'] = self.solution_cfg.mask_selection
                cfg['mask_other'] = self.solution_cfg.mask_other
                cfg['mask_minlen_type'] = self.solution_cfg.mask_minlen_type
                cfg['no_overlap'] = self.solution_cfg.no_mask_overlap
                cfg['min_space'] = self.solution_cfg.mask_min_space

    def pre_build_dataset(self, dataset_cfg):
        '''prepare for build dataset.'''
        if dataset_cfg.get('meta_file'):
            meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
            if isinstance(meta_data_root, (list, tuple)):
                meta_data_root = meta_data_root[0]
            meta_file = osp.join(meta_data_root, dataset_cfg.meta_file)
            self.meta_data = get_meta(meta_file)
        else:
            self.meta_data = {}
        self.solution_cfg.setdefault('fbank_dim', dataset_cfg.get("fbank_dim", 80))
        self.setup_transform_cfg(dataset_cfg)

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = Wav2vecMetric()
        self.valid_log_buffer = Wav2vecMetric()

    def train_iteration(self):
        self.solution.set_num_updates(self.iter)
        super().train_iteration()

    @torch.no_grad()
    def validation(self):
        '''validation func.'''
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self.valid_log_buffer.reset()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0
        if self.solution_cfg.get('valid_multi_cer', False):
            dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
        else:
            dataset_names = [None]
        for dataset_name in dataset_names:
            while batch_data is not None:
                self.call_hook('before_val_iter')
                try:
                    self.solution.eval()
                    validation_out = self.solution(batch_data)
                    self._val_iter += 1
                    batch_data = self.valid_data_loader.next()
                    self.call_hook('after_val_iter')
                    self.valid_log_buffer.update(validation_out)
                except RuntimeError as e:
                    clear_cuda_error()
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            self.log_metric(self.valid_log_buffer, name=dataset_name)
        self.after_validation()
