''' BaseVadRunner '''
import os.path as osp
import torch
from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    get_meta,
)
from core.runner.metric.vad_metric import VadMetric
from core.utils import logging
from ..base_runner import BaseRunner, RUNNERS


@RUNNERS.register_module()
class BaseVadRunner(BaseRunner):
    '''Base VAD Runner'''

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)
        self.tgt_vocab_size = dataset_cfg.tgt_vocab_size
        # align to 8
        # self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1)//8)*8
        logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
        self.solution_cfg.setdefault('tgt_vocab_size', self.tgt_vocab_size)

        # self.solution_cfg.setdefault('reorder_tgt_dict', self.reorder_tgt_dict)
        self.in_out_ratio = dataset_cfg.get('in_out_ratio', 3)
        if self.in_out_ratio < self.solution_cfg.get('downsampling_size', 3):
            self.in_out_ratio = self.solution_cfg.get('downsampling_size', 3)

        self.setup_transform_cfg(dataset_cfg)

    def load_cmvn(self, dataset_cfg):
        '''load cmvn'''
        meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
        if isinstance(meta_data_root, str):
            meta_file = osp.join(meta_data_root, dataset_cfg.meta_file)
        else:
            meta_file = osp.join(meta_data_root[0], dataset_cfg.meta_file)
        meta_data = get_meta(meta_file)
        self.cmvn_mean = meta_data['cmvn_mean']
        self.cmvn_var = meta_data['cmvn_var']

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        super().build_dataset(dataset_cfg)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        split_path_list_by_rank = dataset_cfg.get('split_path_list_by_rank', 1)
        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.train_item_trans,
            self.draw_batch_fn,
            device_transforms=self.train_device_trans,
            split_path_list_by_rank=split_path_list_by_rank,
            shuffle=True,
        )

        valid_split_each_dataset = self.solution_cfg.get('valid_multi_cer', False)
        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            self.val_bucket_schedule,
            dataset_cfg,
            self.valid_item_trans,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
            split_each_dataset=valid_split_each_dataset,
        )

    @staticmethod
    def get_test_file_list(dataset_cfg, test_sets):
        '''get test file list.'''
        data_root = dataset_cfg.get('data_root', None)
        if isinstance(data_root, list):
            data_root = data_root[0]
        test_data_root = dataset_cfg.get('test_data_root', data_root)
        test_files = []
        for test_file in test_sets:
            test_files += [osp.join(test_data_root, test_file)]
        return test_files

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = VadMetric()
        self.valid_log_buffer = VadMetric()

    def export_onnx(self):
        '''export onnx.'''
        if self.solution_cfg.get('model_do_cmvn', False):
            self.load_cmvn(self.dataset_cfg)
            self.solution.load_cmvn(self.cmvn_mean, self.cmvn_var)
        super().export_onnx()

    @torch.no_grad()
    def inference(self):
        '''inference'''
