''' BaseVadRunner '''
import os.path as osp
import torch
from core.dataset import BalancedHDFSDataset, ValidHDFSDataset, get_meta
from core.runner.metric.vad_metric import TsVadMetric
from ..base_runner import BaseRunner, RUNNERS


@RUNNERS.register_module()
class BaseVadTsRunner(BaseRunner):
    '''Base VAD Runner'''

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)
        self.setup_transform_cfg(dataset_cfg)
        self.meta_data_list = [get_meta(meta_file) for meta_file in self.meta_file_list]
        if self.meta_data_list and 'embedding_keys' in self.meta_data_list[0]:
            self.meta_data['embedding_keys'] = self.meta_data_list[0]['embedding_keys']

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        super().build_dataset(dataset_cfg)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        self.train_data_loader = BalancedHDFSDataset(
            self.train_file_list,
            dataset_cfg,
            self.train_item_trans,
            self.draw_batch_fn,
            device_transforms=self.train_device_trans,
        )

        dataset_cfg.chunk_size = dataset_cfg.get("valid_chunk_size", 2)
        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            '',
            dataset_cfg,
            self.valid_item_trans,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
            device_transforms=self.valid_device_trans,
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
        self.train_log_buffer = TsVadMetric()
        self.valid_log_buffer = TsVadMetric()

    @torch.no_grad()
    def inference(self):
        '''inference'''
