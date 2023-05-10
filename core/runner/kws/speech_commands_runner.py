''' SpeechCommandsRunner '''
import os.path as osp
import torch
from core.runner.metric.kws_metric import KwsMetric
from core.dataset.speech_commands_dataset.dataset import SpeechCommandDataset
from .base_kws_runner import BaseKwsRunner
from ..base_runner import RUNNERS
from ...utils import logging


@RUNNERS.register_module()
class SpeechCommandsRunner(BaseKwsRunner):
    '''Speech Commands for KWS'''

    def __init__(self, cfg, inference=False, export_onnx=False):
        super().__init__(cfg, inference, export_onnx)
        self.inference_cfg = cfg.inference

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        if not self.is_inference:
            self.train_data_loader = SpeechCommandDataset(dataset_cfg, 'training')
            self.valid_data_loader = SpeechCommandDataset(dataset_cfg, 'validation')
        else:
            self.train_data_loader = None
            self.valid_data_loader = None

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = KwsMetric()
        self.valid_log_buffer = KwsMetric()
        self.inference_log_buffer = KwsMetric()

    @staticmethod
    def get_inference_data_list(dataset_cfg, inference_cfg):
        '''
        to get valid_file_list and train_file_list
        '''
        inference_data_root = dataset_cfg.get("data_root", None)
        inference_file_list = inference_cfg.get("test_file_list", None)
        inference_files = []
        inference_files += [osp.join(inference_data_root, inference_file_list)]
        return inference_files

    def build_inference_dataset(self, dataset_cfg):
        '''build inference dataset'''
        self.inference_data_loader = SpeechCommandDataset(dataset_cfg, 'testing')

    @torch.no_grad()
    def inference(self):
        '''inference'''
        self.build_inference_dataset(self.dataset_cfg)
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.inference_data_loader.reset()
        self.inference_log_buffer.reset()
        self.mode = 'test'
        batch_data = self.inference_data_loader.next()
        self._val_iter = 0
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                self.solution.eval()
                inference_out = self.solution(batch_data)
                self._val_iter += 1
                batch_data = self.inference_data_loader.next()
                self.call_hook('after_val_iter')
                self.inference_log_buffer.update(inference_out)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                continue
        self.call_hook('after_val_epoch')
        self.log_metric(self.inference_log_buffer)
