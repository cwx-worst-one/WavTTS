"""BestrqPretrainRunner"""

import os.path as osp
import torch
from core.utils import logging
from core.extensions import clear_cuda_error
from core.runner.pretrain.wav2vec_pretrain_runner import Wav2vecPretrainRunner
from core.runner.metric.asr_metric import BestrqMetric, KmeansMetric
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class BestrqPretrainRunner(Wav2vecPretrainRunner):
    """BestrqPretrainRunner"""

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = BestrqMetric()
        self.valid_log_buffer = BestrqMetric()

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
        dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
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


@RUNNERS.register_module()
class KmeansTrainingRunner(BestrqPretrainRunner):
    """BestrqPretrainRunner"""

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = KmeansMetric()
        self.valid_log_buffer = KmeansMetric()

    @torch.no_grad()
    def inference(self):
        '''inference'''
        self.build_metrics()
        self.need_build_data_loader = True
        self.build_dataset(self.cfg.data)
        self.resume()
        self.validation()

    def after_validation(self):
        pass
