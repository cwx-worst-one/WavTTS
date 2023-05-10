"""HuBERTPretrainRunner"""

import os.path as osp
from core.runner.pretrain.bestrq_pretrain_runner import BestrqPretrainRunner
from core.runner.metric.asr_metric import HuBERTMetric
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class HuBERTPretrainRunner(BestrqPretrainRunner):
    """HuBERTPretrainRunner"""

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = HuBERTMetric()
        self.valid_log_buffer = HuBERTMetric()
