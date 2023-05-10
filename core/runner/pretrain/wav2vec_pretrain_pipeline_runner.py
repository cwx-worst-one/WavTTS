"""Wav2vecPretrainPipelineRunner"""

from core.runner.metric.asr_metric import Wav2vecMetric
from .base_pipeline_pretrain_runner import BasePipelinePretrainRunner
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class Wav2vecPretrainPipelineRunner(BasePipelinePretrainRunner):
    """Wav2vecPipelinePretrainRunner"""

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = Wav2vecMetric()
        self.valid_log_buffer = Wav2vecMetric()
