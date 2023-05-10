"""BertPretrainPipelineRunner"""


from .base_pipeline_pretrain_runner import BasePipelinePretrainRunner
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class BertPretrainPipelineRunner(BasePipelinePretrainRunner):
    """BertPretrainPipelineRunner"""
