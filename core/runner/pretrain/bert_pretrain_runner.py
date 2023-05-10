"""BertPretrainRunner"""

from core.runner.lm.base_nnlm_runner import BaseLMRunner
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class BertPretrainRunner(BaseLMRunner):
    """BertPretrainRunner"""
