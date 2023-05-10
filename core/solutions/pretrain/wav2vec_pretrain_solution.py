"""wav2vec pretrain"""

from core.models.pretrained.wav2vec2_model import *
from core.models.pretrained.wav2vec2_pipeline_model import *
from core.solutions.base_solution import BaseSolution, register_solution


@register_solution("BaseWav2vecPretrainModel")
class BaseWav2vecPretrainModel(BaseSolution):
    """wav2vec pretrain model"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args
        self.wav2vec_model = eval(args.wav2vec_type)(args)

    def forward(self, batch_data):
        """forward"""
        forward_out = self.wav2vec_model(batch_data)
        return forward_out

    def forward_attention(self, batch_data):
        """forward attention"""
        forward_out = self.wav2vec_model.forward_attention(batch_data)
        return forward_out

    def set_num_updates(self, num_updates):
        '''set_num_updates.'''
        self.wav2vec_model.set_num_updates(num_updates)
