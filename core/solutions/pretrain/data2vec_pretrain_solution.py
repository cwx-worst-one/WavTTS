"""data2vec pretrain"""

from core.models.pretrained.data2vec_model import *
from core.solutions.base_solution import BaseSolution, register_solution


@register_solution("BaseData2vecPretrainModel")
class BaseData2vecPretrainModel(BaseSolution):
    """data2vec pretrain model"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args
        self.data2vec_model = eval(args.data2vec_type)(args)

    def forward(self, batch_data):
        """forward"""
        forward_out = self.data2vec_model(batch_data)
        return forward_out

    def set_num_updates(self, num_updates):
        '''set_num_updates.'''
        self.data2vec_model.set_num_updates(num_updates)
