# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from torch import nn
from core.solutions.inference import INFERS


SOLUTION_REGISTRY = {}

'''
 Helper class to register solution.
'''


def register_solution(name):
    def register_solution_cls(cls):
        if name in SOLUTION_REGISTRY:
            raise ValueError('Cannot register duplicate solution ({})'.format(name))
        if not issubclass(cls, BaseSolution):
            raise ValueError(
                'Solution ({}: {}) must extend ByteSolution'.format(name, cls.__name__)
            )
        SOLUTION_REGISTRY[name] = cls
        return cls

    return register_solution_cls


class BaseSolution(nn.Module):
    '''
    Abstract base class for solution in dolphin. Solution is consist of multi-components.
    - skeleton_model, core of model.
    '''

    def forward(self, batch_data):
        '''
        main function for training process.
        '''
        raise NotImplementedError

    def register_infers(self):
        '''
        register_infers
        '''
        self._infer_names = []

    def export(self, *_args, **kwargs):
        '''export onnx'''
        for infer_name in self._infer_names:
            INFERS[infer_name].export(**kwargs)
