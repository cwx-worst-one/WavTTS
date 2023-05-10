''' unfold '''
import torch
from core.extensions.panther_symbol import PantherUnfoldFunc


__all__ = [
    'PantherUnFold',
]


class PantherUnFold(torch.nn.Module):
    '''PantherUnFold'''

    __constants__ = ['kernel_size', 'dilation', 'padding', 'stride']

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        """forward"""
        return PantherUnfoldFunc.apply(
            x, self.kernel_size, self.dilation, self.padding, self.stride
        )

    def extra_repr(self):
        """extra_repr"""
        return (
            'kernel_size={kernel_size}, dilation={dilation}, padding={padding},'
            ' stride={stride}'.format(**self.__dict__)
        )
