'''
test dist model.
'''
import pytest
import torch
import torchvision
from core.utils.dist_util import distributed_init
from core.extensions import dist_parallel


# TODO(liyong): add mpi pytest support
@pytest.mark.isolate
def _test_pytorch_dist_model():
    '''test pytorch dist model.'''
    distributed_init()

    m = torchvision.models.resnet18().cuda()
    dist_parallel(m, None)
    x = torch.rand([32, 3, 224, 224]).cuda()
    y = m(x)
    y.sum().backward()
