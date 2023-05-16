'''optimizer helper'''
import torch
from packaging import version
from core.utils import get_world_size, dist_broadcast


class OptimizerHelper:
    '''optmizer helper'''

    def __init__(self, optimizer):
        '''init.'''
        self.optimizer = optimizer
        self.tensor_param = []

    def init_state(self):
        '''init state'''

    def get_tensor_param(self):
        '''get tensor paramm'''
        return self.tensor_param


class AdadeltaOptimizerHelper(OptimizerHelper):
    '''optimizer helper for adam'''

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.tensor_param = ['square_avg', 'acc_delta']

    def init_state(self):
        '''init state'''
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.optimizer.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data.float())
                    state['acc_delta'] = torch.zeros_like(p.data.float())


class AdamaxOptimizerHelper(OptimizerHelper):
    '''optimizer helper for adam'''

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.tensor_param = ['exp_avg', 'exp_inf']

    def init_state(self):
        '''init state'''
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.optimizer.state[p]
                if len(state) == 0:
                    if version.parse(torch.__version__) >= version.parse('1.12'):
                        state['step'] = torch.tensor(0.0)
                    else:
                        state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data.float())
                    state['exp_inf'] = torch.zeros_like(p.data.float())


class AdamOptimizerHelper(OptimizerHelper):
    '''optimizer helper for adam'''

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.tensor_param = ['exp_avg_sq', 'exp_avg', 'max_exp_avg_sq']
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if len(state) == 0 and (group.get('amsgrad') is not None):
                    self.tensor_param.append('sq')
                    return

    def init_state(self):
        '''init state'''
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.optimizer.state[p]
                if len(state) == 0:
                    if version.parse(torch.__version__) >= version.parse('1.12'):
                        step = 1 if group['capturable'] or group['fused'] else 0
                        state['step'] = torch.tensor((step,), dtype=torch.float, device='cuda')
                    else:
                        state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data.float())
                    state['exp_avg_sq'] = torch.zeros_like(p.data.float())
                    if group.get('amsgrad'):
                        state['sq'] = torch.zeros_like(p.data.flaot())


class SgdOptimizerHelper(OptimizerHelper):
    '''optimizer helper for sgd'''

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.tensor_param = []
        for group in self.optimizer.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                if momentum != 0:
                    self.tensor_param.append('momentum_buffer')
                    return

    def init_state(self):
        '''init state'''
        for group in self.optimizer.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if momentum != 0:
                    state = self.optimizer.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(d_p).detach()


OPTIMIZER_HELPER = dict(
    Adam=AdamOptimizerHelper,
    FusedAdam=AdamOptimizerHelper,
    SGD=SgdOptimizerHelper,
    FusedSGD=SgdOptimizerHelper,
    AdamW=AdamOptimizerHelper,
    Adamax=AdamaxOptimizerHelper,
    Adadelta=AdadeltaOptimizerHelper,
)


def get_optimizer_helper(optimizer):
    '''get optimizer helper'''
    optimizer_class_name = optimizer.__class__.__name__
    if optimizer_class_name not in OPTIMIZER_HELPER:
        raise RuntimeError('Not supported optimizer ', optimizer_class_name)
    return OPTIMIZER_HELPER[optimizer_class_name](optimizer)


def dist_broadcast_optimizer(
    model, optimizer, root_rank=0, tp_param_filter=None, tp_root=0, tp_group=None
):
    '''
    distributed broadcast optimizer.
    Args:
        optimizer(Optimizer): optimizer.
        root_rank(int): root rank.

    Return:
        Optimizer: broadcasted optimizer.
    '''
    # init optimizer's state
    # It should be in optimizer.
    if get_world_size() == 1 or optimizer.__class__.__name__ == 'OSS':
        return
    optimizer_helper = get_optimizer_helper(optimizer)
    optimizer_helper.init_state()
    tensor_param = optimizer_helper.get_tensor_param()
    # pylint:disable=too-many-nested-blocks
    for name, p in model.named_parameters():
        state = optimizer.state[p]
        for item in tensor_param:
            if item in state:
                tp_flag = False
                if tp_param_filter:
                    for fl in tp_param_filter:
                        if fl in name:
                            tp_flag = True
                            break
                if tp_flag:
                    dist_broadcast(state[item], root_rank=tp_root, group=tp_group)
                else:
                    dist_broadcast(state[item], root_rank=root_rank)
