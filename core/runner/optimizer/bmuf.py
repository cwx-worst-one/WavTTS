''' BMUF. '''
import torch
from core.utils import dist_allreduce, dist_broadcast_model
from core.utils.dist_util import dist_parallel
from core.extensions import fused_bmuf, gather_fn, master_params
from .optimizer_helper import get_optimizer_helper, dist_broadcast_optimizer
from .builder import OPTIMIZERS


@OPTIMIZERS.register_module()
class BMUF:
    '''
    BMUF implemented.
    '''

    def __init__(self, **bmuf_config):
        '''init.'''
        self.bmuf_warmup_steps = bmuf_config['warmup_steps']
        self.use_nesterov = bmuf_config['use_nesterov']
        self.bmuf_block_lr = bmuf_config['block_lr']
        self.bmuf_block = bmuf_config['block']
        self.bmuf_block_momentum = bmuf_config['block_momentum']
        self.average_sync = bmuf_config.get('average_sync', True)
        self.warmup_sync = bmuf_config.get('warmup_sync', False)
        if self.warmup_sync:
            self.dist_parrallel_hook = None

        self.global_params = None
        self.delta = None

    def set_iters(self, iters=None, **_kwargs):
        '''set iters'''
        if self.warmup_sync and self.bmuf_warmup_steps < (iters + 1) and self.dist_parrallel_hook:
            # clear dist parrallel hook during bmuf warmup
            self.dist_parrallel_hook.clear(None)

    def need_grad_sync(self, iters):
        '''wehther need to do grad sync.'''
        return self.warmup_sync and self.bmuf_warmup_steps >= (iters + 1)

    def init(self, solution, optimizer, train_cfg, iters, **_kwargs):
        '''before run.'''
        if self.need_grad_sync(iters):
            assert self.bmuf_warmup_steps % train_cfg.get('grad_accum_step', 1) == 0
            _, self.dist_parrallel_hook = dist_parallel(solution, optimizer, **train_cfg)

    def init_bmuf(self, optimizer):
        '''init bmuf.'''
        params = list(master_params(optimizer))
        # Initialize global momentum parameters and store global copy on each worker
        self.global_params = [torch.zeros_like(p.data) for p in params]
        self.delta = [p.data.new_zeros(p.data.size()) for p in params]
        for param, global_param in zip(params, self.global_params):
            global_param.copy_(param.data)
        return params

    @torch.no_grad()
    def step(self, solution, optimizer, params, iters, **_kwargs):
        '''step.'''
        if self.global_params is None and (iters + 1) > self.bmuf_warmup_steps:
            # resume from chkpt, exec init_bmuf to init bmuf parameters
            self.init_bmuf(optimizer)
        # after warmup
        if self.bmuf_warmup_steps == (iters + 1):
            # clear dist parrallel hook during bmuf warmup
            if self.warmup_sync and self.dist_parrallel_hook:
                self.dist_parrallel_hook.clear(optimizer)
            self.init_bmuf(optimizer)
            # broadcast parameters and optimizer
            dist_broadcast_model(solution, root_rank=0)
            dist_broadcast_optimizer(solution, optimizer, root_rank=0)
            # update global_params
            for (param, global_param) in zip(params, self.global_params):
                global_param.copy_(param.data)
        # bmuf
        # end of block
        if (iters + 1) > self.bmuf_warmup_steps and (iters + 1) % self.bmuf_block == 0:
            index = 0
            for (param, delta, global_param) in zip(params, self.delta, self.global_params):
                # G(t) = Averge(param) - global_param
                tensor_name = str('bmuf/').join(str(index))
                index += 1
                sync_param = torch.zeros_like(param.data)
                sync_param.copy_(param.data)
                # in case of tensor in-place when use byteps
                sync_param = dist_allreduce(sync_param, name=tensor_name)
                sync_param -= global_param
                # delta(t) = block_momentum * delta(t-1) + block_lr*G(t)
                delta.data.copy_(
                    self.bmuf_block_momentum * delta.data + self.bmuf_block_lr * sync_param
                )
                # update
                # W(t) = W(t-1) + delta(t), W(t-1) should be global_param instead of param
                param.data.copy_(global_param + delta.data)
                if self.use_nesterov:
                    param.data.copy_(param.data + self.bmuf_block_momentum * delta.data)
                global_param.copy_(param.data)
            if self.average_sync:
                self.average_params(optimizer)

            # do master param to model param, for amp level O2
            if hasattr(optimizer, '_master_params_to_model_params'):
                optimizer._master_params_to_model_params()  # pylint: disable=protected-access

    def clear(self):
        '''after run.'''
        self.global_params = None
        self.delta = None
        if self.warmup_sync and self.dist_parrallel_hook:
            self.dist_parrallel_hook.clear(None)
            self.dist_parrallel_hook = None

    @staticmethod
    def average_params(optimizer):
        '''average optimizer state'''
        optimizer_helper = get_optimizer_helper(optimizer)
        tensor_param = optimizer_helper.get_tensor_param()
        state_dict = optimizer.state_dict()
        for key in state_dict["state"]:
            value = state_dict["state"][key]
            for item in tensor_param:
                if item in value:
                    value[item] = dist_allreduce(
                        value[item], name=str(key).join('/{}'.format(item))
                    )


@OPTIMIZERS.register_module()
class FusedBMUF(BMUF):
    '''
    Fused BMUF implemented.
    '''

    def __init__(self, **bmuf_config):
        '''init.'''
        super().__init__(**bmuf_config)
        self.combined_tensor = None

    def init_bmuf(self, optimizer):
        '''init bmuf.'''
        params = super().init_bmuf(optimizer)
        total_numel = sum(p.numel() for p in params)
        self.combine_tensor(optimizer, total_numel)
        return params

    def clear(self):
        '''after run.'''
        super().clear()
        self.combined_tensor = None

    @torch.no_grad()
    def step(self, solution, optimizer, params, iters, **_kwargs):
        '''step.'''
        if self.global_params is None and (iters + 1) > self.bmuf_warmup_steps:
            # resume from chkpt, exec init_bmuf to init bmuf parameters
            self.init_bmuf(optimizer)
        # after warmup
        if self.bmuf_warmup_steps == (iters + 1):
            # clear dist parrallel hook during bmuf warmup
            if self.warmup_sync and self.dist_parrallel_hook:
                self.dist_parrallel_hook.clear(optimizer)
            self.init_bmuf(optimizer)
            # broadcast parameters and optimizer
            dist_broadcast_model(solution, root_rank=0)
            dist_broadcast_optimizer(solution, optimizer, root_rank=0)
            # update global_params
            for (param, global_param) in zip(params, self.global_params):
                global_param.copy_(param.data)
        # bmuf end of block
        if (iters + 1) > self.bmuf_warmup_steps and (iters + 1) % self.bmuf_block == 0:
            # combine solution weight tensor
            sync_params = gather_fn(params, self.combined_tensor)

            # do all_reduce
            dist_allreduce(self.combined_tensor, name='all_param')

            # do update
            fused_bmuf(
                params,
                sync_params,
                self.delta,
                self.global_params,
                self.bmuf_block_lr,
                self.bmuf_block_momentum,
                self.use_nesterov,
            )

            # do master param to model param, for amp level O2
            if hasattr(optimizer, '_master_params_to_model_params'):
                optimizer._master_params_to_model_params()  # pylint: disable=protected-access

    def combine_tensor(self, optimizer, total_numel):
        '''reorg optimizer state params'''
        if not self.average_sync:
            self.combined_tensor = torch.zeros(total_numel, dtype=torch.float32, device='cuda')
            return
        optimizer_helper = get_optimizer_helper(optimizer)
        tensor_num = len(optimizer_helper.tensor_param)
        self.combined_tensor = torch.zeros(
            (tensor_num + 1) * total_numel, dtype=torch.float32, device='cuda'
        )

        offset = total_numel  # first total_numel for sync params of solution.
        optimizer_helper.init_state()
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                for key in optimizer_helper.get_tensor_param():
                    if key not in state:
                        continue
                    if torch.is_tensor(state[key]):
                        numel = p.numel()
                        tensor_value = self.combined_tensor[offset : offset + numel].view(p.shape)
                        offset += numel
                        tensor_value.copy_(state[key])
                        state[key] = tensor_value
