''' optimizer constructor. '''
import torch
from packaging import version
from core.utils import build_from_cfg
from .builder import OPTIMIZER_BUILDERS, OPTIMIZERS


@OPTIMIZER_BUILDERS.register_module()
class DefaultOptimizerConstructor:
    """Default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``bias_lr_mult`` (float): It will be multiplied to the learning\
      rate for all bias parameters (except for those in normalization\
      layers).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight\
      decay for all bias parameters (except for those in\
      normalization layers and depthwise conv layers).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight\
      decay for all weight and bias parameters of normalization\
      layers.
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight\
      decay for all weight and bias parameters of depthwise conv\
      layers.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters\
      would not be added into optimizer. Default: False

    Args:
        - model (:obj:`nn.Module`): The model with parameters to be optimized.
        - optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are
                - `type`: class name of the optimizer.
            Optional fields are
                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        - paramwise_cfg (dict, optional): Parameter-wise options.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
    """

    def __init__(self, optimizer_cfg):
        '''init.'''
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict', f'but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg

    def __call__(self, model):
        '''__call__.'''
        # pylint: disable=too-many-branches
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_name = optimizer_cfg.pop('type')
        use_fused_optimizer = optimizer_cfg.pop('use_fused_optimizer', True)
        if version.parse(torch.__version__) >= version.parse('2.0'):
            # used Fused Optimizer in torch
            # torch.optim.Adam support fused in pytorch2.0
            optimizer_name = optimizer_name.replace('Fused', '')
            optimizer_name = optimizer_name.replace('ByteAdam', 'AdamW')
            if optimizer_name in ('Adam', 'AdamW'):
                optimizer_cfg['fused'] = use_fused_optimizer
            elif optimizer_name in (
                'ASGD',
                'Adadelta',
                'Adagrad',
                'Adamax',
                'NAdam',
                'RAdam',
                'RMSprop',
                'Rprop',
                'SGD',
            ):
                optimizer_cfg['foreach'] = use_fused_optimizer
        else:
            if (
                optimizer_cfg.get('use_fused_optimizer', True)
                and optimizer_name in ['Adam', 'AdamW', 'FusedByteAdam']
                and not optimizer_cfg.get('amsgrad', False)
                and not optimizer_cfg.get('maximize', False)
            ):
                optimizer_cfg['adam_w_mode'] = optimizer_name in ('AdamW', 'FusedByteAdam')
                optimizer_name = 'FusedAdam'
            if (
                optimizer_cfg.get('use_fused_optimizer', True)
                and optimizer_name == 'SGD'
                and not optimizer_cfg.get('maximize', False)
            ):
                optimizer_name = 'FusedSGD'
        optimizer_cfg['type'] = optimizer_name

        if not isinstance(optimizer_cfg.get('lr', 0), list):
            optimizer_cfg['params'] = model.parameters()
        else:
            optimizer_cfg['params'] = []
            name_list = []
            for lr_list in optimizer_cfg['lr']:
                optimizer_cfg['params'].append(
                    {
                        'params': [
                            parameter
                            for name, parameter in model.named_parameters()
                            for prefix in lr_list[1]
                            if prefix in name
                        ],
                        'lr': lr_list[0],
                    }
                )
                name_list += [
                    name
                    for name, _ in model.named_parameters()
                    for prefix in lr_list[1]
                    if prefix in name
                ]
            optimizer_cfg.pop('lr')

            if len(name_list) != len(set(name_list)):
                repeat_list = []
                for name in set(name_list):
                    if name_list.count(name) > 1:
                        repeat_list.append(name)
                raise RuntimeError("Have parameter defined by multiple lr," + str(repeat_list))

            if not len(list(model.parameters())) == len(name_list):
                undefined_list = []
                for name, _ in model.named_parameters():
                    if name_list.count(name) == 0:
                        undefined_list.append(name)
                raise RuntimeError("Have parameter lr is not set," + str(undefined_list))

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZER_BUILDERS.register_module()
class LayerwiseOptimizerConstructor:
    """Layerwise constructor for optimizers.

    refer to the implementation in DefaultOptimizerConstructor,
    lr setting in optimizer config use the format as follows to support layerwise lr:
        [(prefix1, lr1), ..., (prefixN, lrN), default_lr]
    """

    def __init__(self, optimizer_cfg):
        '''init.'''
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict', f'but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg

    def __call__(self, model):
        '''__call__.'''
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg
        if not isinstance(optimizer_cfg.get('lr', 0), list):
            raise ValueError('Layerwise lr, point out the layerwise lr list')

        optimizer_cfg['params'] = []
        # lr_list format: [(prefix1, lr1), .., (prefixN, lrN), default_lr ]
        lr_list = optimizer_cfg.get('lr', 0)
        default_lr = lr_list.pop(-1)

        for name, parameter in model.named_parameters():
            not_have_prefix = True
            for prefix_lr in lr_list:
                prefix, lr = prefix_lr
                if prefix in name:
                    not_have_prefix = False
                    optimizer_cfg['params'].append({'params': parameter, 'lr': float(lr)})
                    break
            if not_have_prefix:
                optimizer_cfg['params'].append({'params': parameter, 'lr': default_lr})
        optimizer_cfg.pop('lr')

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
