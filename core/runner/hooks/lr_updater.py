''' lr updater. '''
# pylint:disable=too-many-lines

import math
from math import cos, pi, inf
import inspect
import torch
from core.utils import Registry
from .hook import HOOKS, Hook


@HOOKS.register_module()
class LrUpdaterHook(Hook):
    """LR Scheduler

    Args:
        - by_epoch(bool): learning rate 是否随着epoch的变化而变化,default=True.
        - warmup(string): warmup使用的类型,可以是None(不使用warmup),'constant','linear','exp'.
        - warmup_iters(int): warmup持续的epochs 或者 iterations,default=0.
        - warmup_ratio(float): 初始的lr值,default=0.1.
        - warmup_by_epoch(bool): 当 warm_by_epoch = True ,warmup_iters 代表着warmup持续的epoch.
        - warmup_by_epoch(bool): 当 warm_by_epoch = False,代表着warmup持续的iterations.default=False

    """

    def __init__(
        self,
        by_epoch=True,
        warmup=None,
        warmup_iters=0,
        warmup_ratio=0.1,
        warmup_by_epoch=False,
        **_kwargs,
    ):
        """init."""
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ["constant", "linear", "exp"]:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"'
                )
        if warmup is not None:
            assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        """set lr."""
        # pylint: disable=no-self-use
        for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
            param_group["lr"] = lr

    def get_lr(self, runner, base_lr):
        """get lr."""
        raise NotImplementedError

    def get_regular_lr(self, runner):
        """get regular lr."""
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        """get warmup lr."""
        if self.warmup == "constant":
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.base_lr]
        elif self.warmup == "linear":
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.base_lr]
        elif self.warmup == "exp":
            k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.base_lr]
        return warmup_lr

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["initial_lr"] for group in runner.optimizer.param_groups]

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        self.regular_lr = self.get_regular_lr(runner)
        if self.warmup is None or cur_iter >= self.warmup_iters:
            self._set_lr(runner, self.regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(runner, warmup_lr)

    def state_dict(self):
        """state dict."""
        # pylint: disable=no-self-use
        # no need to save any values
        return {}

    def load_state_dict(self, state_dict):
        """state dict."""
        self.__dict__.update(state_dict)


@HOOKS.register_module()
class FixedLrUpdaterHook(LrUpdaterHook):
    """固定的LR

    .. note::

        之后所有类型的lr方法都会以lr_updater为基类,所以相同的参数不在赘述!

    Args:
            - 无

    Example::

            lr_scheduler=dict(
                by_epoch=False,
                warmup='linear',
                warmup_iters=100,
                warmup_ratio=0.001,
                warmup_by_epoch=False,
                policy='fixed',
            )
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """直接返回lr"""
        return base_lr


@HOOKS.register_module()
class StepLrUpdaterHook(LrUpdaterHook):
    """根据gamma每到step_size epochs 或者 step_size iter时降低每一组参数的学习率

    Args:
        - step(int): 如果step是int那么每step次,lr=lr*gamma.
        - step(list): 如果step是list,那么在list处,lr=lr*gamma.
        - gamma(float): 每次相乘的系数.

    Example::

            lr_scheduler=dict(
                by_epoch=True,
                warmup=None,
                warmup_iters=0,
                warmup_ratio=1e-3,
                warmup_by_epoch=False,
                policy='step',
                gamma=0.8,
                step=[3 + i for i in range(50)]
            ),

    """

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """直接返回lr"""
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma ** (progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma**exp


@HOOKS.register_module()
class PlateauLrUpdaterHook(LrUpdaterHook):
    """先warmup后exp的LR

    .. note::

        | decay_rate = (end_lr / peak_lr) ** (each_decay_step / (end_decay_step - start_decay_step))
        | decay_rate 代表了每次调整lr时乘上的参数,由设定的参数决定
        | progress = decay_rate ** ((runner.iter - start_decay_step) // each_decay_step + 1)
        | progress   代表了最后lr的变化,每each_decay_step多乘一个decay_rate

    Args:
        - end_lr(float):终止时lr的值
        - peak_lr(float):峰值lr
        - each_decay_step(int):设定了每隔多少个step调整一次lr,以及每次调整lr需要乘的参数
        - end_decay_step(int):终止调整lr的step
        - start_decay_step(int):开始调整lr的step=2

    Example::

            lr_scheduler=dict(
                policy='Plateau',
                by_epoch=False,
                stop_warmup_step=1000,
                start_decay_step=45000000,
                end_decay_step=750000000,
                each_decay_step=3000,
                peak_lr=0.001,
                init_lr=0,
                shrink_time=0.1,
                end_lr=0.0001,
                learning_rate=0.001,
                warmup='linear',
                warmup_iters=1000,
                warmup_ratio=1e-3,
                warmup_by_epoch=False,
            ),
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """
        返回目前的lr

        | 如果 runner.iter>warmup_iters 且 runner.iter<=start_decay_step 结果为base_lr
        | 如果 runner.iter>end_decay_step 结果为end_lr
        | others:decay_rate = (end_lr / peak_lr) ** (each_decay_step /\
        (end_decay_step - start_decay_step))
        | progress = decay_rate ** ((runner.iter - start_decay_step) // each_decay_step + 1)

        返回 base_lr*progress
        """
        end_lr = float(self.kwargs["end_lr"])
        peak_lr = float(self.kwargs["peak_lr"])
        each_decay_step = int(self.kwargs["each_decay_step"])
        end_decay_step = int(self.kwargs["end_decay_step"])
        start_decay_step = int(self.kwargs["start_decay_step"])
        if runner.iter >= self.warmup_iters and runner.iter <= start_decay_step:
            return base_lr
        if runner.iter > end_decay_step:
            return end_lr
        decay_rate = (end_lr / peak_lr) ** (each_decay_step / (end_decay_step - start_decay_step))
        progress = decay_rate ** ((runner.iter - start_decay_step) // each_decay_step + 1)
        return base_lr * progress

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        self.regular_lr = self.get_regular_lr(runner)
        if self.warmup and cur_iter <= self.warmup_iters:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(runner, warmup_lr)
        else:
            self._set_lr(runner, self.regular_lr)


@HOOKS.register_module()
class TriStageLrUpdaterHook(LrUpdaterHook):
    """分三个阶段的LR

    .. note::

        | self.peak_lr = lr  ,设置 peak_lr
        | self.init_lr = init_lr_scale * lr ,设置 init_lr
        | self.final_lr = final_lr_scale * lr, 设置 final_lr
        | self.warmup_steps = int(max_train_steps * phase_ratio[0]) ,设置warm_steps
        | self.hold_steps = int(max_train_steps * phase_ratio[1]) ,设置hold_steps
        | self.decay_steps = int(max_train_steps * phase_ratio[2]) ,设置decay_steps
        | self.warmup_rate = ((self.peak_lr - self.init_lr) / self.warmup_steps\
        if self.warmup_steps != 0 else 0)
        | self.decay_factor = -math.log(final_lr_scale) / self.decay_steps
        | steps_in_stage = iter-offset

    Args:
        - lr (int): 最大的lr.
        - init_lr_scale (float): 设置初始lr
        - final_lr_scale(float): 设置最终lr
        - max_train_steps(int): 最大的训练步数
        - phase_ratio(string): 相对比例,可以用来设置 warmup_steps,hold_steps,decay_steps
        - warmup_steps(int): 当step<warmup_steps时,state=0
        - hold_steps(string): 当step<warm_steps+hold_steps时,stage=1
        - decay_steps(string): 当step<warm_steps+hold_steps+decay_steps,stage=2

    Example::

        lr_scheduler=dict(
            by_epoch=False,
            policy="TriStage",
            lr=2e-5,
            phase_ratio="[0.1,0.4,0.5]",
            init_lr_scale=0.01,
            final_lr_scale=0.05,
            max_train_steps=4000,
         ),
    """

    def __init__(
        self,
        lr,
        init_lr_scale,
        final_lr_scale,
        max_train_steps,
        phase_ratio=None,
        warmup_steps=None,
        hold_steps=None,
        decay_steps=None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(**kwargs)
        # calculate LR at each point
        self.peak_lr = lr
        self.init_lr = init_lr_scale * lr
        self.final_lr = final_lr_scale * lr

        # remember the steps at each stage
        if phase_ratio is not None:
            phase_ratio = eval(phase_ratio)
            assert max_train_steps > 0
            assert sum(phase_ratio) == 1, "phase ratios must add up to 1"
            self.warmup_steps = int(max_train_steps * phase_ratio[0])
            self.hold_steps = int(max_train_steps * phase_ratio[1])
            self.decay_steps = int(max_train_steps * phase_ratio[2])
        else:
            self.warmup_steps = warmup_steps
            self.hold_steps = hold_steps
            self.decay_steps = decay_steps

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        )
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

    def get_lr(self, runner, base_lr):
        """
        返回目前的lr

        | 如果 stage=0  结果为 init_lr + warmup_rate * steps_in_stage
        | 如果 stage=1  结果为 peak_lr
        | 如果 stage=2  结果为 peak_lr * math.exp(-decay_factor * steps_in_stage)
        | 如果 stage=3  结果为 final_lr

        """
        stage, steps_in_stage = self._decide_stage(runner.iter)
        if stage == 0:
            new_lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            new_lr = self.peak_lr
        elif stage == 2:
            new_lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            new_lr = self.final_lr
        else:
            raise ValueError("Undefined stage")
        return new_lr

    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset


@HOOKS.register_module()
class LayerwiseTriStageLrUpdaterHook(TriStageLrUpdaterHook):
    """继承自TriStageLrUpdaterHook,重置final lr

    .. note::

        lr_prefix_multiplier_list format:
        [('prefix1', lr_multiplier1), ..., ('prefixN', lr_multiplierN)],
        final lr of parameters with prefix = scheduled_lr * lr_multiplier
    """

    def __init__(
        self,
        lr,
        init_lr_scale,
        final_lr_scale,
        max_train_steps,
        phase_ratio=None,
        warmup_steps=None,
        hold_steps=None,
        decay_steps=None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(
            lr,
            init_lr_scale,
            final_lr_scale,
            max_train_steps,
            phase_ratio,
            warmup_steps,
            hold_steps,
            decay_steps,
            **kwargs,
        )

    def _set_lr(self, runner, lr_groups):
        """set layerwise lr."""
        # pylint: disable=no-self-use
        model = runner.solution
        if hasattr(runner.solution, "module"):
            model = runner.solution.module

        lr_cfg = runner.lr_cfg
        optimizer_cfg = runner.optimizer_cfg
        # lr_prefix_multiplier_list format:
        #   [('prefix1', lr_multiplier1), ..., ('prefixN', lr_multiplierN)]
        #   final lr of parameters with prefix = scheduled_lr * lr_multiplier
        lr_prefix_multiplier_list = lr_cfg.get("lr_prefix_multiplier_list", 0)
        assert isinstance(lr_prefix_multiplier_list, list)

        scheduled_lr_list = []
        for name, _ in model.named_parameters():
            not_have_prefix = True
            for lr_prefix_multiplier in lr_prefix_multiplier_list:
                prefix, multiplier = lr_prefix_multiplier
                if prefix in name:
                    not_have_prefix = False
                    scheduled_lr_list.append(lr_groups[0] * float(multiplier))
                    break
            if not_have_prefix:
                scheduled_lr_list.append(lr_groups[0])

        if "lr" in optimizer_cfg:
            optimizer_cfg.pop("lr")

        for param_group, lr in zip(runner.optimizer.param_groups, scheduled_lr_list):
            param_group["lr"] = lr


@HOOKS.register_module()
class ExpLrUpdaterHook(LrUpdaterHook):
    """每一步都调整的LR

    .. note::

        | by_epoch=true 那么根据epoch修改lr,否则根据iter修改lr,赋值给progress
        | 每一个step,lr=lr*gamma

    Args:
        - gamma(float): 修改lr时乘的参数

    """

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """
        返回目前的lr

        结果为 base_lr*gamma*progress

        """
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


@HOOKS.register_module()
class PolyLrUpdaterHook(LrUpdaterHook):
    """从base_lr逐渐降低到min_lr

    .. note::

        | coeff = (1 - (progress - self.warmup_iters) / (max_progress - self.warmup_iters))\
        ** self.power
        | 如果by_epoch=True   progress=epoch,max_progress=max_epochs
        | 如果by_epoch=False  progress=iter,max_progress=max_iters


    Args:
        - power(float): 在设定coeff时,乘方数
        - min_lr(float): 最小的lr,最后会达到的值

    Example::

        lr_scheduler=dict(
            by_epoch=False,
            warmup='linear',
            warmup_iters=800,
            warmup_ratio=0.00125,
            warmup_by_epoch=False,
            policy='poly',
            ignore_warmup=False,
        ),

    """

    def __init__(self, power=1.0, min_lr=0.0, **kwargs):
        self.power = power
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """
        返回目前的lr=(base_lr - self.min_lr) * coeff + self.min_lr
        """

        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (
            1 - (progress - self.warmup_iters) / (max_progress - self.warmup_iters)
        ) ** self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@HOOKS.register_module()
class InvLrUpdaterHook(LrUpdaterHook):
    """根据1/(1+gamma*step)^p每一步衰减lr

    .. note::

        | 如果by_epoch=True   progress=epoch
        | 如果by_epoch=False  progress=iter


    Args:
        - power(float): 计算lr时的乘方数
        - gamma(float): 计算lr时,progress的系数

    """

    def __init__(self, gamma, power=1.0, offset=0.0, scale=1.0, **kwargs):
        self.gamma = gamma
        self.power = power
        self.offset = offset
        self.scale = scale
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """
        返回目前的lr=base_lr * (1 + self.gamma * progress) ** (-self.power)
        """
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.scale * (1 + self.gamma * progress + self.offset) ** (-self.power)


@HOOKS.register_module()
class CosineAnealingLrUpdaterHook(LrUpdaterHook):
    """余弦lr,从start到end

    .. note::

        | 如果by_epoch=True                progress=epoch,max_progress=max_epochs
        | 如果by_epoch=False               progress=iter,max_progress=max_iters
        | 如果target_lr_ratio = none       target_lr=min_lr
        | 如果target_lr_ratio != none      target_lr=base_lr*min_lr_ratio


    Args:
        - min_lr(float): 最小的lr值
        - min_lr_ratio(float): 计算target_lr,未设定时 target_lr=min_lr

    Example::

        lr_scheduler=dict(
            by_epoch=True,
            min_lr = 0,
            policy='CosineAnealing'
        ),

    """

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """
        返回目前的lr=annealing_cos(base_lr * start_ratio, base_lr * end_ratio, progress /\
        (end_iter - start_iter))

        .. note::
            | def annealing_cos(start, end, factor):
            | 从start 到 end 的一根余弦曲线
            | cos_out = cos(pi * factor) + 1
            | return end + 0.5 * (start - end) * cos_out
        """

        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class CyclicLrUpdaterHook(LrUpdaterHook):
    """以余弦方式周期性上升下降的lr

    .. note::
        Implemet the cyclical learning rate policy (CLR) described
        in https://arxiv.org/pdf/1506.01186.pdf

        | Different from the original paper, we use cosine anealing rather than\
        triangular policy inside a cycle. This improves the performance in the 3D detection area.

    .. note::
        | max_iter_per_phase = runner.max_iters 代表周期
        | 有两组参数,一组上升一组下降分别为
        | [0,iter_up_phase,max_iter_per_phase,1,target_ratio[0]]
        | [iter_up_phase,max_iter_per_phase,max_iter_per_phase]
        | [x1,x2,x3,x4,x5,x6]
        | 周期为x3,从x1到x2步,输出从x5 变为x6

    Args:
        - target_ratio(tuple[float]): 最大的lr和最小的lr与初始lr的相对比例
        - cyclic_times(float): 训练期间的循环次数
        - step_ratio_up(float): 循环中的LR上升速率
        - by_epoch(bool): 是否根据epoch来更新LR,目前只能等于False

    """

    # pylint: disable=duplicate-code
    def __init__(
        self,
        by_epoch=False,
        target_ratio=(10, 1e-4),
        cyclic_times=1,
        step_ratio_up=0.4,
        **kwargs,
    ):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            # none sense comments, avoid duplicate-code
            target_ratio = (
                (target_ratio[0], target_ratio[0] / 1e5) if len(target_ratio) == 1 else target_ratio
            )
        else:
            raise ValueError(
                f"target_ratio should be either float or tuple, got {type(target_ratio)}"
            )

        assert len(target_ratio) == 2, '"target_ratio" must be list or tuple of two floats'
        # none sense comments, avoid duplicate-code
        assert 0 <= step_ratio_up < 1.0, '"step_ratio_up" must be in range [0,1)'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.lr_phases = []  # init lr_phases

        assert not by_epoch, 'currently only support "by_epoch" = False'
        super().__init__(by_epoch, **kwargs)

    def before_run(self, runner):
        super().before_run(runner)
        # initiate lr_phases
        # total lr_phases are separated as up and down
        max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.lr_phases.append([0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
        self.lr_phases.append(
            [
                iter_up_phase,
                max_iter_per_phase,
                max_iter_per_phase,
                self.target_ratio[0],
                self.target_ratio[1],
            ]
        )

    # pylint: disable=inconsistent-return-statements
    def get_lr(self, runner, base_lr):
        """
        返回目前的lr=annealing_cos(base_lr, target_lr, progress / max_progress)

        .. note::
            | def annealing_cos(start, end, factor):
            | 从start 到 end 的一根余弦曲线
            | cos_out = cos(pi * factor) + 1
            | return end + 0.5 * (start - end) * cos_out

        """
        curr_iter = runner.iter
        for (
            start_iter,
            end_iter,
            max_iter_per_phase,
            start_ratio,
            end_ratio,
        ) in self.lr_phases:
            curr_iter %= max_iter_per_phase
            if start_iter <= curr_iter < end_iter:
                progress = curr_iter - start_iter
                return annealing_cos(
                    base_lr * start_ratio,
                    base_lr * end_ratio,
                    progress / (end_iter - start_iter),
                )


@HOOKS.register_module()
class ValidLrUpdaterHook(LrUpdaterHook):
    """一段epochs,loss没有下降时,lr就会减少

    .. note::
        Refer to:
        https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau

    .. note::
        当一段连续的epochs,loss没有下降,lr就会减少

    Args:
        - gamma(float):learning rate 减小时要乘的因子,new_lr=lr*factor
        - patience(int):如果连续patience个epchs没有提升那么lr就会下降.Default:1
        - threshold(float):测量新最佳值的阈值,default:1e-4
        - threshold_mode(bool):"rel" or "abs".
        - threshold_mode(bool):在'rel' mode dynamic_threshold=best*(1-threshold).
        - threshold_mode(bool):在'abs' mode dynamic_threshold=best-threshold.
        - cooldown(int):lr降低后恢复正常运行前等待的epoch数.default:0
        - min_lr(float or list):标量或标量列表,对于所有参数的下限，或者每个参数组各自的下限.default:0
        - eps(float):lr最小的下降,如果新的lr与原先的lr相差小于eps,则忽略不计,default:1e-8

    Example::

        lr_scheduler=dict(
            by_epoch=False,
            warmup='linear',
            warmup_iters=1,
            warmup_ratio=0.001,
            warmup_by_epoch=False,
            policy='valid',
            gamma=0.9,
        ),

    """

    def __init__(
        self,
        gamma,
        key="loss",
        patience=0,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
        **kwargs,
    ):
        self.key = key
        if gamma >= 1.0:
            raise ValueError("gamma should be < 1.0.")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        self.update_count = 0
        self.gamma = gamma
        self.min_lr = min_lr
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = inf
        self.num_bad_epochs = 0
        self.eps = eps
        self.valid_lr = 0
        self.last_loss = None
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        """Get the new learning rate"""
        cur_loss = runner.valid_log_buffer.get_value(self.key)
        if cur_loss not in (None, 0.0) and self.last_loss != cur_loss:
            if self.is_better(cur_loss, self.best):
                self.best = cur_loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
            self.last_loss = cur_loss

        # The current lr
        self.valid_lr = max(base_lr * self.gamma**self.update_count, self.min_lr)
        if self.num_bad_epochs > self.patience:
            self.update_count += 1
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            new_lr = max(base_lr * self.gamma**self.update_count, self.min_lr)
            if self.valid_lr - new_lr > self.eps:
                self.valid_lr = new_lr
        return self.valid_lr

    def is_better(self, a, best):
        """check the best loss"""
        if self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        return a < best - self.threshold

    def state_dict(self):
        """state dict."""
        return {
            "key": self.key,
            "valid_lr": self.valid_lr,
            "update_count": self.update_count,
            "gamma": self.gamma,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "cooldown_counter": self.cooldown_counter,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "min_lr": self.min_lr,
            "eps": self.eps,
        }


def annealing_cos(start, end, factor):
    """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * (start - end) * cos_out


@HOOKS.register_module()
class PytorchLrUpdateHook(Hook):
    """套一个pytorch自带的scheduler

    .. note::
        | 这是唯一一个不是继承LrUpdater的LR,主要功能是留下了一个调用pytorch中的lr的接口
        | 主要是通过policy来选用掉用何种lr,下面提供了一个表格
        | 你可以参考文档 https://pytorch.org/docs/stable/optim.html

    ===================================
      policy:
    ===================================
    - CosineAnnealingLR
    - CosineAnnealingWarmRestarts
    - CyclicLr
    - ExponentialLR
    - LambdalLR
    - MultiplicativeLR
    - OneCycleLR
    - StepLR
    - _LRScheduler

    ===================================

    Args:
        - by_epoch(float):最小的lr值
        - scheduler:其余参数按照pytorch设置

    Example::

        lr_scheduler=dict(
            by_epoch=True,
            policy='StepLR',  # use torch lr_scheduler
            gamma=0.5,
            step_size=5
        )
    """

    def __init__(self, by_epoch, lr_scheduler):
        """init."""
        self.by_epoch = by_epoch
        self.lr_scheduler = lr_scheduler

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.lr_scheduler.step()

    def after_train_iter(self, runner):
        if self.by_epoch:
            return
        cls_name = self.lr_scheduler.__class__.__name__
        if cls_name in ("ReduceLROnPlateau",):
            return
        self.lr_scheduler.step()

    def after_val_epoch(self, runner):
        cls_name = self.lr_scheduler.__class__.__name__
        if cls_name in ("ReduceLROnPlateau",):
            cur_loss = runner.valid_log_buffer.get_value("loss")
            self.lr_scheduler.step(cur_loss)


@HOOKS.register_module()
class NoamLrUpdaterHook(LrUpdaterHook):
    """ValidLrUpdaterHook.

    .. note::
        | NoamLrUpdaterHook for transformer
        | 分为三个阶段 一开始 step / self.warmup_start  然后到1,最后为(self.warmup_end / step) ** 0.5


    Args:
        - warmup_start(int):代表着线性增长的总步数 default=1000
        - warmup_end(int):代表着下降开始 default=10000

    """

    def __init__(
        self,
        warmup_start=1000,
        warmup_end=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

    def get_lr(self, runner, base_lr):
        """Noam scheme learning rate decay"""
        step = float(runner.iter + 1)
        return base_lr * torch.minimum(
            torch.minimum(torch.tensor(step / self.warmup_start), torch.tensor(1)),
            torch.tensor((self.warmup_end / step) ** 0.5),
        )


@HOOKS.register_module()
class TransformerLrUpdaterHook(LrUpdaterHook):
    """TransformerLrUpdaterHook.

    .. note::
        | TransformerLrUpdaterHook for transformer
        | See: https://arxiv.org/abs/1706.03762

    Args:
        - d_model(int): Transformer/Conformer 的维度
        - warmup_steps(int): warmup的步数

    """

    def __init__(
        self,
        d_model=512,
        warmup_steps=4000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self.warmup is not None:
            raise ValueError(
                "The warmup step shall be set with warmup_steps in TransformerLrUpdaterHook."
            )
        self.warmup_steps = warmup_steps
        self.d_model = d_model

    def get_lr(self, runner, base_lr):
        """Noam scheme learning rate decay"""
        step = float(runner.iter + 1)
        lr = (
            base_lr
            * self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        return lr


LR_SCHEDULER = Registry("lr_scheduler")


def register_torch_lr_schedulers():
    """register torch lr scheduler"""
    torch_lr_schedulers = []
    base_lr_scheduler = getattr(torch.optim.lr_scheduler, "_LRScheduler")
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith("__"):
            continue
        _lr_scheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_lr_scheduler) and issubclass(_lr_scheduler, base_lr_scheduler):
            LR_SCHEDULER.register_module()(_lr_scheduler)
            torch_lr_schedulers.append(module_name)
    return torch_lr_schedulers


TORCH_LR_SCHEDULER = register_torch_lr_schedulers()
