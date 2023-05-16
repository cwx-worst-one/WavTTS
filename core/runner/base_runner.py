''' Base Runner. '''
# pylint:disable=too-many-lines
import os
import os.path as osp
import time
import copy
import random
import pickle
from abc import ABCMeta, abstractmethod
import types
from collections import OrderedDict
from packaging import version
import numpy as np
import torch

from core.dataset import (
    get_meta,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.runner.hooks import (
    HOOKS,
    LR_SCHEDULER,
    Hook,
    FlopsHook,
    ProfilerHook,
    DebugHook,
    NanDebugHook,
    MemMonitorHook,
    LoggingHook,
)
from core.runner.optimizer import (
    DefaultOptimizerConstructor,
    dist_broadcast_optimizer,
    LayerwiseOptimizerConstructor,
)
from core.solutions import setup_solution
from core.utils import (
    Config,
    get_dist_info,
    mkdir_or_exist,
    symlink,
    dist_allreduce,
    dist_broadcast,
    build_from_cfg,
    Registry,
    hdfs_get,
    dist_hdfs_get,
    dist_broadcast_model,
    dist_barrier,
    logging,
    hdfs_put,
    hdfs_mkdir,
    get_rank,
    ReduceOp,
    load_checkpoint,
    save_checkpoint,
    weights_to_cpu,
)
from core.utils.profile.mem import invoke_memory_profile
from core.extensions import zero_grad_, clear_cuda_error, dist_parallel, initialize_topology, mpu
from core.slim.compressor import DolphinCompressor
from core.slim.distiller import DistillerManager
from .priority import get_priority
from .utils import format_file_list, get_time_str, get_time
from .metric import MetricLogger, FalconMetricLogger, TensorBoardLogger
from .optimizer import OPTIMIZERS


RUNNERS = Registry('runner')


class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

    Args:
        model (:obj:`torch.nn.Module`): The model to be run,
            which responsible for implements train_step,
            and val_step.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
    """

    # pylint: disable=too-many-public-methods

    def __init__(self, cfg, inference=False, export_onnx=False):
        '''init.'''
        self.cfg = cfg
        self.dataset_cfg = cfg.data
        self.train_cfg = cfg.train
        self.solution_cfg = cfg.solution
        self.lr_cfg = self.train_cfg.lr_scheduler
        self.val_cfg = cfg.valid
        self.optimizer_cfg = cfg.optimizer
        self.opt_util_cfg = cfg.optimizer_config
        self.pipeline_cfg = self.solution_cfg.get('pipeline', None)
        self.tensor_parallel_size = self.solution_cfg.get('tensor_parallel_size', 1)
        self._rank, self._world_size = get_dist_info()
        if self.tensor_parallel_size > 1:
            # initialize tensor parallel global variable
            tp_cfg = {
                'num_stages': 1,
                'model_parallel_size': self.tensor_parallel_size,
            }
            initialize_topology(tp_cfg)
        self.dp_rank = (
            self._rank if self.tensor_parallel_size == 1 else mpu.get_data_parallel_rank()
        )
        # path to best.pth
        self.best_pth = None
        if cfg.optimizer_config.get('bmuf_config', None) and self.world_size > 1:
            self.use_bmuf = True
            logging.info("BMUF optimizer used for distributed training mode!")
        else:
            self.use_bmuf = False
        self.work_dir = cfg.get('work_dir', './')
        mkdir_or_exist(self.work_dir)

        self.timestamp = get_time_str()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._val_iter = 0
        self._max_epochs = cfg.train.max_epochs
        self._max_iters = cfg.train.max_iters
        self._best_metric = None
        self.meta_data = None
        self.valid_data_loader = None
        self.train_data_loader = None
        self.falcon_report = cfg.get('falcon_report', 0)
        self.report_key_list = ['loss', 'nll_loss', 'cer', 'acc', 'time', 'data_time', 'fps', 'tps']
        self.args = cfg
        self.flush_steps = cfg.log_config.get('flush_steps', 530)
        self.build_metrics()
        self.build_metric_loggers(cfg.log_config.interval)
        self.is_inference = inference
        self.is_export_onnx = export_onnx
        cfg.setdefault('lm', Config())
        self.onnx_quantization = self.solution_cfg.get('onnx_quantization', False) or cfg.lm.get(
            'onnx_quantization', False
        )
        self.need_build_data_loader = (
            not self.is_inference
            and not (self.is_export_onnx and not self.onnx_quantization)
            or 'slim_config' in self.solution_cfg
            or 'slim_init_config' in self.solution_cfg
        )
        self.parallel_load = self.train_cfg.get('parallel_load', False)
        self.setup_torch_state()
        if self.need_build_data_loader:
            self.get_data_list(self.dataset_cfg)
        self.build_dataset(self.dataset_cfg)
        self.solution_cfg_distill = copy.deepcopy(self.solution_cfg)
        self.build_solution()

        self.loss = None
        self.solution_out = None
        self.dist_handler = None
        self.compressor = None

        # move solution to GPU
        self.solution.cuda()

        # setup auto parallel and distribution
        self.setup_dist_model()

        self.build_lr_scheduler()

        # Register Hooks
        self.register_training_hooks(
            lr_config=cfg.train.lr_scheduler,
            checkpoint_config=cfg.train.checkpoint_config,
        )

        # Resume from checkpoint during runner initialization
        self.resume()

        # wait reset epoch and data_count by self.resume
        self.start_data_loader()

        # move to cuda once again
        self.solution.cuda()

        # Broadcast model state dict from rank 0
        self.optimizer.zero_grad(set_to_none=True)

        if self.tensor_parallel_size > 1:
            # broadcast model and optimizer state in tp format
            tp_group = mpu.get_data_parallel_group()
            tp_root = mpu.get_data_parallel_src_id()
            dist_broadcast_model(
                self.solution,
                root_rank=0,
                tp_root=tp_root,
                tp_group=tp_group,
            )
            dist_broadcast_optimizer(
                self.solution,
                self.optimizer,
                root_rank=0,
                tp_root=tp_root,
                tp_group=tp_group,
            )
        elif not self.pipeline_cfg:
            # Broadcast model state dict from rank 0
            dist_broadcast_model(self.solution, root_rank=0)
            dist_broadcast_optimizer(self.solution, self.optimizer, root_rank=0)

        # init cuda for distributed before train iteration to avoid memory insufficient.
        dist_allreduce(torch.ones(2, device='cuda'), name='init_cuda')

        logging.info("rank %d in world_size %d in runner", self.rank, self.world_size)

    def setup_dist_model(self):
        '''setup dist model.'''
        if self.pipeline_cfg is not None:
            parallel_mode = 'pipeline'
        elif self.use_bmuf:
            parallel_mode = (
                'bmufddp'
                if self.args.optimizer_config.bmuf_config.get('within_ddp', False)
                else 'bmuf'
            )
            bmuf_cfg = self.opt_util_cfg.get('bmuf_config', None)
            bmuf_type = bmuf_cfg.pop('type', 'FusedBMUF')
            bmuf_cfg.setdefault('iters', self.iter)
            if self.train_cfg.get("use_dolphin_bmuf", True):
                bmuf_cfg.setdefault('bmuf_cls', OPTIMIZERS.get(bmuf_type))
        elif self.world_size > 1:
            parallel_mode = 'ddp'
        else:
            parallel_mode = 'none'

        enable_oss = self.opt_util_cfg.get('enable_oss', False) and parallel_mode != 'none'

        if not enable_oss:
            self.build_optimizer()
        self.build_compressor()
        # grad_clip after accumulation
        grad_clip_after = self.opt_util_cfg.get('grad_clip_mode', 'accum_after') != 'accum_before'
        self.solution, self.optimizer, self.dist_handler = dist_parallel(
            model=self.solution,
            optimizer_constructor=self._optimizer_constructor if enable_oss else self.optimizer,
            parallel_mode=parallel_mode,
            oss_config=self.opt_util_cfg,
            bmuf_config=bmuf_cfg if self.use_bmuf else None,
            ddp_config=self.train_cfg,
            amp_config=self.train_cfg,
            pipe_config=self.pipeline_cfg,
            grad_clip_after=grad_clip_after,  # default is True
            tp_size=self.tensor_parallel_size,
        )

    def setup_torch_state(self):
        '''setup rand seed.'''
        # set random seed
        seed = self.args.get('seed', 123356)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # deterministic mostly for conv, and may slowdown the training.
        self.solution_cfg.deterministic = self.train_cfg.get('deterministic', False)
        if self.train_cfg.get('deterministic', False):
            torch.backends.cudnn.deterministic = True

        # set cufft plan cache size
        max_cufft_plan_cache = self.train_cfg.get('max_cufft_plan_cache', 512)
        torch.backends.cuda.cufft_plan_cache.max_size = max_cufft_plan_cache

    def build_metrics(self):
        '''build metric func.'''
        self.train_log_buffer = None
        self.valid_log_buffer = None

    def build_metric_loggers(self, interval):
        '''build logger'''
        self.metric_loggers = [MetricLogger(interval, rank=self.rank)]
        self.metric_loggers.append(
            TensorBoardLogger(interval, rank=self.rank, flush_steps=self.flush_steps)
        )
        if self.falcon_report or self.args.get('zeus'):
            self.metric_loggers.append(
                FalconMetricLogger(self.args.log_config.interval, rank=self.rank)
            )
        self.log_buffer_by_dataset = {}

    def build_compressor(self):
        '''compress the solution and build teacher model if need'''
        if 'slim_config' in self.solution_cfg or 'slim_init_config' in self.solution_cfg:
            if self.train_data_loader is not None:
                self.train_data_loader.reset()
            self.compressor = DolphinCompressor(self.solution, self.cfg)
            local_path = self.compressor.init()
            self.compressor.compress(offline_dataloader=self.train_data_loader)
            self.compressor.post_process(self.optimizer)
            if self.solution_cfg.get('use_distiller', None):
                self.distiller = DistillerManager(
                    self.solution, self.solution_cfg_distill, local_path
                )

    def log_metric(self, message, name=None):
        '''log metric, it is usually used to output log information
        Args:
            message: Metrics or dict.
            name: if name not None, output "Epoch(val: {name})", else output Epoch(val)
        '''
        runner_dict = OrderedDict()
        runner_dict['mode'] = self.mode
        runner_dict['name'] = name
        # NOTE runner_dict['epoch'] and runner_dict['iter'] start at 1
        runner_dict['epoch'] = self.epoch + 1
        runner_dict['iter'] = self.iter + 1
        if self.mode == 'train':
            try:
                lr = sorted(list(set(self.current_lr())))
                if len(lr) == 1 or len(lr) > 3:
                    runner_dict['lr'] = lr[0]
                else:
                    runner_dict['lr'] = lr
            except:
                runner_dict['lr'] = self.current_lr()[0]
            runner_dict['remain_steps'] = self.max_iters - self.iter - 1
        else:
            # only record validation metric
            dataset_name = name if name is not None else self.mode
            self.log_buffer_by_dataset[dataset_name] = message.get(copy=True)
        for metric_logger in self.metric_loggers:
            metric_logger.log(runner_dict, message)

    def report_metric(self):
        '''
        log report to falcon. it is usually used to report the final information,
        such as the final cer information. if you report the same key more than once,
        only the last one will be saved
        '''
        # pylint: disable=no-member
        log_buffer_list = {"train": self.train_log_buffer.get()}
        log_buffer_list.update(self.log_buffer_by_dataset)

        if self.falcon_report or self.args.get('zeus'):
            message = {"iter": self.iter, "best.pth": self.best_pth}
            for name, log_buffer in log_buffer_list.items():
                sub_msg = dict()
                for key in self.report_key_list:
                    sub_msg[key] = log_buffer.get(key, 0)
                message[name] = sub_msg
            for metric_logger in self.metric_loggers:
                metric_logger.report(message)

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def val_iter(self):
        """int: val iter"""
        return self._val_iter

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @property
    def best_metric(self):
        """best_metric"""
        return self._best_metric

    @get_time('time')
    def train_iteration(self):
        '''train iteration.'''
        if hasattr(self.solution, 'set_num_updates'):
            self.solution.set_num_updates(self.iter)
        i = 0
        while i < self.grad_accum_step:
            batch_data = self.next_train_batch()
            try:
                self.solution.train()
                batch_data['loss_scale'] = self.dist_handler.get_scale(scaler_idx=0)
                solution_out = self.solution(batch_data)
                self.loss = solution_out['backward_loss'] / self.grad_accum_step

                if self.compressor is not None:
                    slim_loss = self.compressor.get_slim_loss(global_step=self.iter)
                    self.loss += slim_loss / self.grad_accum_step
                    solution_out['slim_loss'] = slim_loss

                if self.solution_cfg.get('use_distiller', None):
                    distiller_loss = self.distiller.caculate_distill_loss()
                    self.loss += distiller_loss
                    solution_out['distiller_loss'] = distiller_loss
                self.dist_handler.backward(self.loss, unscale=(i + 1 == self.grad_accum_step))

            except RuntimeError as e:
                self.handle_error(e, batch_data)
                continue
            self.train_log_buffer.update(solution_out)
            i = i + 1
        if self.opt_util_cfg.grad_clip:
            gnorm = self.clip_grads()
            self.train_log_buffer.update({'gnorm': gnorm})
        self.dist_handler.step(iters=self.iter)  # Pass in iter because of bmuf step

    def train(self):
        '''train func.'''
        self.before_train()
        while self.iter < self.args.train.max_iters:
            # train step begin
            self.call_hook('before_train_iter')
            self.train_iteration()
            self.call_hook('after_train_iter')
            self.log_metric(self.train_log_buffer)
            if hasattr(self.args.valid, 'interval'):
                if (self.iter + 1) % self.args.valid.interval == 0:
                    self.validation()
            elif hasattr(self.args.train, 'iters_per_epoch'):
                if (self.iter + 1) % self.args.train.iters_per_epoch == 0:
                    self.validation()
            if (
                hasattr(self.args.train, 'iters_per_epoch')
                and self._inner_iter + 1 == self.args.train.iters_per_epoch
            ):
                self.after_train_epoch()
                # epoch ending or data iter error
                self.train_data_loader.reset()
                if self._epoch >= self.args.train.max_epochs:
                    break
                self.call_hook('before_train_epoch')
                # Check current learning rate
                max_lr = max(param_group['lr'] for param_group in self.optimizer.param_groups)
                if max_lr < self.args.train.get('final_lr', 1e-7):
                    break
                self._inner_iter = 0
            else:
                self._inner_iter += 1
            # check early stop every checkpoint
            if (
                self.compressor is not None
                and hasattr(self.train_cfg.checkpoint_config, 'interval')
                and (self.iter + 1) % self.train_cfg.checkpoint_config.interval == 0
                and self.compressor.check_early_stop()
            ):
                break
            self._iter += 1
        self.after_train()

    def after_validation(self):
        '''after validation.'''
        self.call_hook('after_val_epoch')
        self.mode = 'train'
        self.solution.train()

    def next_valid_batch(self):
        '''next valid batch data'''
        batch_data = self.valid_data_loader.next()
        if (
            batch_data is not None
            and not mpu.is_unitialized()
            and mpu.get_model_parallel_world_size() > 1
        ):
            # broadcast batch data in tp group
            batch_data = self.broadcast_batch(batch_data)
        return batch_data

    @torch.no_grad()
    def validation(self):
        '''validation func.'''
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        if self.solution_cfg.get('valid_multi_cer', False):
            dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
        else:
            dataset_names = [None]
        for dataset_name in dataset_names:
            self.valid_log_buffer.reset()
            start_time = time.time()
            batch_data = self.next_valid_batch()
            self._val_iter = 0
            while batch_data is not None:
                self.call_hook('before_val_iter')
                try:
                    self.solution.eval()
                    validation_out = self.solution(batch_data)
                    self._val_iter += 1
                    batch_data = self.next_valid_batch()
                    self.call_hook('after_val_iter')
                    self.valid_log_buffer.update(validation_out)
                except RuntimeError as e:
                    clear_cuda_error()
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            end_time = time.time()
            self.valid_log_buffer.update({'time': end_time - start_time})
            start_time = time.time()
            self.log_metric(self.valid_log_buffer, name=dataset_name)
        self.after_validation()

    @abstractmethod
    def inference(self):
        '''inference func.'''

    def run(self):
        '''runner's entry.'''
        self.call_hook('before_run')
        if self.is_export_onnx:
            self.export_onnx()
        elif self.is_inference:
            self.inference()
        else:
            self.train()
            self.train_data_loader.terminate()
            self.valid_data_loader.terminate()
        self.call_hook('after_run')
        self.report_metric()
        for metric_logger in self.metric_loggers:
            metric_logger.close()

    def update_best_metric(self):
        '''
        update best metric to current metric if current is better.

        Return:
            True: current metric is better.
            False: otherwise.
        '''
        best_metric_name = self.train_cfg.checkpoint_config.get('best_metric_name', 'loss')
        metric_value = self.valid_log_buffer.get_value(best_metric_name)
        if metric_value == 0.0:
            logging.error("rank %d catch cur loss is 0.0, not update best metric", self.rank)
            return False
        best_metric_type = self.train_cfg.checkpoint_config.get('best_metric_type', 'min')
        if best_metric_type == 'max':
            if self._best_metric is None or metric_value >= self._best_metric:
                self._best_metric = metric_value
                return True
        else:
            if self._best_metric is None or metric_value <= self._best_metric:
                self._best_metric = metric_value
                return True
        return False

    def get_state_dict(self):
        """Get state dictionary"""
        # get meta information
        meta = {
            'epoch': self.epoch,
            'iter': self.iter,
            'inner_data_count': 0,
        }
        # update inner_utt
        if self.train_data_loader:
            dataloader_state_dict = self.train_data_loader.state_dict()
            meta.update(dataloader_state_dict)
        if self._best_metric is not None:
            meta['best'] = self._best_metric

        model = self.solution

        state_dict = {
            'meta': meta,
            'model': weights_to_cpu(model.state_dict()) if self.dp_rank == 0 else None,
        }
        # get lr scheduler state_dict
        if self.lr_scheduler is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        # get dist handler state_dict
        if self.dist_handler is not None:
            state_dict['parallel_handler'] = self.dist_handler.state_dict()
        return state_dict

    def save_checkpoint(self, out_dir, filename_tmpl='step_{}.pth', create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """

        filename = filename_tmpl.format(self._iter + 1)
        if self.tensor_parallel_size > 1:
            filename = "step_{}_mp_{}.pth".format(self._iter + 1, mpu.get_model_parallel_rank())
        filepath = os.path.join(out_dir, filename)
        state_dict = self.get_state_dict()
        if self.dp_rank == 0:
            save_checkpoint(filepath, state_dict)

        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink and self.rank == 0 and self.tensor_parallel_size == 1:
            symlink(filename, os.path.join(out_dir, 'latest.pth'))

        return filename

    def pre_build_dataset(self, dataset_cfg):
        '''prepare for build dataset.'''
        if dataset_cfg.get('meta_file'):
            meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
            if isinstance(meta_data_root, (list, tuple)):
                meta_data_root = meta_data_root[0]
            meta_file = osp.join(meta_data_root, dataset_cfg.meta_file)
            self.meta_data = get_meta(meta_file)
        else:
            self.meta_data = {}
        self.solution_cfg.setdefault('fbank_dim', dataset_cfg.get("fbank_dim"))
        self.solution_cfg.setdefault('use_eos', dataset_cfg.get("use_eos"))
        if self.meta_data:
            self.solution_cfg.setdefault('tgt_dict', self.meta_data.get('tgt_dict'))
        self.solution_cfg.setdefault('iters_per_epoch', self.args.train.get("iters_per_epoch"))

    def build_dataset(self, dataset_cfg):
        '''build dataset.'''
        self.pre_build_dataset(dataset_cfg)
        valid_item_trans_cfg = dataset_cfg.get("valid_item_transform", [])
        self.valid_item_trans = build_item_augmentation(valid_item_trans_cfg, self.meta_data)
        train_item_trans_cfg = dataset_cfg.get("train_item_transform", [])
        self.train_item_trans = build_item_augmentation(train_item_trans_cfg, self.meta_data)
        draw_batch_cfg = dataset_cfg.get("batch_transform", [])
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)
        draw_train_batch_cfg = dataset_cfg.get('train_batch_transform', [])
        self.draw_train_batch_fn = build_draw_batch_fn(draw_train_batch_cfg, self.meta_data)
        draw_valid_batch_cfg = dataset_cfg.get("valid_batch_transform", [])
        self.draw_valid_batch_fn = build_draw_batch_fn(draw_valid_batch_cfg, self.meta_data)
        train_device_trans_cfg = dataset_cfg.get('train_device_transform', [])
        self.train_device_trans = build_device_augmentation(train_device_trans_cfg, self.meta_data)
        valid_device_trans_cfg = dataset_cfg.get('valid_device_transform', [])
        self.valid_device_trans = build_device_augmentation(valid_device_trans_cfg, self.meta_data)

        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            self.val_bucket_schedule = dataset_cfg.bucket_schedule_val
        elif hasattr(dataset_cfg, 'bucket_schedule'):
            self.val_bucket_schedule = dataset_cfg.bucket_schedule

    def start_data_loader(self):
        '''post build dataset.'''
        if not self.need_build_data_loader:
            return
        self.train_data_loader.reset()
        self.valid_data_loader.reset()

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                '%s - lr is not applicable because optimizer does not exist.'
                % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            )
        return [group['lr'] for group in self.optimizer.param_groups]

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list: Current momentum of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError('momentum is not applicable because optimizer does not exist.')
        momentums = []
        for group in self.optimizer.param_groups:
            if 'momentum' in group.keys():
                momentums.append(group['momentum'])
            elif 'betas' in group.keys():
                momentums.append(group['betas'][0])
            else:
                momentums.append(0)
        return momentums

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :cls:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def _resume_local(
        self,
        checkpoint_dir,
        resume_file,
        resume_optimizer=True,
        resume_progress=True,
        resume_lr_scheduler=True,
        resume_amp=True,
        map_location='default',
    ):
        '''resume.'''
        # pylint:disable=too-many-branches
        if not resume_file:
            return False
        checkpoint_files = []
        for fl in resume_file.split(','):
            checkpoint_file = osp.join(checkpoint_dir, fl)
            if not osp.exists(checkpoint_file):
                return False
            checkpoint_files.append(checkpoint_file)

        if map_location == 'default':
            if len(checkpoint_files) > 1:
                map_location = 'cpu'
            else:
                device_id = torch.cuda.current_device()
                map_location = f'cuda:{device_id}'

        checkpoint = load_checkpoint(
            self.solution,
            checkpoint_files,
            map_location=map_location,
        )

        if resume_progress:
            self._epoch = checkpoint['meta']['epoch']
            skip_data_num = checkpoint['meta'].get('inner_data_count', 0)
            if self.train_data_loader is not None:
                self.train_data_loader.reset_epoch_count(self._epoch, skip_data_num)
            self._iter = checkpoint['meta']['iter'] + 1
            if not self.is_inference and not self.is_export_onnx and self._max_iters <= self._iter:
                raise ValueError(
                    'The current iter is greater or equal to max iter after resume checkpoint'
                )
            if 'best' in checkpoint['meta']:
                self._best_metric = checkpoint['meta']['best']
        if 'optimizer' in checkpoint and resume_optimizer and self.dist_handler is None:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint and resume_lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if self.dist_handler and 'parallel_handler' not in checkpoint:
            # original dolphin checkpoint
            checkpoint['parallel_handler'] = dict()
            if resume_amp and 'amp' in checkpoint:
                checkpoint['parallel_handler']['amp'] = checkpoint['amp']
            if resume_optimizer and 'optimizer' in checkpoint:
                checkpoint['parallel_handler']['optimizer'] = checkpoint['optimizer']

        if 'parallel_handler' in checkpoint and self.dist_handler:
            self.dist_handler.load_state_dict(
                checkpoint['parallel_handler'],
                resume_amp=resume_amp,
                resume_optimizer=resume_optimizer,
                resume_iters=self._iter,
            )

        logging.info('resumed epoch %d, iter %d', self.epoch, self.iter)
        return True

    def _resume_from_hdfs_path(
        self,
        hdfs_path,
        checkpoint_dir,
        resume_optimizer=True,
        resume_progress=True,
        resume_lr_scheduler=True,
        resume_amp=True,
    ):
        '''try resume from ${resume_hdfs_chkpt}'''
        if not hdfs_path:
            return False

        hdfs_components = hdfs_path.split('|')
        if len(hdfs_components) > 1:
            hdfs_root = hdfs_components[0]
            hdfs_ckpts = []
            for fl in hdfs_components[1].split(','):
                hdfs_ckpts.append(osp.join(hdfs_root, fl))
        else:
            hdfs_ckpts = hdfs_components[0].split(',')
        # downloaded hdfs checkpoints
        local_files = []
        for i, hdfs_file in enumerate(hdfs_ckpts):
            if not self.parallel_load:
                local_file = 'resume_ckpt-{}.pth'.format(i)
                dist_hdfs_get(hdfs_file, checkpoint_dir, local_file)
            else:
                local_file = 'resume_ckpt_mp_{}-{}.pth'.format(mpu.get_model_parallel_rank(), i)
                if self.dp_rank == 0:
                    mkdir_or_exist(checkpoint_dir)
                    hdfs_get(hdfs_file, osp.join(checkpoint_dir, local_file))
                dist_barrier()
            local_files.append(local_file)

        if self._resume_local(
            checkpoint_dir,
            ','.join(local_files),
            resume_optimizer,
            resume_progress,
            resume_lr_scheduler,
            resume_amp,
        ):
            logging.all_rank_info('rank %d: resume checkpoint on hdfs: %s', self.rank, hdfs_path)
            return True
        return False

    def _resume_from_last_trial(
        self,
        checkpoint_dir,
        resume_file,
        resume_optimizer=True,
        resume_progress=True,
        resume_lr_scheduler=True,
        resume_amp=True,
    ):
        '''try resume from ${remote_save_root}/${out_dir}/${resume_file}'''
        remote_save_root = self.train_cfg.get('remote_save_root', None)
        if not remote_save_root or not resume_file:
            return False
        hdfs_path = '{}|{}'.format(
            osp.join(
                remote_save_root, self.train_cfg.save_dir, self.train_cfg.save_name, 'checkpoints'
            ),
            resume_file,
        )
        if self._resume_from_hdfs_path(
            hdfs_path,
            checkpoint_dir,
            resume_optimizer,
            resume_progress,
            resume_lr_scheduler,
            resume_amp,
        ):
            logging.info('rank %d: resume checkpoint from last trial', self.rank)
            return True
        return False

    def _resume_from_pretrain(
        self,
        pretrain_chkpt,
        checkpoint_dir,
    ):
        '''try resume from ${resume_pretrain_chkpt}'''
        if not pretrain_chkpt:
            return False
        if self._resume_from_hdfs_path(
            pretrain_chkpt,
            checkpoint_dir,
            resume_optimizer=False,
            resume_progress=False,
            resume_lr_scheduler=False,
            resume_amp=False,
        ):
            logging.info('rank %d: resume from pretrain checkpoint', self.rank)
            return True
        return False

    @staticmethod
    def convert_chkpt_name(chkpt_name=None):
        '''convert chkpt name'''
        if chkpt_name is None:
            return chkpt_name
        step = chkpt_name.split('_')[-1].replace('.pth', '')
        if 'step_{}.pth'.format(step) not in chkpt_name:
            return chkpt_name
        chkpt_name = chkpt_name.replace(
            'step_{}.pth'.format(step),
            'step_{}_mp_{}.pth'.format(step, mpu.get_model_parallel_rank()),
        )
        return chkpt_name

    def resume(self):
        '''do checkpoint resume.'''
        resume_file = self.train_cfg.get('resume', None)
        hdfs_path = self.train_cfg.get('resume_hdfs_chkpt', None)
        pretrain_chkpt = self.train_cfg.get('resume_pretrain_chkpt', None)
        resume_optimizer = self.train_cfg.get('resume_optimizer', True)
        resume_progress = self.train_cfg.get('resume_progress', True)
        resume_lr_scheduler = self.train_cfg.get('resume_lr_scheduler', True)
        resume_amp = self.train_cfg.get('resume_amp', True) and not self.is_export_onnx
        if self.tensor_parallel_size > 1 and self.parallel_load:
            resume_file = self.convert_chkpt_name(resume_file)
            hdfs_path = self.convert_chkpt_name(hdfs_path)
            pretrain_chkpt = self.convert_chkpt_name(pretrain_chkpt)
        checkpoint_dir = osp.join(
            self.train_cfg.save_root,
            self.train_cfg.save_dir,
            self.train_cfg.save_name,
            'checkpoints',
        )

        if self._resume_local(
            checkpoint_dir,
            resume_file,
            resume_optimizer,
            resume_progress,
            resume_lr_scheduler,
            resume_amp,
        ):
            logging.info('rank %d: resume from local checkpoint', self.rank)
            return

        if self._resume_from_hdfs_path(
            hdfs_path,
            checkpoint_dir,
            resume_optimizer,
            resume_progress,
            resume_lr_scheduler,
            resume_amp,
        ):
            return
        if self._resume_from_last_trial(
            checkpoint_dir,
            resume_file,
            resume_optimizer,
            resume_progress,
            resume_lr_scheduler,
            resume_amp,
        ):
            return
        if self._resume_from_pretrain(pretrain_chkpt, checkpoint_dir):
            return
        # the model is resumed with solution.slim_init_config.resume_pretrain_ckpt
        if (
            'slim_init_config' in self.solution_cfg
            and 'resume_pretrain_ckpt' in self.solution_cfg.slim_init_config
        ):
            return
        if self.is_inference or self.is_export_onnx:
            raise ValueError('Failed to resume any checkpoint in inference or export mode. Exit!')

        logging.info('rank %d: failed to resume any checkpoint in train mode.', self.rank)

    def register_lr_hook(self, lr_config):
        '''register_lr_hook.'''
        if self.lr_scheduler is not None:
            pytorch_lr_config = dict()
            pytorch_lr_config['type'] = 'PytorchLrUpdateHook'
            pytorch_lr_config['lr_scheduler'] = self.lr_scheduler
            pytorch_lr_config['by_epoch'] = lr_config.get('by_epoch', False)
            hook = build_from_cfg(pytorch_lr_config, HOOKS)
            self.register_hook(hook)
            return
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater later.
            # Since this is not applicable for `CosineAnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)
        self.lr_scheduler = hook

    def register_momentum_hook(self, momentum_config):
        '''register_momentum_hook.'''
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for `CosineAnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook)

    def register_checkpoint_hook(self, checkpoint_config):
        '''register_checkpoint_hook.'''
        if checkpoint_config is None:
            return

        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            out_dir = os.path.join(
                self.train_cfg.save_root,
                self.train_cfg.save_dir,
                self.train_cfg.save_name,
                'checkpoints',
            )
            if "max_keep_ckpts" in checkpoint_config:
                checkpoint_config.setdefault(
                    "max_hdfs_keep_ckpts", checkpoint_config.get("max_keep_ckpts")
                )
            if "max_hdfs_keep_ckpts" in checkpoint_config:
                checkpoint_config.setdefault(
                    "max_local_keep_ckpts", checkpoint_config.get("max_hdfs_keep_ckpts")
                )
            if "max_keep_ckpts" in checkpoint_config:
                del checkpoint_config["max_keep_ckpts"]
            checkpoint_config.setdefault('out_dir', out_dir)
            hook = build_from_cfg(checkpoint_config, HOOKS)
        elif isinstance(checkpoint_config, Hook):
            hook = checkpoint_config
        else:
            logging.info(
                'rank %d: unsupported checkpoint config type: %s',
                self.rank,
                type(checkpoint_config),
            )
            return
        self.register_hook(hook)

    def register_training_hooks(self, lr_config, checkpoint_config=None, momentum_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - MomentumUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - LoggerHook(s)
        """
        if self.train_cfg.get('profile_mem', False):
            invoke_memory_profile(
                self.solution,
                self.args.get('project', 'Solution'),
                max_depth=self.train_cfg.get('profile_mem_depth', 10),
            )
        if self.train_cfg.get('debug_nan', False):
            self.register_hook(NanDebugHook(), 'VERY_LOW')
        if self.train_cfg.get('calc_flops', True):
            self.register_hook(FlopsHook(**self.train_cfg), 'VERY_LOW')
        if os.getenv('ARNOLD_PROFILER', None) == '2' or self.train_cfg.get('profile', False):
            self.register_hook(ProfilerHook(**self.train_cfg), 'VERY_LOW')
        if os.getenv('DOLPHIN_DEBUG_DEPTH', None) or self.train_cfg.get(
            'dolphin_debug_depth', False
        ):
            dolphin_debug_depth = self.train_cfg.get(
                'dolphin_debug_depth', int(os.getenv('DOLPHIN_DEBUG_DEPTH', '0'))
            )
            self.register_hook(DebugHook(dolphin_debug_depth))
        self.register_momentum_hook(momentum_config)
        self.register_lr_hook(lr_config)
        self.register_hook(MemMonitorHook(self.train_cfg.get('memory_interval', 1000)))
        self.register_hook(LoggingHook())
        # checkpoint hook must after logger hook,
        # it relies on loss or accuracy from log_buffer.
        self.register_checkpoint_hook(checkpoint_config)

    def build_solution(self):
        '''build_solution.'''
        solution_cfg = self.solution_cfg
        solution_cfg.setdefault('is_inference', self.is_inference)
        onnx_dir = os.path.join(
            self.train_cfg.save_root, self.train_cfg.save_dir, self.train_cfg.save_name, 'onnx'
        )
        solution_cfg.setdefault('onnx_dir', onnx_dir)

        self.solution = setup_solution(solution_cfg)

        param_numel = sum(p.numel() for p in self.solution.parameters() if p.requires_grad)
        logging.info("Model has %.2fM Parameters", (param_numel / 1024 / 1024))
        self.params = [(name, p) for name, p in self.solution.named_parameters() if p.requires_grad]

    def _optimizer_constructor(self, params=None, **_kwargs):
        '''optimizer constructor for auto parallel'''
        numel = sum(p.numel() for group in params for p in group['params'] if p.requires_grad)
        logging.all_rank_info("OSS has %.2fM Parameters" % (numel / (1e6 + 1e-3)))
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cfg['params'] = params
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)

    def build_optimizer(self):
        '''build_optimizer'''
        optimizer_cfg = self.optimizer_cfg
        if isinstance(optimizer_cfg.get('lr', 0), list) and isinstance(
            optimizer_cfg.get('lr', 0)[-1], float
        ):
            opt_builder = LayerwiseOptimizerConstructor(optimizer_cfg)
        else:
            opt_builder = DefaultOptimizerConstructor(optimizer_cfg)
        logging.info("Optimizer Config: {}".format(optimizer_cfg))
        self.optimizer = opt_builder(self.solution)
        if version.parse(torch.__version__) < version.parse('1.9'):
            self.optimizer.zero_grad = types.MethodType(zero_grad_, self.optimizer)

    def build_lr_scheduler(self):
        '''build lr scheduler'''
        self.lr_scheduler = None
        assert 'policy' in self.lr_cfg
        policy_type = self.lr_cfg.get('policy')
        if LR_SCHEDULER.get(policy_type) is None:
            return
        args = self.lr_cfg.copy()
        args.pop('policy')
        args.pop('by_epoch', None)
        args.setdefault('optimizer', self.optimizer)
        if policy_type == 'LambdaLR' and 'lr_lambda' in args:
            args.lr_lambda = eval(args.lr_lambda)
        args['type'] = policy_type
        self.lr_scheduler = build_from_cfg(args, LR_SCHEDULER)

    def export_onnx(self):
        '''export onnx'''
        rank = get_rank()
        if rank != 0:
            return
        out_dir = self.solution_cfg.get('onnx_dir')
        self.solution.eval()
        self.solution.register_infers()
        self.solution.export(data_loader=self.train_data_loader)
        self.save_onnx(out_dir)

    def save_onnx(self, local_dir):
        '''save onnx'''
        if self.rank != 0:
            return
        remote_save_root = self.train_cfg.get('remote_save_root', None)
        if remote_save_root:
            save_dir = osp.join(remote_save_root, self.train_cfg.save_dir, self.train_cfg.save_name)
            hdfs_mkdir(save_dir)
            hdfs_put(local_dir, save_dir)

    def clip_grads(self, params=None):
        '''clip_grads.'''
        grad_norm = self.dist_handler.clip_grad_norm(
            params=params,
            max_grad_clip=self.opt_util_cfg.max_grad_clip,
            **self.opt_util_cfg.grad_clip,
        )
        return grad_norm

    def before_train(self):
        '''before train'''
        self.mode = 'train'
        self.call_hook('before_epoch')
        self.grad_accum_step = self.train_cfg.get('grad_accum_step', 1)
        self.grad_accum_mode = self.train_cfg.get('grad_accum_mode', 'AVG')

        if (
            hasattr(self.train_cfg.lr_scheduler, 'by_epoch')
            and self.train_cfg.lr_scheduler.by_epoch
            and not self.use_bmuf
            and self.world_size > 1
        ):
            self.train_cfg.sync_epoch = True
            logging.info("DDP and by_epoch used together need to use sync_epoch")
        if self.solution_cfg.get('jointer_split_num', 1) > 1:
            # NOTE: backward is done in solution forward when splitting jointer.
            # Skip apex amp grad stash mechanism to enable dynamic loss scale.
            # pylint: disable=protected-access
            self.optimizer._prepare_amp_backward = types.MethodType(lambda opt: '', self.optimizer)
        if self.train_cfg.get('use_regularization', None):
            self.ckp_weight = self.get_model_params()

    def broadcast_batch(self, batch_data):
        '''broadcast batch data in tp group'''
        assert self.tensor_parallel_size > 1
        result_batch = {}
        pickleable_dict = {}
        tp_scr_id = (self._rank // self.tensor_parallel_size) * self.tensor_parallel_size
        for k, v in batch_data.items():
            if not torch.is_tensor(v):
                pickleable_dict[k] = v
                continue
            send_shape = torch.LongTensor(data=v.size()).to('cuda')
            dist_broadcast(send_shape, tp_scr_id, group=mpu.get_model_parallel_group())
            send_data = v
            if mpu.get_model_parallel_rank() != 0:
                send_data = torch.zeros(send_shape.tolist(), dtype=v.dtype, device=v.device)
            dist_broadcast(send_data, tp_scr_id, group=mpu.get_model_parallel_group())
            result_batch[k] = send_data
        msg = pickle.dumps(pickleable_dict)
        msg = torch.ByteTensor(torch.ByteStorage.from_buffer(msg)).cuda()
        length_tensor = torch.tensor([len(msg)], dtype=torch.long).cuda()
        dist_broadcast(length_tensor, root_rank=tp_scr_id, group=mpu.get_model_parallel_group())
        if mpu.get_model_parallel_rank() != 0:
            msg = torch.empty(length_tensor.item(), dtype=torch.uint8).cuda()
        dist_broadcast(msg, root_rank=tp_scr_id, group=mpu.get_model_parallel_group())
        result_batch.update(pickle.loads(msg.cpu().numpy().tobytes()))
        return result_batch

    @get_time('data_time')
    def next_train_batch(self):
        '''next train batch'''
        batch_data = self.train_data_loader.next()
        batch_data = self.dist_sync_epoch(batch_data)
        if batch_data is None:
            # epoch ending or dataiter error
            self.train_data_loader.reset()
            if (
                not self.train_cfg.get(
                    'sync_epoch', self.train_cfg.get('drop_when_epoch_end', False)
                )
                or self.world_size == 1
            ):
                self.after_train_epoch()
            self.call_hook('before_train_epoch')
            self._inner_iter = 0
            batch_data = self.train_data_loader.next()
            if batch_data is None:
                raise RuntimeError(
                    "%s - rank %d data loader error"
                    % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.rank)
                )
        if self.tensor_parallel_size > 1:
            # broadcast batch data in tp group
            batch_data = self.broadcast_batch(batch_data)
        return batch_data

    def handle_error(self, e, batch_data):
        '''handle error'''
        if 'NCCL communicator was aborted' in str(e):
            raise e
        # destroy auto grad graph through delete loss,
        # so the gpu memory could be free.
        clear_cuda_error()
        try:
            del self.solution_out, self.loss
        except Exception:
            pass
        torch.cuda.empty_cache()
        clear_cuda_error()
        self.optimizer.zero_grad(set_to_none=True)
        batch_info = ' '.join(
            [
                '{}:{}'.format(k, v.shape)
                for k, v in batch_data.items()
                if isinstance(v, (torch.Tensor, np.ndarray))
            ]
        )
        logging.error(
            "rank %d iter %d batch_info %r catch %s",
            self.rank,
            self._iter,
            batch_info,
            str(e),
            exc_info=True,
        )
        self.train_data_loader.set_oom_info()

    def after_train(self):
        '''after train'''
        self.call_hook('after_epoch')
        if self.dist_handler:
            self.dist_handler.clear()

    def dist_sync_epoch(self, batch):
        '''sync epoch info between multi rank.'''
        if self.world_size == 1:
            return batch
        if not self.train_cfg.get('sync_epoch', self.train_cfg.get('drop_when_epoch_end', False)):
            # sync_epoch is False and drop_when_epoch_end is False
            return batch
        if self.inner_iter < self.train_cfg.get('min_sync_iter', 0):
            return batch
        if not hasattr(self, 'end_of_epoch'):
            self.end_of_epoch = torch.ByteTensor([False]).cuda()
        if self.train_cfg.get("drop_when_epoch_end", False):
            # sync_epoch = True and drop_when_epoch_end = True
            self.end_of_epoch[0] = batch is None
        else:
            # Only rank 0 reach end, epoch be increased.
            self.end_of_epoch[0] = self.rank == 0 and batch is None
        dist_allreduce(self.end_of_epoch, name='sync_epoch', op=ReduceOp.SUM)
        if self.end_of_epoch:
            # Increase epoch if rank 0 reach end of epoch.
            self.after_train_epoch()
            if self.train_cfg.get('drop_when_epoch_end', False):
                while batch is not None:
                    # pylint:disable=not-callable
                    batch = self.train_data_loader.next()
        return batch

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transform config'''

    def get_data_list(self, dataset_cfg):
        '''
        get valid_file_list,train_file_list
        '''
        data_root = dataset_cfg.get("data_root", None)
        train_data_root = dataset_cfg.get("train_data_root", data_root)
        valid_data_root = dataset_cfg.get("valid_data_root", data_root)
        meta_data_root = dataset_cfg.get('meta_data_root', train_data_root)
        train_file_list = dataset_cfg.get("train_file_list", None)
        valid_file_list = dataset_cfg.get("valid_file_list", None)
        meta_file_list = dataset_cfg.get('meta_file', None)
        data_path_separator = dataset_cfg.get('data_path_separator', '')
        dataset_file_nums = []

        # multiple data inputs from the command cfg can be splited by data_path_separator
        if (
            data_path_separator != ''
            and not isinstance(train_data_root, list)
            and not isinstance(meta_file_list, list)
        ):
            train_data_root = train_data_root.split(data_path_separator)
            train_file_list = train_file_list.split(data_path_separator)
            meta_file_list = meta_file_list.split(data_path_separator)

        # get train_file_list and eval_file_list
        if isinstance(train_data_root, str):
            train_data_root = [train_data_root]
            train_file_list = [train_file_list]
        self.train_file_list = []
        assert len(train_data_root) == len(train_file_list)
        for train_root, train_file in zip(train_data_root, train_file_list):
            train_dataset_now = [osp.join(train_root, p) for p in eval(train_file)]
            self.train_file_list += train_dataset_now
            dataset_file_nums.append(len(train_dataset_now))

        dataset_cfg.dataset_length = dataset_file_nums
        logging.info(f"The training list: \n{format_file_list(self.train_file_list)}\n")

        # get valida_file_lists
        if isinstance(valid_data_root, str):
            valid_data_root = [valid_data_root]
            valid_file_list = [valid_file_list]
        self.valid_file_list = []
        assert len(valid_data_root) == len(valid_file_list)
        for valid_root, valid_file in zip(valid_data_root, valid_file_list):
            valid_dataset_now = [osp.join(valid_root, p) for p in eval(valid_file)]
            self.valid_file_list += valid_dataset_now
        logging.info(f"The valid list: \n{format_file_list(self.valid_file_list)}\n")

        # get meta_file_lists
        if isinstance(meta_data_root, str):
            meta_data_root = [meta_data_root]
        if isinstance(meta_file_list, str) or not meta_file_list:
            meta_file_list = [meta_file_list]
        self.meta_file_list = []
        for meta_root, meta_file in zip(meta_data_root, meta_file_list):
            if meta_file:
                self.meta_file_list.append(osp.join(meta_root, meta_file))
        logging.info(f"The meta list: \n{format_file_list(self.meta_file_list)}\n")

    def get_eval_list(self, eval_data_root, eval_file_list):
        '''
        to get eval_file_list
        '''
        if isinstance(eval_data_root, str):
            eval_data_root = [eval_data_root]
            eval_file_list = [eval_file_list]
        self.eval_file_list = []
        assert len(eval_data_root) == len(eval_file_list)
        for eval_root, eval_file in zip(eval_data_root, eval_file_list):
            eval_dataset_now = [os.path.join(eval_root, p) for p in eval(eval_file)]
            self.eval_file_list += eval_dataset_now

    def get_model_params(self):
        '''get model parameters'''
        ckp_weight = {
            n: p.data.clone().detach()
            for n, p in self.solution.named_parameters()
            if p.requires_grad
        }
        return ckp_weight

    def after_train_epoch(self):
        '''after train epoch'''
        self._epoch += 1
        self.call_hook('after_train_epoch')
        self.train_log_buffer.reset()
