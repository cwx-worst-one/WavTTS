''' base se runner. '''
from abc import ABCMeta
import os
import time
import torch

try:
    from byteslim.quant.post_quant.utils.get_loss import get_kurt_loss
except ImportError:
    # To use byteslim, include byteslim scm at task building stage.
    # For more detail, see dolphin tutorial.
    # An exception will be raised if byteslim is used.
    pass
from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.utils import logging
from core.runner.metric.se_metric import *
from core.utils.path import mkdir_or_exist
from core.utils import hdfs_mkdir, hdfs_put, dist_allreduce, ReduceOp
from core.extensions import clear_cuda_error
from ..base_runner import BaseRunner, RUNNERS
from ..utils import get_time


@RUNNERS.register_module()
class BaseSeModuleRunner(BaseRunner, metaclass=ABCMeta):
    '''Base SE Runner.'''

    def __init__(self, cfg, inference=False, export_onnx=False):
        super().__init__(cfg, inference, export_onnx)
        self.loss_total = 0

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        parse_inference_cfg = dataset_cfg.get('inference_item_transform', [])
        self.parse_fn_inference = build_item_augmentation(parse_inference_cfg, self.meta_data)
        draw_inference_batch_cfg = dataset_cfg.get('inference_batch_transform', [])
        self.draw_inference_batch_fn = build_draw_batch_fn(draw_inference_batch_cfg, self.meta_data)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        parse_cfg = dataset_cfg.get('train_item_transform', [])
        self.parse_fn = build_item_augmentation(parse_cfg, self.meta_data)
        parse_valid_cfg = dataset_cfg.get('valid_item_transform', [])
        self.parse_fn_valid = build_item_augmentation(parse_valid_cfg, self.meta_data)

        draw_batch_cfg = dataset_cfg.get('batch_transform', [])
        train_draw_batch_cfg = dataset_cfg.get('train_batch_transform', draw_batch_cfg)
        valid_draw_batch_cfg = dataset_cfg.get('valid_batch_transform', draw_batch_cfg)
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)
        self.train_draw_batch_fn = build_draw_batch_fn(train_draw_batch_cfg, self.meta_data)
        self.valid_draw_batch_fn = build_draw_batch_fn(valid_draw_batch_cfg, self.meta_data)

        device_trans_cfg = dataset_cfg.get('device_transform', [])  # device transform
        train_device_trans_cfg = dataset_cfg.get(
            'train_device_transform', device_trans_cfg
        )  # train device transform
        valid_device_trans_cfg = dataset_cfg.get(
            'valid_device_transform', device_trans_cfg
        )  # train device transform
        self.train_device_trans_fn = build_device_augmentation(
            train_device_trans_cfg, self.meta_data
        )
        self.valid_device_trans_fn = build_device_augmentation(
            valid_device_trans_cfg, self.meta_data
        )

        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule

        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.parse_fn,
            self.train_draw_batch_fn,
            self.train_device_trans_fn,
            shuffle=True,
        )

        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            val_bucket_schedule,
            dataset_cfg,
            self.parse_fn_valid,
            self.valid_draw_batch_fn,
            self.valid_device_trans_fn,
            split_path_list_by_rank=False,
        )
        if self.args.train.get('eval_after_epoch', False):
            self.build_test_dataset()

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
            if self.args.train.get('eval_after_epoch', False) and self.end_of_epoch:
                self.inference_once()
            # Increase epoch if rank 0 reach end of epoch.
            self._epoch += 1
            if self.train_cfg.get('drop_when_epoch_end', False):
                while batch is not None:
                    # pylint:disable=not-callable
                    batch = self.train_data_loader.next()
        return batch

    @get_time('data_time')
    def next_train_batch(self):
        '''next train batch'''
        batch_data = self.train_data_loader.next()
        if hasattr(self.args.train, 'iters_per_epoch'):
            # if set iters_per_epoch, will not do dist_sync_epoch operator
            if batch_data is None:
                self.train_data_loader.reset()
                batch_data = self.train_data_loader.next()
                if batch_data is None:
                    raise RuntimeError(
                        "%s - rank %d data loader error"
                        % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.rank)
                    )
            return batch_data
        batch_data = self.dist_sync_epoch(batch_data)
        if batch_data is None:
            self.call_hook('after_train_epoch')
            # epoch ending or dataiter error
            self.train_data_loader.reset()
            if (
                not self.train_cfg.get(
                    'sync_epoch', self.train_cfg.get('drop_when_epoch_end', False)
                )
                or self.world_size == 1
            ):
                self._epoch += 1
            self.call_hook('before_train_epoch')
            self._inner_iter = 0
            self.train_log_buffer.reset()
            batch_data = self.train_data_loader.next()
            if batch_data is None:
                raise RuntimeError(
                    "%s - rank %d data loader error"
                    % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.rank)
                )
        return batch_data

    def build_test_dataset(self):
        '''build test dataset'''
        data_root = self.dataset_cfg.get("data_root", None)
        eval_data_root = self.dataset_cfg.get("eval_data_root", data_root)
        eval_file_list = self.dataset_cfg.get("eval_file_list", None)
        self.get_eval_list(eval_data_root, eval_file_list)

        if hasattr(self.dataset_cfg, 'bucket_schedule_val'):
            eval_bucket_schedule = self.dataset_cfg.bucket_schedule_val
        elif hasattr(self.dataset_cfg, 'bucket_schedule'):
            eval_bucket_schedule = self.dataset_cfg.bucket_schedule

        eval_device_trans_cfg = self.dataset_cfg.get(
            'inference_device_transform', []
        )  # device transform
        self.eval_device_trans_fn = build_device_augmentation(eval_device_trans_cfg, self.meta_data)

        # TODO(@litianyu): rename this flag
        # whether to calculate pesq separately for each data set
        split_each_dataset = self.args.solution.inference.get('eval_multi_cer', False)
        eval_batch_size = self.args.data.get('eval_batch_size', 1)
        eval_dataset_cfg = self.dataset_cfg
        eval_dataset_cfg.max_batch_size = eval_batch_size
        self.eval_data_loader = ValidHDFSDataset(
            self.eval_file_list,
            eval_bucket_schedule,
            eval_dataset_cfg,
            self.parse_fn_inference,
            self.draw_inference_batch_fn,
            self.eval_device_trans_fn,
            split_path_list_by_rank=False,
            split_each_dataset=split_each_dataset,
        )

    @torch.no_grad()
    def inference_once(self):
        '''do inference once'''
        self.mode = 'eval'
        self.solution.eval()
        self.eval_data_loader.reset()
        if self.solution_cfg.inference.get('eval_multi_cer', False):
            dataset_names = [os.path.basename(p) for p in self.eval_data_loader.origin_path_list]
        else:
            dataset_names = [None]

        for dataset_name in dataset_names:
            self.eval_log_buffer.reset()
            batch_data = self.eval_data_loader.next()
            self._val_iter = 0
            while batch_data is not None:
                self.call_hook('before_val_iter')
                try:
                    self.solution.eval()
                    evaluation_out = self.solution.inference(batch_data)
                    self._val_iter += 1
                    batch_data = self.eval_data_loader.next()
                    self.call_hook('after_val_iter')
                    self.eval_log_buffer.update(evaluation_out)
                except RuntimeError as e:
                    clear_cuda_error()
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            if self.args.solution.inference.get('log_metric', False):
                self.log_metric(self.eval_log_buffer, name=dataset_name)
            if self.args.solution.inference.get('final_call', None) is not None:
                call_func = getattr(self.eval_log_buffer, self.args.solution.inference.final_call)
                save_files = call_func(self.solution.infer_save_dir)
                if self.args.solution.inference.remote_save_dir:
                    for save_file in save_files:
                        hdfs_put(save_file, self.args.solution.inference.remote_save_dir)
        self.mode = 'train'
        self.solution.train()

    @torch.no_grad()
    def inference(self):
        '''inference'''
        self.build_test_dataset()
        # TODO(@litianyu.y): replace se inference cfg form solution.inference to inference
        inference_cfg = self.args.solution.inference
        save_dir = self.args.solution.inference.get('save_dir', './inference')
        mkdir_or_exist(save_dir)
        chkpts_path = inference_cfg.get('chkpts_path', None)
        if self.args.solution.inference.get("remote_save_dir", False):
            hdfs_mkdir(self.args.solution.inference.remote_save_dir)
        if not chkpts_path:
            self.inference_once()
            return

        if '|' in chkpts_path:
            chkpt_remote_dir, chkpts_path = chkpts_path.split('|')
        else:
            chkpt_remote_dir = os.path.join(
                self.train_cfg.remote_save_root,
                self.train_cfg.save_dir,
                self.train_cfg.save_name,
                'checkpoints',
            )
        chkpt_paths = chkpts_path.split(',')
        checkpoint_dir = os.path.join(
            self.train_cfg.save_root,
            self.train_cfg.save_dir,
            self.train_cfg.save_name,
            'checkpoints',
        )
        for chkpt in chkpt_paths:
            step_stat_dir = save_dir + '/' + chkpt.split('.pth')[0]
            mkdir_or_exist(step_stat_dir)
            chkpt = os.path.join(chkpt_remote_dir, chkpt)
            if self._resume_from_pretrain(chkpt, checkpoint_dir):
                logging.info('resume chkpt: %s', chkpt)
                logging.info('save enh wavs to %s', step_stat_dir)
                self.args.solution.inference.save_dir = step_stat_dir
                self.inference_once()
            else:
                logging.info('resume failed: %s', chkpt)

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = eval(self.args.train.get('metric', 'SeMetric'))()
        self.valid_log_buffer = eval(self.args.train.get('metric', 'SeMetric'))()
        self.eval_log_buffer = eval(self.args.solution.inference.get('metric', 'SeMetric'))()

    def update_best_metric(self):
        '''
        update best metric to current metric if current is better.

        Return:
            True: current metric is better.
            False: otherwise.
        '''
        best_metric_name = self.train_cfg.best_metric_config.get('best_metric_name', None)
        if best_metric_name is None:
            return False
        metric_value = self.valid_log_buffer.get_value(best_metric_name)
        if metric_value == 0.0:
            logging.error("rank %d catch cur loss is 0.0, not update best metric", self.rank)
            return False
        best_metric_type = self.train_cfg.best_metric_config.get('best_metric_type', 'min')
        if best_metric_type == 'max':
            if self._best_metric is None or metric_value >= self._best_metric:
                self._best_metric = metric_value
                return True
        else:
            if self._best_metric is None or metric_value <= self._best_metric:
                self._best_metric = metric_value
                return True
        return False

    @get_time('time')
    def train_iteration(self):
        '''train iteration.'''
        i = 0
        while i < self.grad_accum_step:
            batch_data = self.next_train_batch()
            try:
                self.solution.train()
                batch_data['loss_scale'] = self.dist_handler.get_scale(scaler_idx=0)
                solution_out = self.solution(batch_data)
                if self.train_cfg.get('qat', False):
                    kurt_loss = get_kurt_loss(self.solution)
                    self.loss = (solution_out['backward_loss'] + kurt_loss) / self.grad_accum_step
                else:
                    self.loss = solution_out['backward_loss'] / self.grad_accum_step

                if self.solution_cfg.get('use_distiller', None):
                    distiller_loss = self.distiller.caculate_distill_loss()
                    self.loss += distiller_loss
                    solution_out['distiller_loss'] = distiller_loss
                self.dist_handler.backward(self.loss, unscale=(i + 1 == self.grad_accum_step))
                if self.train_cfg.get('skip_nan', False) and (
                    torch.isnan(self.loss) or torch.isinf(self.loss)
                ):
                    logging.error("rank %d, iter %d get nan loss", self.rank, self.iter)
                    self.optimizer.zero_grad()
            except RuntimeError as e:
                self.handle_error(e, batch_data)
                continue
            self.train_log_buffer.update(solution_out)
            i = i + 1
        if self.opt_util_cfg.grad_clip:
            gnorm = self.clip_grads()
            if self.train_cfg.get('skip_nan', False) and (torch.isnan(gnorm) or torch.isinf(gnorm)):
                logging.error("rank %d, iter %d get nan gnorm", self.rank, self.iter)
                self.optimizer.zero_grad()
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
                if self.args.train.get('eval_after_epoch', False):
                    self.inference_once()
                self.call_hook('after_train_epoch')
                self._epoch += 1
                # epoch ending or data iter error
                self.train_data_loader.reset()
                if self._epoch >= self.args.train.max_epochs:
                    break
                self.call_hook('before_train_epoch')
                self.train_log_buffer.reset()
                # Check current learning rate
                max_lr = max(param_group['lr'] for param_group in self.optimizer.param_groups)
                if max_lr < self.args.train.get('final_lr', 1e-7):
                    break
                self._inner_iter = 0
            else:
                self._inner_iter += 1
            if self._epoch >= self.args.train.max_epochs:
                break
            self._iter += 1
        self.after_train()
