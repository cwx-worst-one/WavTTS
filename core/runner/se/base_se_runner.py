''' base se runner. '''
import os.path as osp
from abc import ABCMeta
import torch
from core.dataset import HDFSDataset, ValidHDFSDataset, build_item_augmentation, build_draw_batch_fn
from core.utils import logging
from core.runner.metric.se_metric import SeMetric
from ..base_runner import BaseRunner, RUNNERS
from ..utils import get_time


@RUNNERS.register_module()
class BaseSeRunner(BaseRunner, metaclass=ABCMeta):
    '''Base SE Runner.'''

    def __init__(self, cfg, inference=False, export_onnx=False):
        super().__init__(cfg, inference, export_onnx)
        self.loss_total = 0

    @staticmethod
    def get_data_list(dataset_cfg):
        '''
        to get valid_file_list and train_file_list
        '''
        data_root = dataset_cfg.get("data_root", None)
        train_data_root = dataset_cfg.get("train_data_root", data_root)
        valid_data_root = dataset_cfg.get("valid_data_root", data_root)
        train_file_list = dataset_cfg.get("train_file_list", None)
        valid_file_list = dataset_cfg.get("valid_file_list", None)
        if isinstance(train_data_root, str):
            train_data_root = [train_data_root]
            train_file_list = [train_file_list]
        train_files = []
        assert len(train_data_root) == len(train_file_list)
        for train_root, train_file in zip(train_data_root, train_file_list):
            print(train_file)
            train_dataset_now = [osp.join(train_root, p) for p in train_file]
            train_files += train_dataset_now
        if isinstance(valid_data_root, str):
            valid_data_root = [valid_data_root]
            valid_file_list = [valid_file_list]
        valid_files = []
        assert len(valid_data_root) == len(valid_file_list)
        for valid_root, valid_file in zip(valid_data_root, valid_file_list):
            valid_dataset_now = [osp.join(valid_root, p) for p in valid_file]
            valid_files += valid_dataset_now
        return train_files, valid_files

    @staticmethod
    def get_eval_list(eval_data_root, eval_file_list):
        '''
        to get eval_file_list
        '''
        if isinstance(eval_data_root, str):
            eval_data_root = [eval_data_root]
            eval_file_list = [eval_file_list]
        eval_files = []
        assert len(eval_data_root) == len(eval_file_list)
        for eval_root, eval_file in zip(eval_data_root, eval_file_list):
            eval_dataset_now = [osp.join(eval_root, p) for p in eval(eval_file)]
            eval_files += eval_dataset_now
        return eval_files

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        # self.pre_build_dataset(dataset_cfg)
        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)
        parse_cfg = dataset_cfg.train_item_transform
        self.parse_fn = build_item_augmentation(parse_cfg, self.meta_data)
        draw_batch_cfg = dataset_cfg.batch_transform
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)

        if self.is_inference or self.is_export_onnx:
            self.train_data_loader = None
            self.valid_data_loader = None

            data_root = self.dataset_cfg.get("data_root", None)
            eval_data_root = self.dataset_cfg.get("eval_data_root", data_root)
            valid_file_list = self.dataset_cfg.get("valid_file_list", None)
            eval_file_list = self.dataset_cfg.get("eval_file_list", None)

            valid_file_lists = self.get_eval_list(eval_data_root, valid_file_list)
            eval_file_lists = self.get_eval_list(eval_data_root, eval_file_list)

            if hasattr(self.dataset_cfg, 'bucket_schedule_val'):
                eval_bucket_schedule = self.dataset_cfg.bucket_schedule_val
            else:
                eval_bucket_schedule = self.dataset_cfg.bucket_schedule

            self.eval_data_loader1 = ValidHDFSDataset(
                valid_file_lists,
                eval_bucket_schedule,
                self.dataset_cfg,
                self.parse_fn_eval,
                self.draw_batch_fn,
                split_path_list_by_rank=False,
            )

            self.eval_data_loader2 = ValidHDFSDataset(
                eval_file_lists,
                eval_bucket_schedule,
                self.dataset_cfg,
                self.parse_fn_eval,
                self.draw_batch_fn,
                split_path_list_by_rank=False,
            )
        train_file_list, valid_file_list = self.get_data_list(dataset_cfg)

        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule

        self.train_data_loader = HDFSDataset(
            train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.parse_fn,
            self.draw_batch_fn,
            shuffle=True,
        )

        self.valid_data_loader = ValidHDFSDataset(
            valid_file_list,
            val_bucket_schedule,
            dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
        )

    @get_time('time')
    def train_iteration(self):
        batch_data = self.next_train_batch()
        try:
            grad_accum_step = self.train_cfg.get('grad_accum_step', 1)
            self.solution.train()
            solution_out = self.solution.forward(batch_data)
            self.loss = solution_out["backward_loss"] / grad_accum_step
            self.loss_total += self.loss
            solution_out["avg_loss"] = self.loss_total / self._inner_iter
            delay_unscale = ((self.iter + 1) % grad_accum_step) != 0
            self.dist_handler.backward(self.loss, unscale=not delay_unscale)
        except RuntimeError as e:
            self.handle_error(e, batch_data)
        self.train_log_buffer.update(solution_out)
        if not delay_unscale:
            if self.opt_util_cfg.grad_clip:
                gnorm = self.clip_grads()
                self.train_log_buffer.update({'gnorm': gnorm})
            self.dist_handler.step(iters=self.iter)

    @torch.no_grad()
    def validation(self):
        '''valid'''
        # Switch to eval model
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self.valid_log_buffer.reset()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                self.solution.eval()
                validation_out = self.solution.forward(batch_data)
                self._val_iter += 1
                batch_data = self.valid_data_loader.next()
                self.call_hook('after_val_iter')
                self.valid_log_buffer.update(validation_out)
            except RuntimeError as e:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                continue
        self.call_hook('after_val_epoch')
        self.log_metric(self.valid_log_buffer)
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()

    @torch.no_grad()
    def inference(self, data_loader):
        '''inference'''
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_log_buffer.reset()
        # self.eval_data_loader.reset()
        # batch_data = self.eval_data_loader.next()

        data_loader.reset()
        batch_data = data_loader.next()
        self._val_iter = 0
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                self.solution.eval()
                validation_out = self.solution.inference(batch_data)
                self._val_iter += 1
                # batch_data = self.eval_data_loader.next()
                batch_data = data_loader.next()
                self.call_hook('after_val_iter')
                self.valid_log_buffer.update(validation_out)
            except RuntimeError as e:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                continue
        self.call_hook('after_val_epoch')
        self.log_metric(self.valid_log_buffer)

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = SeMetric()
        self.valid_log_buffer = SeMetric()

    def update_best_metric(self):
        '''Update best metric to save checkpoint'''
        cur_loss = self.valid_log_buffer.get_value('loss')
        if cur_loss == 0.0:
            logging.error("rank %d catch cur loss is 0.0, not update best metric", self.rank)
            return False
        if self._best_metric is None or cur_loss <= self._best_metric:
            self._best_metric = cur_loss
            return True
        return False

    def run(self):
        '''Entrypoint of runner'''
        self.call_hook('before_run')
        if self.is_export_onnx:
            self.export_onnx()
        elif self.is_inference:
            self.inference(self.eval_data_loader1)
            self.inference(self.eval_data_loader2)
        else:
            self.train()
            self.train_data_loader.terminate()
            self.valid_data_loader.terminate()
        self.call_hook('after_run')
        self.report_metric()
