''' text injection runner. '''
import os.path as osp
import time
import torch

from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.utils import (
    logging,
)
from core.runner.asr.base_cif_runner import BaseCifRunner
from ..utils import get_time, format_file_list
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class CifMostRunner(BaseCifRunner):
    '''CIF Most Runner.'''

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        self.pre_build_dataset(dataset_cfg)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.text_only_train_data_loader = None
            self.valid_data_loader = None
            return

        # item transform
        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)
        parse_cfg = dataset_cfg.train_item_transform
        self.parse_fn = build_item_augmentation(parse_cfg, self.meta_data)
        ## add text_only item_transform cfg
        if dataset_cfg.get('add_text_only_data_loader', None):
            parse_text_only_cfg = dataset_cfg.text_only_train_item_transform
            self.parse_text_only_fn = build_item_augmentation(parse_text_only_cfg, self.meta_data)

        # batch transform
        draw_batch_cfg = dataset_cfg.batch_transform
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)
        ## add text_only batch_transform cfg
        if dataset_cfg.get('add_text_only_data_loader', None):
            draw_text_only_batch_cfg = dataset_cfg.text_only_batch_transform
            self.draw_text_only_batch_fn = build_draw_batch_fn(
                draw_text_only_batch_cfg, self.meta_data
            )

        device_trans_cfg = dataset_cfg.get('train_device_transform', None)
        self.device_trans = build_device_augmentation(device_trans_cfg, self.meta_data)

        self.get_data_list(dataset_cfg)
        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule
        ## add bucket_schedule for text_only data
        if dataset_cfg.get('add_text_only_data_loader', None):
            text_only_bucket_schedule = dataset_cfg.text_only_bucket_schedule

        split_path_list_by_rank = dataset_cfg.get('split_path_list_by_rank', 1)
        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.parse_fn,
            self.draw_batch_fn,
            device_transforms=self.device_trans,
            split_path_list_by_rank=split_path_list_by_rank,
            shuffle=True,
        )

        valid_split_each_dataset = self.solution_cfg.get('valid_multi_cer', False)
        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            val_bucket_schedule,
            dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
            split_each_dataset=valid_split_each_dataset,
        )

        # add text_only data_loader
        self.text_only_train_data_loader = None
        if dataset_cfg.get('add_text_only_data_loader', None):
            self.text_only_train_data_loader = HDFSDataset(
                self.text_only_train_file_list,
                text_only_bucket_schedule,
                dataset_cfg,
                self.parse_text_only_fn,
                self.draw_text_only_batch_fn,
                device_transforms=self.device_trans,
                split_path_list_by_rank=split_path_list_by_rank,
                shuffle=True,
                max_batch_size=dataset_cfg.get(
                    'text_only_max_batch_size', dataset_cfg.max_batch_size
                ),
                bucket_schedule_key=dataset_cfg.get(
                    'text_only_bucket_schedule_key', dataset_cfg.bucket_schedule_key
                ),
                cache_name='text_only_cache_name',
            )

    @get_time('data_time')
    def next_train_batch(self):
        '''next train batch'''
        batch_data = self.train_data_loader.next()
        batch_data = self.dist_sync_epoch(batch_data)

        text_only_batch_data = None
        if self.text_only_train_data_loader is not None:
            text_only_batch_data = self.text_only_train_data_loader.next()
            text_only_batch_data = self.dist_sync_epoch(text_only_batch_data)
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

        if self.text_only_train_data_loader is not None and text_only_batch_data is None:
            self.call_hook('after_train_epoch')
            # epoch ending or dataiter error
            self.text_only_train_data_loader.reset()
            self.call_hook('before_train_epoch')
            self._inner_iter = 0
            # self.train_log_buffer.reset()
            text_only_batch_data = self.text_only_train_data_loader.next()
            if text_only_batch_data is None:
                raise RuntimeError(
                    "%s - rank %d data loader error"
                    % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.rank)
                )

        return batch_data, text_only_batch_data

    def start_data_loader(self):
        '''post build dataset.'''
        if not self.need_build_data_loader:
            return
        self.train_data_loader.reset()
        self.valid_data_loader.reset()
        if self.text_only_train_data_loader is not None:
            self.text_only_train_data_loader.reset()

    @get_time('time')
    def train_iteration(self):
        '''train iteration.'''
        i = 0
        loss_scale = self.dist_handler.get_scale(scaler_idx=0)
        enable_grad_clip = self.opt_util_cfg.grad_clip and self.iter > self.lr_cfg.get(
            'warmup_steps', 0
        )
        self.grads_accumulator.reset(enable_grad_clip=enable_grad_clip, loss_scale=loss_scale)
        while i < self.grad_accum_step:
            batch_data, text_only_batch_data = self.next_train_batch()
            try:
                self.solution.train()
                batch_data['loss_scale'] = loss_scale
                self.solution_out = self.solution(batch_data, text_only_batch_data)

                # in cif, the accumulated gradients are not averaged by accumulation steps,
                # and the gradients are cliped before accumulation, these decease the final
                # cer by about 1%
                if self.grad_accum_mode == 'AVG':
                    self.loss = self.solution_out['backward_loss'] / self.grad_accum_step
                else:
                    self.loss = self.solution_out['backward_loss']
                self.dist_handler.backward(self.loss, unscale=False)
                self.grads_accumulator.clip_and_accumulate(
                    self.dist_handler.parameters,
                    optimizer=self.optimizer,
                    last_step=(i + 1 == self.grad_accum_step),
                )
            except RuntimeError as e:
                self.handle_error(e, batch_data)
                continue
            with torch.no_grad():
                self.train_log_buffer.update(self.solution_out)
            i = i + 1
        # get grads_norm and clip grads if needed
        gnorm = self.grads_accumulator.after_accumulation(self.dist_handler.parameters, dict())
        self.solution_out.update(gnorm)
        with torch.no_grad():
            self.train_log_buffer.update(gnorm)
        # optimizer step
        self.dist_handler.step(iters=self.iter)

    def get_data_list(self, dataset_cfg):
        '''
        get valid_file_list,train_file_list
        '''
        data_root = dataset_cfg.get("data_root", None)
        train_data_root = dataset_cfg.get("train_data_root", data_root)
        valid_data_root = dataset_cfg.get("valid_data_root", data_root)
        text_only_data_root = dataset_cfg.get("text_only_data_root", None)
        meta_data_root = dataset_cfg.get('meta_data_root', train_data_root)
        train_file_list = dataset_cfg.get("train_file_list", None)
        valid_file_list = dataset_cfg.get("valid_file_list", None)
        text_only_train_file_list = dataset_cfg.get("text_only_train_file_list", None)
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
            text_only_data_root = text_only_data_root.split(data_path_separator)
            text_only_train_file_list = text_only_train_file_list.split(data_path_separator)
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

        # get text_only_train_file_lists
        if text_only_data_root is not None:
            if isinstance(text_only_data_root, str):
                text_only_data_root = [text_only_data_root]
                text_only_train_file_list = [text_only_train_file_list]
            self.text_only_train_file_list = []
            assert len(text_only_data_root) == len(text_only_train_file_list)
            for text_only_root, text_only_file in zip(
                text_only_data_root, text_only_train_file_list
            ):
                text_only_dataset_now = [osp.join(text_only_root, p) for p in eval(text_only_file)]
                self.text_only_train_file_list += text_only_dataset_now
            logging.info(
                f"The text_only_train list: \n{format_file_list(self.text_only_train_file_list)}\n"
            )

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

    @torch.no_grad()
    def validation(self):
        '''valid'''
        # Switch to eval model
        self.mode = 'val'
        self.solution.eval()
        super().validation()
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()

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
            self.inference()
        else:
            self.train()
            self.train_data_loader.terminate()
            if self.text_only_train_data_loader is not None:
                self.text_only_train_data_loader.terminate()
            self.valid_data_loader.terminate()
        self.call_hook('after_run')
        self.report_metric()
