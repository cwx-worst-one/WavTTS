''' DFSMNRunner '''

import torch

from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
)
from core.utils import logging
from ..base_runner import RUNNERS
from .base_rnnt_runner import BaseRNNTRunner


@RUNNERS.register_module()
class RNNTCePretrainRunner(BaseRNNTRunner):
    '''pretrain acoustic encoder for RNN-T'''

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)

        self.tgt_vocab_size = dataset_cfg.tgt_vocab_size
        # align to 8
        # self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1)//8)*8
        logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
        self.solution_cfg.update({'tgt_vocab_size': self.tgt_vocab_size})

        self.setup_transform_cfg(dataset_cfg)

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        self.pre_build_dataset(dataset_cfg)

        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        parse_cfg = dataset_cfg.train_item_transform
        self.parse_fn = build_item_augmentation(parse_cfg, self.meta_data)
        draw_batch_cfg = dataset_cfg.batch_transform
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)

        self.get_data_list(dataset_cfg)

        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule

        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.parse_fn,
            self.draw_batch_fn,
            shuffle=True,
        )

        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            val_bucket_schedule,
            dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
        )

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
                validation_out = self.solution(batch_data)
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
