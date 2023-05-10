''' BaseRNNTRunner '''

import torch
from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)

from .base_asr_runner import BaseAsrRunner
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class BaseRNNTRunner(BaseAsrRunner):
    '''RNNT for TAU'''

    @torch.no_grad()
    def validation(self):
        '''valid'''
        # Switch to eval model
        self.mode = 'val'
        self.solution.eval()
        if hasattr(self.solution.criterion_module, "combine_weight"):
            self.solution.criterion_module.combine_weight()
        super().validation()
        if hasattr(self.solution.criterion_module, "clear_combined_weight"):
            self.solution.criterion_module.clear_combined_weight()
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        self.pre_build_dataset(dataset_cfg)

        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        # item-trans
        parse_cfg = dataset_cfg.train_item_transform
        self.parse_fn = build_item_augmentation(parse_cfg, self.meta_data)

        # draw batch fn
        draw_batch_cfg = dataset_cfg.get("batch_transform", [])
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)
        draw_train_batch_cfg = dataset_cfg.get('train_batch_transform', draw_batch_cfg)
        self.draw_train_batch_fn = build_draw_batch_fn(draw_train_batch_cfg, self.meta_data)
        draw_valid_batch_cfg = dataset_cfg.get("valid_batch_transform", draw_batch_cfg)
        self.draw_valid_batch_fn = build_draw_batch_fn(draw_valid_batch_cfg, self.meta_data)

        # device transform
        device_trans_cfg = dataset_cfg.get('device_transform', [])
        self.device_trans = build_device_augmentation(device_trans_cfg, self.meta_data)
        train_device_trans_cfg = dataset_cfg.get('train_device_transform', device_trans_cfg)
        self.train_device_trans = build_device_augmentation(train_device_trans_cfg, self.meta_data)
        valid_device_trans_cfg = dataset_cfg.get('valid_device_transform', device_trans_cfg)
        self.valid_device_trans = build_device_augmentation(valid_device_trans_cfg, self.meta_data)

        self.get_data_list(dataset_cfg)
        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule

        split_path_list_by_rank = dataset_cfg.get('split_path_list_by_rank', 1)
        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.parse_fn,
            self.draw_train_batch_fn,
            device_transforms=self.train_device_trans,
            split_path_list_by_rank=split_path_list_by_rank,
            shuffle=True,
        )

        valid_split_each_dataset = self.solution_cfg.get('valid_multi_cer', False)
        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            val_bucket_schedule,
            dataset_cfg,
            self.parse_fn_eval,
            self.draw_valid_batch_fn,
            device_transforms=self.valid_device_trans,
            split_path_list_by_rank=False,
            split_each_dataset=valid_split_each_dataset,
        )

    def build_beam_search(self, key='inference'):
        '''build beam search'''
        cfg = self.args.get(key, None)
        if cfg is not None:
            if key != 'solution':
                cfg.downsampling_size = self.solution_cfg.get('downsampling_size', None)
            if hasattr(self, 'reorder_dict_map'):
                cfg.reorder_dict_map = getattr(self, 'reorder_dict_map')
            self.solution.init_beam_search(cfg, self.lm_solution)
