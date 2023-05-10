''' base asr runner. '''
import os.path as osp
from abc import ABCMeta
import torch
from core.runner.metric.asr_metric import AsrMetric
from core.dataset import HDFSDataset, ValidHDFSDataset
from core.utils import dist_allreduce, ReduceOp
from core.utils import logging
from ..base_runner import BaseRunner, RUNNERS


@RUNNERS.register_module()
class BaseMddRunner(BaseRunner, metaclass=ABCMeta):
    '''Base ASR Runner.'''

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
        super().build_dataset(dataset_cfg)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None

            data_root = self.dataset_cfg.get("data_root", None)
            eval_data_root = self.dataset_cfg.get("eval_data_root", data_root)
            valid_file_list = self.dataset_cfg.get("valid_file_list", None)
            eval_file_list = self.dataset_cfg.get("eval_file_list", None)

            valid_file_lists = self.get_eval_list(eval_data_root, valid_file_list)
            eval_file_lists = self.get_eval_list(eval_data_root, eval_file_list)

            self.eval_data_loader1 = ValidHDFSDataset(
                valid_file_lists,
                self.val_bucket_schedule,
                self.dataset_cfg,
                self.valid_item_trans,
                self.draw_batch_fn,
                split_path_list_by_rank=False,
            )

            self.eval_data_loader2 = ValidHDFSDataset(
                eval_file_lists,
                self.val_bucket_schedule,
                self.dataset_cfg,
                self.valid_item_trans,
                self.draw_batch_fn,
                split_path_list_by_rank=False,
            )

        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.train_item_trans,
            self.draw_batch_fn,
            shuffle=True,
        )

        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            self.val_bucket_schedule,
            dataset_cfg,
            self.valid_item_trans,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
        )

    @torch.no_grad()
    def evaluation(self, data_loader):
        '''evaluation'''
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
                validation_out = self.solution.evaluation(batch_data)
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
        self.train_log_buffer = AsrMetric()
        self.valid_log_buffer = AsrMetric()

    def inference(self):
        '''inference.'''
        self.evaluation(self.eval_data_loader1)
        self.evaluation(self.eval_data_loader2)

    def dist_sync_epoch(self, batch):
        '''sync epoch info between multi rank.'''
        if self.world_size == 1:
            return batch
        if not self.train_cfg.get('sync_epoch', self.train_cfg.get('drop_when_epoch_end', False)):
            return batch
        if self.inner_iter < self.train_cfg.get('min_sync_iter', 0):
            return batch
        if not hasattr(self, 'end_of_epoch'):
            self.end_of_epoch = torch.ByteTensor([False]).cuda()
        self.end_of_epoch[0] = batch is None
        dist_allreduce(self.end_of_epoch, name='sync_epoch', op=ReduceOp.SUM)
        if self.end_of_epoch:
            self._epoch += 1
            if self.train_cfg.get('drop_when_epoch_end', False):
                while batch is not None:
                    # pylint:disable=not-callable
                    batch = self.train_data_loader.next()
        return batch
