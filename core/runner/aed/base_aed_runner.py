'''base aed runner'''
import time
import json
import os
import os.path as osp
import numpy as np
from core.dataset import HDFSDataset, ValidHDFSDataset, BalancedHDFSDataset
from core.runner.base_runner import RUNNERS, BaseRunner
from core.runner.metric.aed_metric import *
from core.utils import logging, dist_allreduce, ReduceOp
from core.utils.recall import (
    compute_average_precision,
    compute_area_under_curve,
    compute_equal_error_rate,
    compute_d_prime,
)
from core.extensions import clear_cuda_error
from core.utils import hdfs_put, hdfs_get, hdfs_mkdir, get_rank, load_checkpoint


@RUNNERS.register_module()
class BaseAedRunner(BaseRunner):
    '''Base AED Runner'''

    @staticmethod
    def _get_file_list(dataset_cfg, data_root_name, file_list_name, file_pattern_name):
        '''get file list'''
        ret_file_list = []

        # get file list from data_root and file_list attr
        data_root = dataset_cfg.get("data_root", None)
        cur_data_root = dataset_cfg.get(data_root_name, data_root)
        cur_file_list = dataset_cfg.get(file_list_name, None)

        if not isinstance(cur_data_root, (list, tuple)):
            cur_data_root = [cur_data_root]
            cur_file_list = [cur_file_list]
        assert len(cur_data_root) == len(cur_file_list)

        for cur_root, cur_file in zip(cur_data_root, cur_file_list):
            if cur_file:
                ret_file_list += [osp.join(cur_root, p) for p in eval(cur_file)]

        # get file list from file pattern
        cur_file_pattern = dataset_cfg.get(file_pattern_name, None)
        if not cur_file_pattern:
            return ret_file_list
        if not isinstance(cur_file_pattern, (list, tuple)):
            cur_file_pattern = cur_file_pattern.split(',')
        for cur_pattern in cur_file_pattern:
            if cur_pattern.startswith("hdfs://"):
                text_wrapper = os.popen(f"hdfs dfs -ls {cur_pattern}*.index")
            elif cur_pattern.startswith("/mnt"):
                text_wrapper = os.popen(f"ls {cur_pattern}*.index")
            for line in text_wrapper:
                for elem in line.strip().split(" "):
                    if elem.startswith("hdfs://") or elem.startswith("/mnt/bd"):
                        ret_file_list.append(elem[:-6])
                        break
            text_wrapper.close()

        return ret_file_list

    def get_data_list(self, dataset_cfg):
        '''
        get valid_file_list,train_file_list
        '''
        self.train_file_list = self._get_file_list(
            dataset_cfg, 'train_data_root', 'train_file_list', 'train_file_pattern'
        )
        logging.info("The training list: [ " + ",".join(self.train_file_list) + "  ]")
        self.valid_file_list = self._get_file_list(
            dataset_cfg, 'valid_data_root', 'valid_file_list', 'valid_file_pattern'
        )
        logging.info("The valid list: [ " + ",".join(self.valid_file_list) + "  ]")
        self.meta_file_list = []

    def pre_build_dataset(self, dataset_cfg):
        '''update item transform and batch transform'''
        super().pre_build_dataset(dataset_cfg)

        def _update_dataset_cfg(sign, min_len, max_len):
            # update item_transform
            for item_trans in dataset_cfg.get('{}_item_transform'.format(sign), []):
                if item_trans['type'] == 'LengthFilter':
                    item_trans['min_len'] = dataset_cfg.filter_len
                elif item_trans['type'] == 'AppendFrames':
                    item_trans['fix_frames'] = max_len
                elif item_trans['type'] == 'CutFeature':
                    item_trans['max_len'] = max_len
            # update batch_transform
            for batch_trans in dataset_cfg.get('{}_batch_transform'.format(sign), []):
                if batch_trans['type'] == 'OnehotLabelCollate':
                    batch_trans['num_classes'] = dataset_cfg.num_classes
                elif batch_trans['type'] == 'LengthRandomClipCollate':
                    batch_trans['min_len'] = min_len
                    batch_trans['max_len'] = max_len

        # update train_*_transform
        _update_dataset_cfg('train', dataset_cfg.min_len, dataset_cfg.max_len)
        # update valid_*_transform
        _update_dataset_cfg('valid', dataset_cfg.eval_len, dataset_cfg.eval_len)
        # update solution_cfg
        self.solution_cfg.setdefault('num_classes', dataset_cfg.num_classes)
        # apply eval_variable: skip all the shape modify
        if dataset_cfg.get('eval_variable', False):
            valid_item_transform = []
            for item_trans in dataset_cfg.get('valid_item_transform', []):
                if item_trans['type'] not in ['AppendFrames', 'CutFeature']:
                    valid_item_transform.append(item_trans)
            if valid_item_transform:
                dataset_cfg['valid_item_transform'] = valid_item_transform
            valid_batch_transform = []
            for batch_trans in dataset_cfg.get('valid_batch_transform', []):
                if batch_trans['type'] not in ['LengthRandomClipCollate']:
                    valid_batch_transform.append(batch_trans)
                else:
                    # there is at least one transform for collate the batch tensor
                    stack_collate = dict(
                        type='StackCollate',
                        key=batch_trans.get('key', None),
                        out_key=batch_trans.get('out_key', None),
                    )
                    valid_batch_transform.append(stack_collate)
            if valid_batch_transform:
                dataset_cfg['valid_batch_transform'] = valid_batch_transform
        # apply specaug transform
        if dataset_cfg.get('specaug', False):
            train_batch_transform = dataset_cfg.get('train_batch_transform', [])
            freq_mask = dict()
            freq_mask['type'] = 'FreqMaskCollate'
            freq_mask['key'] = dataset_cfg.get('spec_key', 'features')
            freq_mask['freq_mask_num'] = dataset_cfg.get('freq_mask_count', 1)
            freq_mask['freq_mask_size'] = dataset_cfg.get('freq_mask_max_bins', 8)
            freq_mask['replace_with_zero'] = True
            train_batch_transform.append(freq_mask)
            time_mask = dict()
            time_mask['type'] = 'TimeMaskCollate'
            time_mask['key'] = dataset_cfg.get('spec_key', 'features')
            time_mask['time_mask_num'] = dataset_cfg.get('time_mask_count', 1)
            time_mask['time_mask_size'] = dataset_cfg.get('time_mask_max_len', 10)
            time_mask['replace_with_zero'] = True
            time_mask['max_time_p'] = 1.0
            train_batch_transform.append(time_mask)

    def build_dataset(self, dataset_cfg):
        '''build dataset'''
        super().build_dataset(dataset_cfg)
        if not self.need_build_data_loader:
            return

        split_path_list_by_rank = dataset_cfg.get('split_path_list_by_rank', 1)
        if dataset_cfg.get('batch_balance', False):
            batch_size = dataset_cfg.get('max_batch_size', 256)
            num_classes = dataset_cfg.get('num_classes', 14)
            dataset_cfg['batch_size_per_class'] = batch_size // num_classes
            dataset_cfg['batch_class_num'] = num_classes
            dataset_cfg['balance_epoch'] = True
            self.train_data_loader = BalancedHDFSDataset(
                self.train_file_list,
                dataset_cfg,
                self.train_item_trans,
                self.draw_train_batch_fn,
                device_transforms=self.train_device_trans,
                none_after_epoch=True,
            )
        else:
            self.train_data_loader = HDFSDataset(
                self.train_file_list,
                dataset_cfg.bucket_schedule,
                dataset_cfg,
                self.train_item_trans,
                self.draw_train_batch_fn,
                device_transforms=self.train_device_trans,
                split_path_list_by_rank=split_path_list_by_rank,
                shuffle=True,
            )

        # apply eval_variable's batch_size and bucket
        if dataset_cfg.get('eval_variable', False):
            dataset_cfg['max_batch_size'] = 1
            self.val_bucket_schedule = ''
        valid_split_each_dataset = self.solution_cfg.get('valid_multi_cer', False)
        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            self.val_bucket_schedule,
            dataset_cfg,
            self.valid_item_trans,
            self.draw_valid_batch_fn,
            split_path_list_by_rank=False,
            split_each_dataset=valid_split_each_dataset,
        )

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = TopHitsMetric()
        self.valid_log_buffer = RecallMetric()

    def train(self):
        '''override train func'''
        self.before_train()
        while self.iter < self.args.train.max_iters or self._epoch < self.args.train.max_epochs:
            # train step begin
            self.call_hook('before_train_iter')
            self.train_iteration()
            self.call_hook('after_train_iter')
            self.log_metric(self.train_log_buffer)
            if (
                hasattr(self.args.valid, 'interval')
                and (self.iter + 1) % self.args.valid.interval == 0
            ):
                self.validation()
            if (
                hasattr(self.args.train, 'iters_per_epoch')
                and self._inner_iter + 1 == self.args.train.iters_per_epoch
            ):
                self._epoch += 1
                self.call_hook('after_train_epoch')
                # epoch ending or data iter error
                self.train_data_loader.reset()
                if self._epoch >= self.args.train.max_epochs:
                    break
                self.call_hook('before_train_epoch')
                self.train_log_buffer.reset()
                self._inner_iter = 0
            else:
                self._inner_iter += 1
            self._iter += 1
        self.validation()
        self.after_train()

        if hasattr(self, 'best_metrics') and self.rank == 0:
            if self.best_metric_type == 'topk':
                for k, metrics_item in enumerate(self.best_metrics):
                    logging.info("{}th_best_metrics: {}".format(k + 1, metrics_item))
            else:
                logging.info("best_metrics: {}".format(self.best_metrics))

    def inference(self):
        '''inference func.'''
        logging.error('there is no inference impl yet')

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
            if self.dataset_cfg.get("labels", None):
                self.generate_service_config()
        self.call_hook('after_run')
        self.report_metric()

    def update_best_metric(self):
        '''override update_best_metric'''
        # pylint:disable=too-many-branches
        updated = False
        best_metric_name = self.args.valid.get('best_metric_name', 'loss')
        metric_value = self.valid_log_buffer.get_value(best_metric_name)
        if metric_value == 0.0:
            logging.error("rank %d catch cur loss is 0.0, not update best metric", self.rank)
        else:
            self.best_metric_type = self.args.valid.get('best_metric_type', 'min')
            if self.best_metric_type == 'max':
                if self._best_metric is None or metric_value >= self._best_metric:
                    self._best_metric = metric_value
                    updated = True
            elif self.best_metric_type == 'topk':
                if self._best_metric is None:
                    updated = True
            else:
                if self._best_metric is None or metric_value <= self._best_metric:
                    self._best_metric = metric_value
                    updated = True
        if updated:
            best_metrics = self.valid_log_buffer.get(copy=True)
            if self.best_metric_type == 'topk':
                topk = self.args.valid.get('topk', 5)
                if hasattr(self, 'best_metrics'):
                    self.best_metrics.append({'iter': self._iter + 1, **best_metrics})
                else:
                    self.best_metrics = [{'iter': self._iter + 1, **best_metrics}]
                self.best_metrics.sort(key=lambda x: x[best_metric_name], reverse=True)
                if len(self.best_metrics) > topk:
                    self.best_metrics = self.best_metrics[:topk]
            else:
                self.best_metrics = {'iter': self._iter + 1, **best_metrics}
        return updated

    def generate_service_config(self):
        '''generate online service config'''
        self.solution.eval()
        self.solution.register_infers()
        checkpoint_dir = osp.join(
            self.train_cfg.save_root,
            self.train_cfg.save_dir,
            self.train_cfg.save_name,
            'checkpoints',
        )

        if self.best_metric_type != 'topk':
            self.best_metrics = [self.best_metrics]

        for k, metrics_item in enumerate(self.best_metrics):
            resume_file = 'step_' + str(metrics_item['iter']) + '.pth'
            load_checkpoint(self.solution, [osp.join(checkpoint_dir, resume_file)])
            rank = get_rank()
            if rank != 0:
                return
            local_dir = self.solution_cfg.get('onnx_dir')
            self.solution.export()
            remote_save_root = self.train_cfg.get('remote_save_root', None)
            if remote_save_root:
                save_dir = osp.join(
                    remote_save_root,
                    self.train_cfg.save_dir,
                    self.train_cfg.save_name,
                    'onnxs',
                    'step_' + str(metrics_item['iter']),
                )
                self.aed_petrel_config(self.dataset_cfg, local_dir, save_dir)
                self.aed_threshold_config(self.dataset_cfg, local_dir)
                hdfs_mkdir(save_dir)
                hdfs_put(local_dir, save_dir, sync=True)
                config_dir = osp.join(save_dir, 'onnx/petrel_config.json')
                if k == 0:
                    local_config_dir = osp.join(local_dir, 'petrel_config.json')
                    hdfs_put(local_config_dir, remote_save_root, sync=True)
                logging.info("{}th_best_metrics: {}".format(k + 1, metrics_item))
                logging.info("service config: {}".format(config_dir))

    @staticmethod
    def aed_petrel_config(dataset_cfg, local_dir, save_dir):
        '''petrel config'''
        config_data = dict()
        config_data['decode_info'] = dataset_cfg.labels
        config_data['onnx'] = osp.join(save_dir, 'onnx/AedClassification_optimized.onnx')
        config_data['input_node_name'] = dataset_cfg.get('input_node_name', 'Placeholder')
        config_data['output_node_name'] = dataset_cfg.get('output_node_name', 'predictions')
        config_data['model_input_shape'] = ','.join(
            ['None', str(dataset_cfg.eval_len), str(dataset_cfg.fbank_dim)]
        )
        config_data['stack_size'] = dataset_cfg.eval_len
        config_data['stack_shift'] = dataset_cfg.eval_len
        config_data['use_tail_stack_zero_padding'] = True
        config_data['stack_least_frame'] = dataset_cfg.get('stack_least_frame', -1)
        config_data['front_zero_padding_frames'] = dataset_cfg.get('front_zero_padding_frames', 0)
        config_data['model_type'] = dataset_cfg.get('model_type', 103)
        config_data['is_fixed_shape'] = dataset_cfg.get('is_fixed_shape', True)
        config_data['use_gpu'] = dataset_cfg.get('use_gpu', True)
        config_data['use_batch'] = dataset_cfg.get('use_batch', False)
        config_data['win_size'] = dataset_cfg.get('win_size', 1)
        config_data['win_shift'] = dataset_cfg.get('win_shift', 1)
        config_path = osp.join(local_dir, 'petrel_config.json')
        with open(config_path, 'w', encoding='utf-8') as f_w:
            json.dump(config_data, f_w)

    @staticmethod
    def aed_threshold_config(dataset_cfg, local_dir):
        '''threshold config'''
        labels = []
        thresholds = []
        hdfs_get(dataset_cfg.labels)
        with open(dataset_cfg.labels.split('/')[-1], 'r', encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            labels.append(line.strip().split(',')[1])
            thresholds.append(0.5)
        config_data = dict(zip(labels, thresholds))
        threshold_path = osp.join(local_dir, 'general_threshold.json')
        with open(threshold_path, 'w', encoding='utf-8') as f_w:
            json.dump(config_data, f_w)


@RUNNERS.register_module()
class AedClassificationRunner(BaseAedRunner):
    '''AedClassificationRunner'''

    @torch.no_grad()
    def validation(self):
        '''override validation func'''
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self.valid_log_buffer.reset()
        self._val_iter = 0

        predicts = []
        onehot_labels = []
        num_samples = 0
        start_time = time.time()
        batch_data = self.valid_data_loader.next()
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                self.solution.eval()
                validation_out = self.solution(batch_data)
                # save collections to cpu to avoid gpu oom
                predicts.extend(validation_out['predicts'].cpu().detach().numpy())
                onehot_labels.extend(batch_data['onehot_labels'].cpu().detach().numpy())
                num_samples += validation_out['tgt_size']

                self._val_iter += 1
                batch_data = self.valid_data_loader.next()
                self.call_hook('after_val_iter')
                self.valid_log_buffer.update(validation_out)
            except RuntimeError as e:
                clear_cuda_error()
                torch.cuda.empty_cache()
                logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                continue
        end_time = time.time()
        self.valid_log_buffer.update({'time': end_time - start_time})

        num_classes = self.solution_cfg.num_classes
        if self.world_size > 1:
            # gather num_samples from all ranks
            dist_samples = [0] * self.world_size
            dist_samples[self.rank] = num_samples
            dist_samples = torch.as_tensor(dist_samples, dtype=torch.int32, device='cuda')
            dist_allreduce(dist_samples, name='validation_samples_sum', op=ReduceOp.SUM)
            # sum with cur rank to avoid empty tensor
            before_rank_samples = torch.sum(dist_samples[: self.rank + 1]) - num_samples
            after_rank_samples = torch.sum(dist_samples[self.rank :]) - num_samples
            # gather predicts and labels from all rank
            dist_predicts = (
                [np.zeros(num_classes) for _ in range(before_rank_samples)]
                + predicts
                + [np.zeros(num_classes) for _ in range(after_rank_samples)]
            )
            dist_labels = (
                [np.zeros(num_classes) for _ in range(before_rank_samples)]
                + onehot_labels
                + [np.zeros(num_classes) for _ in range(after_rank_samples)]
            )
            dist_predicts = torch.as_tensor(dist_predicts, dtype=torch.float32, device='cuda')
            dist_labels = torch.as_tensor(dist_labels, dtype=torch.float32, device='cuda')
            dist_allreduce(dist_predicts, name='validation_predicts_sum', op=ReduceOp.SUM)
            dist_allreduce(dist_labels, name='validation_labels_sum', op=ReduceOp.SUM)
            predicts_array = dist_predicts.cpu().detach().numpy()
            labels_array = dist_labels.cpu().detach().numpy()
        else:
            predicts_array = np.array(predicts)
            labels_array = np.array(onehot_labels)

        m_ap, _ = compute_average_precision(predicts_array, labels_array, num_classes)
        m_auc, _ = compute_area_under_curve(predicts_array, labels_array, num_classes)
        eer, _ = compute_equal_error_rate(predicts_array, labels_array, num_classes)
        d_prime_val = compute_d_prime(m_auc)

        self.valid_log_buffer.update(
            {'mAP': m_ap, 'mAUC': m_auc, 'eer': eer, 'd_prime': d_prime_val}
        )

        self.log_metric(self.valid_log_buffer)
        self.after_validation()
