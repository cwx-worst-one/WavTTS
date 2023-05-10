''' EmotionRecognitionRunner '''
import os.path as osp
from collections import OrderedDict
import re
import copy
import numpy as np
import torch
from core.runner.metric.asr_metric import AsrMetric
from core.runner.metric.aed_metric import EmotionMetric
from core.dataset import (
    HDFSDataset,
    BalancedHDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.runner.asr.base_rnnt_runner import BaseRNNTRunner
from core.utils import (
    compute_emotion_metrics,
    compute_dimemotion_metrics,
    hdfs_put,
    hdfs_mkdir,
    logging,
    mkdir_or_exist,
    dist_allreduce,
    ReduceOp,
)
from core.models.pretrained.bert_utils import build_bert_vocab
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class EmotionRecognitionRunner(BaseRNNTRunner):
    '''EmotionRecognitionRunner'''

    # pylint: disable=too-many-public-methods

    def __init__(self, cfg, inference=False, export_onnx=False):
        super().__init__(cfg, inference=inference, export_onnx=export_onnx)
        self.report_key_list.extend(
            self.solution_cfg.get('metrics', ['CCC', 'RMSE', 'UA', 'UF', 'WP', 'WA', 'WF'])
        )

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transform config.'''
        transform_cfgs = (
            dataset_cfg.train_item_transform,
            dataset_cfg.valid_item_transform,
        )
        for transform_cfg in transform_cfgs:
            for cfg in transform_cfg:
                if cfg.type == 'Textchar2Index':
                    cfg['vocab_dict'] = self.bert_text_lexicon
                    cfg['punc_table'] = self.bert_punc_table
                    cfg['char2index'] = self.bert_char2index

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        # batch balance parms
        if self.dataset_cfg.get('meta_file', None) is not None:
            super().pre_build_dataset(dataset_cfg)

        # save dirs
        self.remote_save_root = None
        if dataset_cfg.get('emotion_keys', ''):
            emotion_keys = dataset_cfg.emotion_keys.split(',')
            assert '' not in emotion_keys, 'Illegal format in emotion_keys argument'
            self.load_emotion_keys_parameters(
                dataset_cfg.train_item_transform, 'EmotionLabelParser', emotion_keys
            )
            self.load_emotion_keys_parameters(
                dataset_cfg.valid_item_transform, 'EmotionLabelParser', emotion_keys
            )
            self.load_emotion_keys_parameters(
                dataset_cfg.batch_transform, 'EmotionFeatCollate', emotion_keys, pad=-100
            )
            self.load_emotion_keys_parameters(
                dataset_cfg.inference_batch_transform,
                'EmotionFeatCollate',
                emotion_keys,
                pad=-100,
            )
            dataset_cfg.pop('emotion_keys')
        if dataset_cfg.get('list_keys', ''):
            list_keys = dataset_cfg.list_keys.split(',')
            assert '' not in list_keys, 'Illegal format in list_keys argument'
            self.load_emotion_keys_parameters(dataset_cfg.batch_transform, 'ListCollate', list_keys)
            self.load_emotion_keys_parameters(
                dataset_cfg.inference_batch_transform, 'ListCollate', list_keys
            )
            dataset_cfg.pop('list_keys')
        if self.train_cfg.get('remote_save_root', None):
            self.remote_save_root = osp.join(
                self.train_cfg.remote_save_root,
                self.train_cfg.save_dir,
                self.train_cfg.save_name,
                'checkpoints',
            )
            logging.all_rank_info(
                'rank %d: save remote checkpoint to %s/%s ',
                self.rank,
                self.remote_save_root,
                'latest',
            )
        self.out_dir = osp.join(
            self.train_cfg.save_root,
            self.train_cfg.save_dir,
            self.train_cfg.save_name,
            'checkpoints',
        )
        self.onnx_out_dir = osp.join(
            self.train_cfg.save_root,
            self.train_cfg.save_dir,
            self.train_cfg.save_name,
            'onnx',
            'origin_onnx',
        )
        # dim emotion
        self.dimemotion = False
        if self.solution_cfg.get('dim_emotion', None) is not None:
            self.dimemotion = True
            # [2, 1, 0, -1, -2]
            self.dim2ctg = self.solution_cfg.dim2ctg.split('|')
            self.dim2ctg = [float(i) for i in self.dim2ctg]
            self.dim2ctgarray = [0, 0]
            for i in range(len(self.dim2ctg)):
                self.dim2ctgarray.extend(
                    [self.dim2ctg[i] for j in range(self.solution_cfg.phone_num)]
                )
            self.dim2ctgarray = np.array(self.dim2ctgarray)
        if self.solution_cfg.get('dim_tgt_size', None) is None:
            self.solution_cfg.dim_tgt_size = self.solution_cfg.tgt_size
            self.solution_cfg.dim_emotion_num = self.solution_cfg.emotion_num
            self.solution_cfg.dim_neutral_label = self.solution_cfg.neutral_label

        # model para
        self.solution_cfg.freeze_encoder = self.solution_cfg.get('freeze_encoder', False)

        # save results
        self.latest_results = {'best_results': {}}
        predict_sets = self.solution_cfg.get('predict_sets', 'emotion_phone')
        self.predict_sets = predict_sets.strip().split('|')
        labels_sets = self.solution_cfg.get('labels_sets', 'emotion_phone')
        self.labels_sets = labels_sets.strip().split('|')
        self.metrics = self.solution_cfg.get(
            'metrics', ['CCC', 'RMSE', 'UA', 'UF', 'WP', 'WA', 'WF']
        )
        bert_vocab_file = self.solution_cfg.get('bert_vocab_dict')
        bert_filter_punc = self.solution_cfg.get('bert_filter_punc', False)
        self.bert_char2index = self.solution_cfg.get('bert_char2index', True)
        self.bert_text_lexicon, self.bert_punc_table = build_bert_vocab(
            bert_vocab_file, self.bert_char2index, bert_filter_punc
        )
        self.setup_transform_cfg(dataset_cfg)

    @staticmethod
    def get_bert_layer(param_name):
        """Get the layer of variable in BERT transformer."""
        layer_num = re.match(r'bert_model\.encoder.layer\.(.*?)\..*', param_name)
        emb = re.match(r'bert_model.embeddings', param_name)
        if layer_num is not None:
            return int(layer_num.group(1)) + 1
        if emb is not None:
            return 0
        return -1

    @staticmethod
    def get_layer_wise_lr(current_layer, lr, coe, max_layer):
        """lr[k-1] = lr[k] * coe"""
        if current_layer < 0:
            # no decay for non-transformer layer
            return lr
        return lr * (coe ** (max_layer - current_layer))

    def build_solution(self):
        '''build_solution.'''
        # pylint:disable=too-many-branches
        super().build_solution()
        # modify the optimizer lr

        # generate multi lr
        if (
            self.optimizer_cfg.get('w2v_lr', None) is not None
            and self.optimizer_cfg.get('bert_lr', None) is not None
            and self.optimizer_cfg.get('classifier_lr', None) is not None
        ):
            self.optimizer_cfg['lr'] = []
            self.optimizer_cfg['lr'].append([self.optimizer_cfg.w2v_lr, ['w2v_model']])
            self.optimizer_cfg['lr'].append([self.optimizer_cfg.bert_lr, ['bert_model']])
            self.optimizer_cfg['lr'].append(
                [self.optimizer_cfg.classifier_lr, ['classifier', 'linear']]
            )

        # add lr_layer_wise_decay_rate for bert model, revised from https://code.byted.org/\
        # lab-speech/bytebot-nlu/blob/HEAD/optimizers/optimization.py
        coe = self.optimizer_cfg.get('lr_layer_wise_decay_rate', 1)
        if coe != 1:
            # get bert_model lr and make changes
            bert_lr = 1e-5
            new_optimizer_cfg_lr = []
            if isinstance(self.optimizer_cfg.get('lr', 0), list):
                for lr_list in self.optimizer_cfg['lr']:
                    new_lr_name = []
                    for prefix in lr_list[1]:
                        if 'bert' in prefix:
                            bert_lr = lr_list[0]
                        else:
                            new_lr_name.append(prefix)
                    if len(new_lr_name) != 0:
                        new_optimizer_cfg_lr.append([lr_list[0], new_lr_name])
            else:
                self.optimizer_cfg['lr_base'] = self.optimizer_cfg['lr']
                bert_lr = self.optimizer_cfg['lr']

            self.optimizer_cfg['lr'] = new_optimizer_cfg_lr

            max_layer = self.solution_cfg.get('num_hidden_layers', 6)
            layer_dict = {}
            for name, _ in self.solution.named_parameters():
                if 'bert' in name:
                    layer_num = str(self.get_bert_layer(name))
                    if layer_num in layer_dict:
                        layer_dict[layer_num].append(name)
                    else:
                        layer_dict[layer_num] = [name]

            for layer_num, layer in layer_dict.items():
                lr = self.get_layer_wise_lr(int(layer_num), bert_lr, coe, max_layer)
                self.optimizer_cfg['lr'].append([lr, layer])

        # check layers without lr
        if isinstance(self.optimizer_cfg.get('lr', 0), list):
            name_list = []
            for lr_list in self.optimizer_cfg['lr']:
                name_list += [
                    name
                    for name, _ in self.solution.named_parameters()
                    for prefix in lr_list[1]
                    if prefix in name
                ]

            if not len(list(self.solution.parameters())) == len(name_list):
                undefined_list = []
                for name, _ in self.solution.named_parameters():
                    if name_list.count(name) == 0:
                        undefined_list.append(name)
                self.optimizer_cfg['lr'].append(
                    [self.optimizer_cfg.get('lr_base', 1e-5), undefined_list]
                )

        # pop useless param
        for name in ['lr_layer_wise_decay_rate', 'bert_lr', 'lr_base', 'w2v_lr', 'classifier_lr']:
            if name in self.optimizer_cfg:
                self.optimizer_cfg.pop(name)

    def _resume_from_pretrain(
        self,
        pretrain_chkpt,
        checkpoint_dir,
    ):
        '''try resume from ${resume_pretrain_chkpt}'''

        if not pretrain_chkpt:
            return False
        pretrain_chkpts = pretrain_chkpt.split('|')
        for _pretrain_chkpt in pretrain_chkpts:
            if not self._resume_from_hdfs_path(
                _pretrain_chkpt,
                checkpoint_dir,
                resume_optimizer=False,
                resume_progress=False,
                resume_lr_scheduler=False,
                resume_amp=False,
            ):
                return False

        logging.info('rank %d: resume from pretrain checkpoint', self.rank)
        return True

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        # support modify train_file_list and valid_file_list
        for run_file in ['train_file', 'valid_file']:
            if dataset_cfg.get(run_file, None) is not None:
                dataset_cfg[run_file + '_list'][0] = '["' + dataset_cfg[run_file] + '"]'
                logging.info(
                    'the new ' + run_file + '_list is %s', dataset_cfg[run_file + '_list'][0]
                )

        self.pre_build_dataset(dataset_cfg)

        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)

        if self.is_inference or self.is_export_onnx:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        parse_cfg = dataset_cfg.train_item_transform
        self.parse_fn = build_item_augmentation(parse_cfg, self.meta_data)
        draw_batch_cfg = dataset_cfg.batch_transform
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)
        device_trans_cfg = dataset_cfg.get('train_device_transform', None)
        self.device_trans = build_device_augmentation(device_trans_cfg, self.meta_data)

        self.get_data_list(dataset_cfg)
        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule
        use_balance = dataset_cfg.get("balance", False)
        if use_balance:
            self.train_data_loader = BalancedHDFSDataset(
                self.train_file_list,
                dataset_cfg,
                self.parse_fn,
                self.draw_batch_fn,
                device_transforms=self.device_trans,
            )
        else:
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

    @staticmethod
    def load_emotion_keys_parameters(cfg, task_type, emotion_keys, **kwargs):
        '''load console parameters into cfg.'''
        for key in emotion_keys:
            if all(op.get('key', '') != key for op in cfg):
                d = dict(type=task_type, key=key, **kwargs)
                cfg.append(d)

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = AsrMetric()
        self.valid_log_buffer = EmotionMetric()

    def update_best_metric(self):
        '''Update best metric to save checkpoint'''
        cur_metric = self.valid_log_buffer.get_value(self.train_cfg.get('compare_metric', 'loss'))
        if cur_metric == 0.0:
            logging.error("rank %d catch cur metric is 0.0, not update best metric", self.rank)
            return False
        if self._best_metric is None or cur_metric > self._best_metric:
            self._best_metric = cur_metric
            return True
        return False

    def ctg_label_phone2utt(self, nums, unpred_label):
        '''transform phone emotion to utterance category emotion'''
        new_nums = [nums[0]] if nums[0] >= self.solution_cfg.phone_tag else []
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] and nums[i] >= self.solution_cfg.phone_tag:
                new_nums.append(nums[i])

        nums = [
            (num - self.solution_cfg.phone_tag) // self.solution_cfg.phone_num for num in new_nums
        ]
        if len(nums) == 0:
            return unpred_label
        return max(nums, key=nums.count)

    def dim_label_phone2utt(self, dim_preds):
        '''transform phone emotion to utterance dimentional emotion'''

        new_dim_pred = np.array(
            [dim_pred for dim_pred in dim_preds if dim_pred.argmax() >= self.solution_cfg.phone_tag]
        )
        if len(new_dim_pred) == 0:
            return 0

        return (self.dim2ctgarray * new_dim_pred).sum() / new_dim_pred.shape[0]

    def ctg_compute(self, preds, labels, neutral_label):
        '''compute predicted category emotion labels'''
        preds = np.array([np.argmax(pred, axis=-1) for pred in preds], dtype='object')
        unpred_label = neutral_label
        labels = np.array([self.ctg_label_phone2utt(label, unpred_label) for label in labels])
        preds = np.array([(self.ctg_label_phone2utt(pred, unpred_label)) for pred in preds])

        return preds, labels

    def dim_compute(self, preds, labels):
        '''compute predicted dimentional emotion labels'''
        preds = np.array([(self.dim_label_phone2utt(pred)) for pred in preds])
        return preds, np.array(labels)

    def get_category_results(
        self, preds, labels, uttid, infos=None, show_log=True, tag='', pre_compute=True
    ):
        '''get predicted emotion results'''
        # pylint:disable=too-many-branches
        if tag[-1] in ['p', 'a', 'd']:
            emotion_num = self.solution_cfg.dim_emotion_num
            neutral_label = self.solution_cfg.dim_neutral_label
        else:
            emotion_num = self.solution_cfg.emotion_num
            neutral_label = self.solution_cfg.neutral_label
        if self.args.inference.get('pred_emotion_devide_2', False):
            preds = (preds / 2).astype('int32')
            labels = (labels / 2).astype('int32')
            neutral_label = int(neutral_label / 2)
            emotion_num = int(emotion_num / 2)
        elif self.args.inference.get('pred_emotion_plus_one_devide_2', False):
            preds = ((preds + 1) / 2).astype('int32')
            labels = ((labels + 1) / 2).astype('int32')
            neutral_label = int((neutral_label + 1) / 2)
            emotion_num = int((emotion_num + 1) / 2)
        if pre_compute:
            preds, labels = self.ctg_compute(preds, labels, neutral_label)

        if self.args.inference.get('save_results', False):
            save_dict = {"preds": preds, "labels": labels, "uttid": uttid, "infos": infos}
            self.save_results(tag, 'ctg_result.npy', save_dict)

        if len(labels) != 0:
            if self.dimemotion:
                compute_dimemotion_metrics(
                    preds,
                    labels,
                    show_log=False,
                )
            results = compute_emotion_metrics(
                predict=np.eye(emotion_num)[preds],
                label=np.eye(emotion_num)[labels],
                show_log=show_log,
                mode=1,
            )  # mode =1 : output confusion_matrix

            confusion_matrix = torch.from_numpy(results).to('cuda')

            dist_allreduce(confusion_matrix, name='confusion_matrix' + tag, op=ReduceOp.SUM)
            if self.rank != 0:
                show_log = False
            results = compute_emotion_metrics(
                confusion_matrix=confusion_matrix.detach().cpu().numpy(),
                show_log=show_log,
                mode=2,
            )
            self.update_results(results, tag)

    def get_dimension_results(
        self, preds, labels, uttid, infos=None, show_log=True, tag='', pre_compute=True
    ):
        '''get predicted emotion results'''
        if pre_compute:
            preds, labels = self.dim_compute(preds, labels)
        if self.args.inference.get('save_results', False):
            save_dict = {"preds": preds, "labels": labels, "uttid": uttid, "infos": infos}
            self.save_results(tag, 'dim_result.npy', save_dict)

        if len(labels) != 0:
            if self.rank != 0:
                show_log = False
            results = compute_dimemotion_metrics(
                preds,
                labels,
                show_log,
            )
            results = torch.from_numpy(np.array([results['CCC'], results['RMSE']])).to('cuda')

            dist_allreduce(results, name='ccc_rmse' + tag, op=ReduceOp.AVG)
            results = results.detach().cpu().numpy()
            self.update_results({'CCC': results[0], 'RMSE': results[1]}, tag)

    def save_results(self, tag, save_name, save_dict):
        '''save results local and hdfs'''
        out_dir = osp.join(self.out_dir, tag, str(self.rank))
        mkdir_or_exist(out_dir)
        save_file = osp.join(out_dir, save_name)
        np.save(save_file, save_dict)
        if self.args.inference.get('save_remote', False):
            remote_save_dir = osp.join(self.remote_save_root, tag, str(self.rank))
            hdfs_mkdir(remote_save_dir)
            hdfs_put(save_file, remote_save_dir)

    @staticmethod
    def update_best_results(best_result, new_result):
        '''update best results'''
        for metric_tag in new_result:
            if metric_tag in best_result:
                if metric_tag == 'RMSE':
                    best_result[metric_tag] = min(
                        best_result[metric_tag], round(new_result[metric_tag], 4)
                    )
                else:
                    best_result[metric_tag] = max(
                        best_result[metric_tag], round(new_result[metric_tag], 4)
                    )
            else:
                best_result[metric_tag] = round(new_result[metric_tag], 4)
        return best_result

    def update_results(self, results, predict_tag):
        '''update latest results'''
        if predict_tag not in self.latest_results:
            self.latest_results[predict_tag] = {}
            self.latest_results['best_' + predict_tag] = {}
        for metric_tag in results:
            self.latest_results[predict_tag][metric_tag] = round(results[metric_tag], 4)
        self.latest_results['best_' + predict_tag] = self.update_best_results(
            self.latest_results['best_' + predict_tag], results
        )
        self.latest_results['best_results'] = self.update_best_results(
            self.latest_results['best_results'], results
        )

    def analysis_results(self, forward_out, uttid, infos=None):
        '''analysis different emotion results'''
        for predict_tag in self.predict_sets:
            if predict_tag in ['original_p', 'original_a', 'original_d', 'original', 'speaker']:
                self.get_dimension_results(
                    np.array(forward_out['preds_' + predict_tag]),
                    np.array(forward_out['labels_' + predict_tag]),
                    uttid,
                    infos=infos,
                    tag=predict_tag,
                    pre_compute=False,
                )
            elif predict_tag in ['emotion_p', 'emotion_a', 'emotion_d', 'emotion']:
                self.get_category_results(
                    np.array(
                        [np.argmax(pred, axis=-1) for pred in forward_out['preds_' + predict_tag]]
                    ),
                    np.array(forward_out['labels_' + predict_tag]),
                    uttid,
                    infos=infos,
                    tag=predict_tag,
                    pre_compute=False,
                )
                original_predict_tag = predict_tag.replace('emotion_phone', 'original').replace(
                    'emotion', 'original'
                )
                if 'labels_' + original_predict_tag in forward_out:
                    preds = np.array(forward_out['preds_' + predict_tag])
                    preds = preds * np.array(self.dim2ctg)
                    preds = preds.sum(-1)
                    self.get_dimension_results(
                        preds,
                        forward_out['labels_' + original_predict_tag],
                        uttid,
                        infos=infos,
                        tag=predict_tag,
                        pre_compute=False,
                    )
            elif predict_tag in [
                'emotion_phone_p',
                'emotion_phone_a',
                'emotion_phone_d',
                'emotion_phone',
            ]:
                self.get_category_results(
                    forward_out['preds_' + predict_tag],
                    forward_out['labels_' + predict_tag],
                    uttid,
                    infos=infos,
                    tag=predict_tag,
                )
                original_predict_tag = predict_tag.replace('emotion_phone', 'original').replace(
                    'emotion', 'original'
                )
                if 'labels_' + original_predict_tag in forward_out:
                    self.get_dimension_results(
                        forward_out['preds_' + predict_tag],
                        forward_out['labels_' + original_predict_tag],
                        uttid,
                        infos=infos,
                        tag=predict_tag,
                    )
            elif predict_tag in ['a_emotion', 't_emotion', 'm_emotion', 'mix_emotion', 'fbank']:
                if self.solution_cfg.get('tf2torch', False):
                    new_label = [1, 3, 5, 4, 2, 0]
                    forward_out['labels_emotion'] = [
                        new_label[label] for label in forward_out['labels_emotion']
                    ]
                if self.solution_cfg.get('tf2torch2', False):
                    new_label = [0, 2]
                    forward_out['labels_emotion'] = [
                        new_label[label] for label in forward_out['labels_emotion']
                    ]
                if self.solution_cfg.get('tf2torch3', False):
                    new_label = [0, 2, 3]
                    forward_out['labels_emotion'] = [
                        new_label[label] for label in forward_out['labels_emotion']
                    ]
                self.get_category_results(
                    np.array(
                        [np.argmax(pred, axis=-1) for pred in forward_out['preds_' + predict_tag]]
                    ),
                    np.array(forward_out['labels_emotion']),
                    uttid,
                    infos=infos,
                    tag=predict_tag,
                    pre_compute=False,
                )
            elif predict_tag in ['mm_emotion']:
                self.get_category_results(
                    forward_out['preds_' + predict_tag],
                    forward_out['labels_emotion_phone'],
                    uttid,
                    infos=infos,
                    tag=predict_tag,
                )

    @torch.no_grad()
    def validation(self):
        '''valid'''
        # pylint:disable=too-many-branches
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self._val_iter = 0
        if self.solution_cfg.get('valid_multi_cer', False):
            dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
        else:
            dataset_names = [None]
        for dataset_name in dataset_names:

            forward_out = OrderedDict()
            for predict_tag in self.predict_sets:
                forward_out['preds_' + predict_tag] = []
            for label_tag in self.labels_sets:
                forward_out['labels_' + label_tag] = []
            uttid = []
            context = []

            self.valid_log_buffer.reset()
            batch_data = self.valid_data_loader.next()
            validation_out = OrderedDict()
            while batch_data is not None:
                self.call_hook('before_val_iter')
                try:
                    self.solution.eval()
                    validation_out = self.solution.forward(batch_data)

                    uttid.extend(batch_data['uttid'])
                    if 'context' in batch_data:
                        context.extend(batch_data['context'])

                    for predict_tag in self.predict_sets:
                        forward_out['preds_' + predict_tag].extend(
                            validation_out[predict_tag].detach().cpu().numpy()
                        )

                    for label_tag in self.labels_sets:
                        if isinstance(batch_data[label_tag], list):
                            forward_out['labels_' + label_tag].extend(batch_data[label_tag])
                        else:
                            forward_out['labels_' + label_tag].extend(
                                batch_data[label_tag].detach().cpu().numpy()
                            )

                    self._val_iter += 1
                    batch_data = self.valid_data_loader.next()
                    self.call_hook('after_val_iter')
                    self.valid_log_buffer.update(validation_out)
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            self.analysis_results(forward_out, uttid, infos={'context': context})

            validation_tag = self.solution_cfg.get('validation_set', self.predict_sets[-1])
            if validation_tag in self.latest_results:
                for metric in self.latest_results[validation_tag]:
                    validation_out[metric] = self.latest_results[validation_tag][metric]

            self.valid_log_buffer.update(validation_out)
            self.log_metric(self.valid_log_buffer, name=dataset_name)
            if self.args.valid.get('save_latest_results', False):
                self.save_results(
                    'latest_results', 'latest_results.npy', {"latest_results": self.latest_results}
                )
            if self.args.valid.get('show_latest_results', True):
                logging.info(str(self.latest_results['best_results']))

        self.after_validation()

    def build_test_dataset(self, test_file):
        '''build test dataset'''
        inf_bucket_schedule = self.dataset_cfg.get('bucket_schedule', '')

        self.test_data_loader = ValidHDFSDataset(
            [test_file],
            inf_bucket_schedule,
            self.dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn_inference,
            split_path_list_by_rank=False,
        )
        self.test_data_loader.reset()

    @torch.no_grad()
    def inference_once(self, test_file):
        '''inference for one test dataset'''
        # pylint:disable=too-many-branches
        self.solution.eval()
        self.build_test_dataset(test_file)

        batch_data = self.test_data_loader.next()

        forward_out = OrderedDict()
        for predict_tag in self.predict_sets:
            forward_out['preds_' + predict_tag] = []
        for label_tag in self.labels_sets:
            forward_out['labels_' + label_tag] = []
        uttid = []
        text_id = []
        context = []
        onnx_final_vesion = self.solution_cfg.get('onnx_final_vesion', True)

        infer_only = self.args.inference.get('inference_only', False)

        # onnx_out_dir is used when test onnx
        onnx_out_dir = self.args.inference.get('onnx_out_dir', None)
        if onnx_out_dir is not None:
            self.solution.get_onnx(onnx_out_dir)

        while batch_data is not None:
            if onnx_out_dir is not None:
                batch_data1 = copy.deepcopy(batch_data)
                validation_out1 = self.solution.test_export_onnx_batch(batch_data1)
                for predict_tag in validation_out1:
                    if 'onnx_' + predict_tag not in forward_out:
                        forward_out['onnx_' + predict_tag] = []
                    forward_out['onnx_' + predict_tag].extend(validation_out1[predict_tag].tolist())
                if onnx_final_vesion and 'onnx_emotion' in forward_out:
                    forward_out['onnx_m_emotion'] = forward_out['onnx_emotion']
                if 'text' in batch_data:
                    text_id.extend(batch_data['text'])
            if self.args.inference.get('label_emotion_devide_2', False):
                batch_data['emotion'] = (
                    (np.array(batch_data['emotion']) / 2).astype('int32').tolist()
                )
            elif self.args.inference.get('label_emotion_plus_one_devide_2', False):
                batch_data['emotion'] = (
                    ((np.array(batch_data['emotion']) + 1) / 2).astype('int32').tolist()
                )

            validation_out = self.solution.forward(batch_data)

            uttid.extend(batch_data['uttid'])
            if 'context' in batch_data:
                context.extend(batch_data['context'])

            for predict_tag in self.predict_sets:
                forward_out['preds_' + predict_tag].extend(
                    validation_out[predict_tag].detach().cpu().numpy()
                )

            if not infer_only:
                for label_tag in self.labels_sets:
                    if isinstance(batch_data[label_tag], list):
                        forward_out['labels_' + label_tag].extend(batch_data[label_tag])
                    else:
                        forward_out['labels_' + label_tag].extend(
                            batch_data[label_tag].detach().cpu().numpy()
                        )
            batch_data = self.test_data_loader.next()

        self.test_data_loader.terminate()

        self.analysis_results(forward_out, uttid, infos={'context': context})

        # do some print and save for onnx test
        if onnx_out_dir is not None:
            for predict_tag in self.predict_sets:
                for i in range(len(forward_out['preds_' + predict_tag])):
                    print(
                        predict_tag,
                        uttid[i],
                        forward_out['preds_' + predict_tag][i],
                        forward_out['onnx_' + predict_tag][i],
                    )
            if onnx_final_vesion:
                save_dict = {'uttid': uttid, 'text_id': text_id, 'context': context}
                for predict_tag in self.predict_sets:
                    save_dict['preds_' + predict_tag] = forward_out['preds_' + predict_tag]
                    save_dict['onnx_' + predict_tag] = forward_out['onnx_' + predict_tag]
                if not infer_only:
                    for label_tag in self.labels_sets:
                        save_dict['labels_' + label_tag] = forward_out['labels_' + label_tag]
                np.save('onnx_data.npy', save_dict)
            else:
                for predict_tag in self.predict_sets:
                    forward_out['preds_' + predict_tag] = forward_out['onnx_' + predict_tag]
                self.analysis_results(forward_out, uttid)

        if self.args.inference.get('save_original_results', False):
            self.save_results(
                'result_original',
                'result_original.npy',
                {
                    "result_original": forward_out,
                    "uttid": uttid,
                    "latest_results": self.latest_results,
                },
            )

    @torch.no_grad()
    def inference(self):
        '''inference main function'''
        test_sets = self.args.inference.test_sets.strip().split('|')
        test_file_path = self.get_test_file_list(self.dataset_cfg, test_sets)
        # inference every test set
        inf_draw_batch_cfg = self.dataset_cfg.inference_batch_transform
        self.draw_batch_fn_inference = build_draw_batch_fn(inf_draw_batch_cfg)
        for test_file in test_file_path:
            self.inference_once(test_file)
