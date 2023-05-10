''' BaseSIDRunner '''

import os.path as osp
import torch
import numpy as np
import kaldiio
from core.extensions import clear_cuda_error
from core.dataset import (
    BalancedHDFSDataset,
    ValidHDFSDataset,
    get_meta,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.runner.metric.sid_metric import SidMetric
from core.utils import (
    dist_barrier,
    hdfs_put,
    hdfs_mkdir,
    dist_allreduce,
    ReduceOp,
    mkdir_or_exist,
    dist_file_get,
    get_world_size,
    get_rank,
)
from core.utils import logging
from core.runner.sid.eer.eer_test import EER
from ..utils import get_time
from ..base_runner import BaseRunner, RUNNERS

try:
    import falconclaw
except ImportError:
    falconclaw = None


@RUNNERS.register_module()
class BaseSidRunner(BaseRunner):
    '''SID Runner'''

    def __init__(self, cfg, inference=False, export_onnx=False):
        super().__init__(cfg, inference, export_onnx)
        self.report_message = {}

    def build_solution(self):
        self.solution_cfg.setdefault('is_inference', self.is_inference)
        self.solution_cfg.setdefault('is_export_onnx', self.is_export_onnx)
        self.solution_cfg.setdefault('iters_per_epoch', self.train_cfg.get("iters_per_epoch", 5000))
        super().build_solution()

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        # Ensure the first meta has the smallest indices
        self.get_data_list(dataset_cfg)
        self.solution_cfg.setdefault('fbank_dim', dataset_cfg.fbank_dim)
        self.solution_cfg.setdefault('iters_per_epoch', self.args.train.iters_per_epoch)
        self.solution_cfg.setdefault('used_epoch_eval', False)
        self.meta_data_list = [get_meta(meta_file) for meta_file in self.meta_file_list]
        for index, meta_data in enumerate(self.meta_data_list):
            logging.info(f"{len(meta_data['spk2idx'].keys())} speakers in the {index} dataset.")

        # Ensure the first meta has the smallest indices
        self.spk2idx = self.meta_data_list[0]['spk2idx']
        spk_index = len(self.spk2idx.keys())
        for meta_data in self.meta_data_list[1:]:
            for spk in meta_data['spk2idx']:
                if spk not in self.spk2idx:
                    self.spk2idx[spk] = spk_index
                    spk_index += 1
        self.meta_data = {
            'dicts': self.spk2idx,
        }
        if 'cmvn_mean' in self.meta_data_list[0]:
            self.meta_data['cmvn_mean'] = self.meta_data_list[0]['cmvn_mean']
            self.meta_data['cmvn_var'] = self.meta_data_list[0]['cmvn_var']
        logging.info(
            "{} speakers found when combining all the datasets.".format(len(self.spk2idx.keys()))
        )
        spk_expansion_times = 1
        item_transform_list = dataset_cfg.get('train_item_transform', [])
        for trans in item_transform_list:
            if trans['type'] == 'SpeedPerturbation':
                speed_rate_list = trans.get('speed_rate_list', [0.9, 1.1])
                cls_relabel = trans.get('cls_relabel', False)
                if cls_relabel:
                    spk_expansion_times *= len(speed_rate_list) + 1
        self.solution_cfg.setdefault(
            'tgt_vocab_size', spk_expansion_times * len(self.spk2idx.keys())
        )
        self.setup_transform_cfg(dataset_cfg)

    def get_data_list(self, dataset_cfg):
        '''get data list'''
        super().get_data_list(dataset_cfg)
        self.eval_file_list = []
        self.eval_file_nums = []
        data_root = dataset_cfg.get('data_root', None)
        self.solution_cfg.setdefault('used_epoch_eval', False)
        used_epoch_eval = self.args.solution.used_epoch_eval
        if used_epoch_eval:
            eval_data_roots = dataset_cfg.get('eval_data_root', data_root)
            eval_file_lists = dataset_cfg.get('eval_file_list', None)
            trial_file_lists = dataset_cfg.get('trials', 'trials')
            if isinstance(eval_data_roots, str):
                eval_data_roots = [eval_data_roots]
                eval_file_lists = [eval_file_lists]
                trial_file_lists = [trial_file_lists]
            if len(trial_file_lists) == 1:
                trial_file_lists = trial_file_lists * len(eval_data_roots)
            assert len(eval_data_roots) == len(eval_file_lists) == len(trial_file_lists)
            for idx, (eval_data_root, eval_file_list, trial_file_list) in enumerate(
                zip(eval_data_roots, eval_file_lists, trial_file_lists)
            ):
                now_eval_file_list = [osp.join(eval_data_root, p) for p in eval(eval_file_list)]
                self.eval_file_list.extend(now_eval_file_list)
                self.eval_file_nums.append(len(now_eval_file_list))
                remote_path = osp.join(eval_data_root, trial_file_list)
                dist_file_get(remote_path, local_file=f'trials{idx}')

            logging.info("The eval list: [ " + ",".join(self.eval_file_list) + " ]")

    def build_dataset(self, dataset_cfg):
        '''build data set
        This process is designed for train/valid and inference separately.
        '''
        super().build_dataset(dataset_cfg)
        if not self.need_build_data_loader:
            # If in inference mode, a totally different pipeline is build.
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        self.train_data_loader = BalancedHDFSDataset(
            self.train_file_list,
            dataset_cfg,
            self.train_item_trans,
            self.draw_train_batch_fn,
            device_transforms=self.train_device_trans,
        )

        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            '',
            dataset_cfg,
            self.valid_item_trans,
            self.draw_valid_batch_fn,
            split_path_list_by_rank=False,
            device_transforms=self.valid_device_trans,
        )

        self.test_eer_loader = None
        if self.args.solution.used_epoch_eval:
            eval_cfg = dataset_cfg.copy()
            eval_cfg['max_batch_size'] = 0
            parse_eval_cfg = dataset_cfg.eval_item_transform
            self.valid_item_trans = build_item_augmentation(parse_eval_cfg, self.meta_data)
            draw_eval_batch_cfg = dataset_cfg.eval_batch_transform
            self.draw_eval_batch_fn = build_draw_batch_fn(draw_eval_batch_cfg, self.meta_data)
            eval_device_trans_cfg = dataset_cfg.get('eval_device_transform', None)
            self.eval_device_trans = build_device_augmentation(
                eval_device_trans_cfg, self.meta_data
            )
            self.test_eer_loader = ValidHDFSDataset(
                self.eval_file_list,
                '',
                eval_cfg,
                self.valid_item_trans,
                self.draw_eval_batch_fn,
                split_path_list_by_rank=False,
                device_transforms=self.eval_device_trans,
                split_each_dataset=True,
            )

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = SidMetric()
        self.valid_log_buffer = SidMetric()

    def after_validation(self):
        if self.args.solution.used_epoch_eval:
            for idx, file_num in enumerate(self.eval_file_nums):
                eer, thresh, dcf_001, dcf_0001 = self.calculate_eer(f'trials{idx}', file_num)
                self.report_message[f'trilas{idx}_eer'] = eer
                logging.info("\n\n" + "#" * 200 + "\n\n")
                logging.info(
                    "rank %d eer: %f thresh: %f dcf_001: %f dcf_0001:%f",
                    self.rank,
                    eer,
                    thresh,
                    dcf_001,
                    dcf_0001,
                )
                logging.info("\n\n" + "#" * 200 + "\n\n")
        super().after_validation()

    def get_test_file_list(self):
        '''get test file list and output file list for inference'''
        if isinstance(self.inference_cfg.data_root, str):
            assert isinstance(self.inference_cfg.save_root, str) and isinstance(
                self.inference_cfg.remote_save_root, str
            )
            test_files = eval(self.inference_cfg.test_file_list)
            test_file_list = [self.inference_cfg.data_root + p for p in test_files]
            test_output_list = [self.inference_cfg.save_root + p for p in test_files]
            test_remote_output_list = [self.inference_cfg.remote_save_root + p for p in test_files]
        else:
            assert len(self.inference_cfg.data_root) == len(self.inference_cfg.test_file_list)
            assert len(self.inference_cfg.save_root) == len(self.inference_cfg.remote_save_root)
            assert len(self.inference_cfg.test_file_list) == len(self.inference_cfg.save_root)
            dataset_num = len(self.inference_cfg.data_root)
            test_file_list = []
            test_output_list = []
            test_remote_output_list = []
            for i in range(dataset_num):
                test_file_list.extend(
                    [
                        osp.join(self.inference_cfg.data_root[i], p)
                        for p in eval(self.inference_cfg.test_file_list[i])
                    ]
                )
                test_output_list.extend(
                    [
                        osp.join(self.inference_cfg.save_root[i], p)
                        for p in eval(self.inference_cfg.test_file_list[i])
                    ]
                )
                test_remote_output_list.extend(
                    [
                        osp.join(self.inference_cfg.remote_save_root[i], p)
                        for p in eval(self.inference_cfg.test_file_list[i])
                    ]
                )
        return test_file_list, test_output_list, test_remote_output_list

    def inference_once(self, test_file, output_file, remote_output_file):
        '''inference a test set'''
        min_chunk_size = self.inference_cfg.min_chunk_size
        max_chunk_size = self.inference_cfg.max_chunk_size
        chunk_shift = self.inference_cfg.chunk_shift
        logging.info("Extract embeddings from {}".format(test_file))
        test_data_loader = ValidHDFSDataset(
            [test_file],
            '',
            self.inference_cfg,
            self.parse_fn_inf,
            self.draw_inf_batch_fn,
            split_path_list_by_rank=False,
        )
        test_data_loader.reset()
        output_file = '{}_{}'.format(output_file, self.rank)
        remote_output_file = '{}_{}'.format(remote_output_file, self.rank)
        mkdir_or_exist(osp.dirname(output_file))
        hdfs_mkdir(osp.dirname(remote_output_file))

        with kaldiio.WriteHelper('ark,scp:{0}.ark,{0}.scp'.format(output_file)) as writer:
            batch_data = test_data_loader.next()
            # Support doing inference in a batch mode
            while batch_data is not None:
                # Split the data to feature and mask
                batch_size, feature_len, dim = batch_data['src'].shape
                curr_chunk_size = feature_len if feature_len <= max_chunk_size else max_chunk_size
                start_index = 0
                batch_index = []
                batch_data['feature'] = []
                batch_data['mask'] = []
                for i in range(batch_size):
                    length = int(batch_data['src_mask'][i].sum())
                    if length < min_chunk_size:
                        batch_index.append([-1])
                    elif length > max_chunk_size:
                        split_num = int(np.ceil(float(length - max_chunk_size) / chunk_shift) + 1)
                        for chunk in range(split_num):
                            start = chunk * chunk_shift
                            if chunk == split_num - 1:
                                # The last feature should be padded
                                batch_data['feature'].append(
                                    torch.cat(
                                        [
                                            batch_data['src'][i, start:length, :],
                                            torch.zeros(
                                                [max_chunk_size - length + start, dim]
                                            ).cuda(),
                                        ],
                                        dim=0,
                                    )
                                )
                                batch_data['mask'].append(
                                    torch.cat(
                                        [
                                            batch_data['src_mask'][i, start:length],
                                            torch.zeros([max_chunk_size - length + start]).cuda(),
                                        ],
                                        dim=0,
                                    )
                                )
                            else:
                                batch_data['feature'].append(
                                    batch_data['src'][i, start : start + max_chunk_size, :]
                                )
                                batch_data['mask'].append(
                                    batch_data['src_mask'][i, start : start + max_chunk_size]
                                )
                        batch_index.append(list(range(start_index, start_index + split_num)))
                        start_index += split_num
                    else:
                        batch_data['feature'].append(batch_data['src'][i, :curr_chunk_size, :])
                        batch_data['mask'].append(batch_data['src_mask'][i, :curr_chunk_size])
                        batch_index.append([start_index])
                        start_index += 1
                feature = torch.stack(batch_data['feature'])
                mask = torch.stack(batch_data['mask'])
                # del batch_data['src']
                # del batch_data['src_mask']
                embedding = self.solution.inference(feature, mask)

                # Recover the embedding for each utterance
                for i in range(batch_size):
                    if batch_index[i][0] == -1:
                        # Skip this utterance since the length is too short.
                        continue
                    chunk_size = torch.sum(mask[batch_index[i]], 1, keepdim=True)
                    average_embedding = (
                        torch.sum(chunk_size * embedding[0][batch_index[i]], dim=0)
                        / chunk_size.sum()
                    )
                    writer(batch_data['uttid'][i], average_embedding.cpu().numpy())
                # next batch
                batch_data = test_data_loader.next()
        test_data_loader.terminate()
        hdfs_put('{}.ark'.format(output_file), '{}.ark'.format(remote_output_file))
        hdfs_put('{}.scp'.format(output_file), '{}.scp'.format(remote_output_file))
        # wait all rank finish
        dist_barrier()

    @torch.no_grad()
    def inference(self):
        '''extract embedding from the node'''
        self.inference_cfg = self.args.inference
        self.parse_fn_inf = build_item_augmentation(
            self.inference_cfg.inference_item_transform, self.meta_data
        )
        self.draw_inf_batch_fn = build_draw_batch_fn(
            self.inference_cfg.inference_batch_transform, self.meta_data
        )
        test_file_list, test_output_list, test_remote_output_list = self.get_test_file_list()
        self.solution.eval()
        self.solution.embedding_nodes = [self.inference_cfg.inference_node_name]
        for test_file, output_file, remote_output_file in zip(
            test_file_list, test_output_list, test_remote_output_list
        ):
            self.inference_once(test_file, output_file, remote_output_file)

    def calculate_eer(self, trial_path, sub_dataset_num=0):
        # pylint: disable=not-callable
        '''
        calculate eer
        '''
        index = []
        eer_class = EER()
        trial_data, utt_num = eer_class.read_trials(trial_path)
        assert utt_num, "utt_num should not be equal to 0"
        # looks like some skips will influence increasing the tensor, so
        # give some redundancies.
        utt_num += utt_num
        embedding_tensor = torch.zeros(
            [utt_num, self.args.solution.embedding_dim], dtype=torch.float32, device='cuda'
        )

        utt_name_tensor = torch.zeros([utt_num, 100], dtype=torch.int64, device='cuda')
        world_size = get_world_size()
        rank = get_rank()
        index = rank * (utt_num // world_size)

        # TODO: at present, only sub_dataset_num equal to 1 is supported
        for _ in range(sub_dataset_num):
            self.test_eer_loader.reset()
            batch_data = self.test_eer_loader.next()
            while batch_data is not None:
                try:
                    embedding = self.get_embedding(batch_data)
                    embedding_tensor[index] = embedding
                    utt = batch_data['utt'][0]
                    utt = torch.as_tensor(
                        self.get_list_by_str(utt), dtype=torch.int64, device='cuda'
                    )
                    utt_name_tensor[index][: utt.shape[0]] = utt
                    batch_data = self.test_eer_loader.next()
                    index += 1

                except RuntimeError as e:
                    clear_cuda_error()
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
        dist_allreduce(utt_name_tensor, name='reduce_utt', op=ReduceOp.SUM)
        dist_allreduce(embedding_tensor, name='reduce_embedding', op=ReduceOp.SUM)
        embedding_dict = {}
        utt_name_tensor = utt_name_tensor.cpu()
        for idx in range(utt_name_tensor.shape[0]):
            utt_name_value = self.get_str_by_list(utt_name_tensor[idx])
            embedding_value = embedding_tensor[idx].cpu().detach().numpy()
            if len(utt_name_value) != 0:
                embedding_dict[utt_name_value] = embedding_value
            else:
                continue

        scores = eer_class.get_scores(trial_data, embedding_dict)
        labels = np.array([pairs[2] for pairs in trial_data])
        return eer_class.calculate_evaluation(labels, scores)

    def get_embedding(self, batch_data):
        # pylint: disable=protected-access
        '''get embedding node'''
        if self.solution.embedding_nodes:
            self.solution.embedding_nodes = [self.args.solution.embedding_layer]
        feature = batch_data['feature']
        mask = batch_data['mask']
        _, embedding, _ = self.solution._forward_impl(feature, mask.float())
        utt_output = embedding[self.args.solution.embedding_layer]
        embedding = utt_output.squeeze(0)
        return embedding

    @staticmethod
    def get_list_by_str(string):
        '''get list by str'''
        str_list = []
        for v in string:
            str_list.append(ord(v))
        return str_list

    @staticmethod
    def get_str_by_list(str_list):
        '''get str by list'''
        string = ""
        for v in str_list:
            if isinstance(v, torch.Tensor):
                v = v.item()
            if v == 0:
                continue
            string += chr(v)
        return string

    @get_time('time')
    def train_iteration(self):
        '''train iteration.'''
        i = 0
        while i < self.grad_accum_step:
            batch_data = self.next_train_batch()
            try:
                self.solution.train()
                batch_data['loss_scale'] = self.dist_handler.get_scale(scaler_idx=0)
                solution_out = self.solution(batch_data, step=self.iter)
                self.loss = solution_out['backward_loss'] / self.grad_accum_step
                self.dist_handler.backward(self.loss, unscale=(i + 1 == self.grad_accum_step))
                if self.train_cfg.get('use_regularization', None):
                    regular_loss = 0.0
                    for name, weight in self.solution.named_parameters():
                        if self.train_cfg.use_regularization == 'L2':
                            regular_loss += ((weight - self.ckp_weight[name]) ** 2).sum()
                    self.loss += self.train_cfg.get('regularization_weight', 1.0) * regular_loss
                    solution_out['regular_loss'] = regular_loss

                if self.solution_cfg.get('use_distiller', None):
                    distiller_loss = self.distiller.caculate_distill_loss()
                    self.loss += distiller_loss
                    solution_out['distiller_loss'] = distiller_loss
            except RuntimeError as e:
                self.handle_error(e, batch_data)
                continue
            self.train_log_buffer.update(solution_out)
            i = i + 1
        if self.opt_util_cfg.grad_clip:
            gnorm = self.clip_grads()
            self.train_log_buffer.update({'gnorm': gnorm})
        self.dist_handler.step(iters=self.iter)

    def run(self):
        super().run()
        if self.falcon_report:
            falconclaw.log_report(self.report_message)
