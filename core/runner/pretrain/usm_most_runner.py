''' USMMostRunner '''

import time
import copy
import torch
from core.utils import logging
from core.runner.utils import get_time
from core.extensions import clear_cuda_error
from core.runner.asr.rnnt_runner import RNNTRunner
from core.runner.metric.asr_metric import MostMetric
from ..base_runner import RUNNERS

try:
    from byteslim.quant.post_quant.utils.get_loss import get_kurt_loss
except ImportError:
    # To use byteslim, include byteslim scm at task building stage.
    # For more detail, see dolphin tutorial.
    # An exception will be raised if byteslim is used.
    pass


class MultiModalMultiTaskDataloader:
    '''Semi-supervised dataloader for Multimodal Multitask training.'''

    def __init__(self):
        '''init'''
        self.loader_map = {}

    def add(self, name, loader):
        '''register data loader to map'''
        if name in self.loader_map:
            raise RuntimeError(f'{name} loader exists!')
        self.loader_map[name] = loader
        # start by default
        if name != 'paired':
            loader.reset()

    def reset(self):
        '''reset'''
        # only reset paired data
        self.loader_map['paired'].reset()

    def terminate(self):
        '''terminate'''
        for _, loader in self.loader_map.items():
            loader.terminate()

    def next(self, data_list):
        '''next'''
        # flush epoch using first key
        key = data_list[0]
        batch = self.loader_map[key].next()
        if batch is None:
            return batch
        batch_map = {key: batch}
        for key in data_list[1:]:
            batch = self.loader_map[key].next()
            if batch is None:
                logging.info(f'{key} loader begins next epoch.')
                self.loader_map[key].reset()
                batch = self.loader_map[key].next()
            batch_map[key] = batch
        return batch_map

    def set_oom_info(self):
        '''set oom info'''
        for _, loader in self.loader_map.items():
            loader.set_oom_info()

    # TODO only support paired data loader
    def state_dict(self):
        '''state dict'''
        return self.loader_map['paired'].state_dict()

    def reset_epoch_count(self, epoch_cout, skip_item_num=0):
        '''reset epoch count'''
        self.loader_map['paired'].reset_epoch_count(epoch_cout, skip_item_num)


@RUNNERS.register_module()
class USMMostRunner(RNNTRunner):
    '''USM Most Runner'''

    def __init__(self, cfg, inference=False, export_onnx=False):
        """Init."""
        super().__init__(cfg, inference=inference, export_onnx=export_onnx)

        self.paired_iter = self.train_cfg.get('paired_iter', -1)
        # unsupervised speech pretraining
        self.speech_only_iter = self.train_cfg.get('speech_only_iter', -1)
        # supervised alignment & duration
        self.modality_match_iter = self.train_cfg.get('modality_match_iter', 20000)
        # supervised text reconstruction
        self.paired_text_end_iter = self.train_cfg.get('paired_text_end_iter', -1)
        # supervised speech reconstruction
        self.paired_speech_end_iter = self.train_cfg.get('paired_speech_end_iter', -1)
        # text reconstruction
        self.text_only_iter = self.train_cfg.get('text_only_iter', 30000)

        # reset iter
        self._epoch = cfg.get('epoch', self._epoch)
        self._iter = cfg.get('iter', self._iter)

        if self.need_build_data_loader:
            super_dataloder = MultiModalMultiTaskDataloader()
            sup_data = self.train_data_loader
            val_data = self.valid_data_loader
            super_dataloder.add('paired', sup_data)  # add paired dataset
            speech_dataset_cfg = cfg.get('speech_data', None)
            if self.speech_only_iter < self.max_iters and speech_dataset_cfg is not None:
                self.get_data_list(speech_dataset_cfg)
                self.build_dataset(speech_dataset_cfg)
                speech_data = self.train_data_loader
                super_dataloder.add('speech', speech_data)  # add speech only dataset
            text_dataset_cfg = cfg.get('text_data', None)
            if self.text_only_iter < self.max_iters and text_dataset_cfg is not None:
                self.get_data_list(text_dataset_cfg)
                self.build_dataset(text_dataset_cfg)
                text_data = self.train_data_loader
                super_dataloder.add('text', text_data)  # add text only dataset
            self.train_data_loader = super_dataloder
            self.valid_data_loader = val_data

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = MostMetric()
        self.valid_log_buffer = MostMetric()

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError('lr is not applicable because optimizer does not exist.')

        # show layer_wise lr
        ret_lr_list = []
        if 'params' in self.optimizer_cfg:
            for item in self.optimizer_cfg['params']:
                lr = item['lr']
                if lr not in ret_lr_list:
                    ret_lr_list.append(lr)
            return [ret_lr_list]

        return [group['lr'] for group in self.optimizer.param_groups]

    @get_time('data_time')
    def next_train_batch(self, batch_list):
        '''next train batch'''
        batch_data = self.train_data_loader.next(batch_list)
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
            batch_data = self.train_data_loader.next(batch_list)
            if batch_data is None:
                raise RuntimeError(
                    "%s - rank %d data loader error"
                    % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.rank)
                )
        return batch_data

    # pylint:disable=too-many-branches
    @get_time('time')
    def train_iteration(self):
        '''train iteration.'''
        if hasattr(self.solution, 'set_num_updates'):
            self.solution.set_num_updates(self.iter)
        i = 0
        while i < self.grad_accum_step:
            batch_list = []
            if self.iter >= self.paired_iter:
                batch_list += ['paired']
            if self.iter >= self.speech_only_iter:
                batch_list += ['speech']
            if self.iter >= self.text_only_iter:
                batch_list += ['text']
            batch_data = self.next_train_batch(batch_list)
            if self.iter >= self.modality_match_iter:
                batch_list += ['speech-text']
                batch_data['speech-text'] = copy.deepcopy(batch_data['paired'])
            try:
                self.solution.train()
                loss_scale = self.dist_handler.get_scale(scaler_idx=0)
                batch_data['loss_scale'] = loss_scale
                if self.iter < self.paired_text_end_iter:
                    batch_data['use_paired_text'] = True
                if self.iter < self.paired_speech_end_iter:
                    batch_data['use_paired_speech'] = True
                solution_out = self.solution(batch_data, batch_list)
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

            except RuntimeError as e:
                self.handle_error(e, batch_data)
                continue
            self.train_log_buffer.update(solution_out)
            i = i + 1
        if self.opt_util_cfg.grad_clip:
            gnorm = self.clip_grads()
            self.train_log_buffer.update({'gnorm': gnorm})
        self.dist_handler.step(iters=self.iter)  # Pass in iter because of bmuf step

    @torch.no_grad()
    def validation_paired(self, name):
        '''paired data validation'''
        self.valid_log_buffer.reset()
        start_time = time.time()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                self.solution.eval()
                batch_list, batch_map = [], {}
                if self.iter >= self.paired_iter:
                    batch_list += ['paired']
                    batch_map['paired'] = copy.deepcopy(batch_data)
                if self.iter < self.paired_text_end_iter:
                    batch_map['use_paired_text'] = True
                if self.iter < self.paired_speech_end_iter:
                    batch_map['use_paired_speech'] = True
                if self.iter >= self.modality_match_iter:
                    batch_list += ['speech-text']
                    batch_map['speech-text'] = copy.deepcopy(batch_data)
                validation_out = self.solution(batch_map, batch_list)
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
        start_time = time.time()
        self.log_metric(self.valid_log_buffer, name=name)

    @torch.no_grad()
    def validation_unpaired(self, name):
        '''unpaired data validation'''
        self.valid_log_buffer.reset()
        start_time = time.time()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                self.solution.eval()
                batch_list, batch_map = [], {}
                if self.iter >= self.text_only_iter:
                    batch_list += ['text']
                    batch_map['text'] = copy.deepcopy(batch_data)
                validation_out = self.solution(batch_map, batch_list)
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
        start_time = time.time()
        self.log_metric(self.valid_log_buffer, name=name)

    @torch.no_grad()
    def validation(self):
        '''valid'''
        # Switch to eval model
        self.mode = 'val'
        self.solution.eval()
        if hasattr(self.solution.criterion_module, "combine_weight"):
            self.solution.criterion_module.combine_weight()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        if self.iter >= self.text_only_iter:
            self.validation_unpaired('txt')
        self.validation_paired('dev')

        if hasattr(self.solution.criterion_module, "clear_combined_weight"):
            self.solution.criterion_module.clear_combined_weight()
        # switch back to train mode
        self.call_hook('after_val_epoch')
        self.mode = 'train'
        self.solution.train()

    @torch.no_grad()
    def inference(self):
        '''inference'''
        infer_cfg = self.args.inference
        if infer_cfg.get('enable_ema', False) and self.solution.ema:
            logging.info('Enable EMA inference.')
            self.solution.ema.set_decay(1.0)
            self.solution.ema.step(self.solution)  # reset skip_keys
            self.solution = self.solution.ema.model
        super().inference()
