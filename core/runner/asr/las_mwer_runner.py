''' LASMWERRunner '''

import torch
from core.utils import logging
from core.utils.cer.cer_metric import EditDistanceCalculator
from core.runner.metric.asr_metric import SDTMetric
from core.extensions import clear_cuda_error
from ..base_runner import RUNNERS
from .base_las_runner import BaseLASRunner


@RUNNERS.register_module()
class LASMWERRunner(BaseLASRunner):
    '''LAS MWER Runner'''

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = SDTMetric()
        self.valid_log_buffer = SDTMetric()

    def prepare_nbest_data(self, batch_data, ed_calculator):
        '''
        generate nbest sequence on-the-fly for mwer loss
        '''
        # inference config
        mwer_beam_size = self.args.solution.get('mwer_beam_size', 4)
        mwer_beam_len_filter = self.args.solution.get('mwer_beam_len_filter', 10)
        mwer_beam_wer_filter_best = self.args.solution.get('mwer_beam_wer_filter_best', 0.2)
        mwer_beam_wer_filter_worst = self.args.solution.get('mwer_beam_wer_filter_worst', 0.5)

        reorder_dict_map = self.reorder_dict_map
        _, hyps_nbest = self.solution.beam_inference(
            batch_data, mwer_beam_size, reorder_dict_map=reorder_dict_map, nbest_out=True
        )

        # prepare mwer training data
        max_len = 0
        hyp_tensor_dtype = hyps_nbest[0][0].dtype
        for i, hyp in enumerate(hyps_nbest):
            for j in range(mwer_beam_size):
                max_len = max(max_len, hyp[j].size()[0])

        # restrict the maximum length no longer than the golden_max_len+10
        golden_max_len = int(batch_data['char_mask'].sum(dim=1).max().item())
        max_len = max(max_len, golden_max_len)  # include ground-truth beam
        max_len = min(max_len, golden_max_len + mwer_beam_len_filter)
        hyps_nbest_mask = torch.zeros(len(hyps_nbest), mwer_beam_size, max_len, dtype=torch.float32)
        hyps_nbest_tensor = torch.full(
            (len(hyps_nbest), mwer_beam_size, max_len), self.tgt_dict.eos(), dtype=hyp_tensor_dtype
        )
        prev_hyps_nbest_tensor = torch.full(
            (len(hyps_nbest), mwer_beam_size, max_len), self.tgt_dict.eos(), dtype=hyp_tensor_dtype
        )
        # wer and average wer
        word_errors_tensor = torch.zeros(len(hyps_nbest), mwer_beam_size, dtype=torch.float32)
        total_words_tensor = batch_data['char_mask'].sum(dim=1)

        for i, hyps_beam in enumerate(hyps_nbest):
            best_wer = 1.0
            worst_wer = 0.0
            for j in range(mwer_beam_size):
                # pylint: disable=line-too-long
                ref = (
                    batch_data['char'][i][: int(batch_data['char_mask'].sum(dim=1)[i])]
                    .cpu()
                    .numpy()
                    .astype('int')
                    .tolist()
                )
                ref = ' '.join(map(str, ref))
                hyp = hyps_beam[j].cpu().numpy().astype('int').tolist()
                hyp = ' '.join(map(str, hyp))

                ed_info, _ = ed_calculator.show_alignment(batch_data['uttid'][i], ref, hyp)
                word_errors_tensor[i][j] = (
                    ed_info['sub_err'] + ed_info['ins_err'] + ed_info['del_err']
                )

                beam_wer = word_errors_tensor[i][j] / ed_info['ref_word_num']
                best_wer = min(best_wer, beam_wer)
                worst_wer = max(worst_wer, beam_wer)

                if best_wer >= mwer_beam_wer_filter_best or worst_wer >= mwer_beam_wer_filter_worst:
                    # ignore this train sample because of bad beams
                    # With mwer loss, the prob of the best beam may be inevitably increased.
                    # If the worst beam is too bad,
                    # the probs of other beams may be inevitably increased.
                    hyps_nbest_mask[i] = hyps_nbest_mask[i] * 0.0
                    break

                for k in range(hyps_beam[j].size()[0]):
                    if k >= max_len:
                        break
                    hyps_nbest_tensor[i][j][k] = hyps_beam[j][k]
                    hyps_nbest_mask[i][j][k] = 1.0
                    if k == 0:
                        prev_hyps_nbest_tensor[i][j][k] = self.tgt_dict.bos()
                    else:
                        prev_hyps_nbest_tensor[i][j][k] = hyps_beam[j][k - 1]

        batch_data['nbest_sample_mask'] = hyps_nbest_mask.cuda()  # B, N_sample, tgt_len
        batch_data['nbest_sample'] = hyps_nbest_tensor.cuda()
        batch_data['prev_nbest_sample'] = prev_hyps_nbest_tensor.cuda()
        batch_data['word_errors'] = word_errors_tensor.cuda()  # B, N_sample
        batch_data['total_words'] = total_words_tensor.cuda()  # B
        # end

        return batch_data

    def train(self):
        '''train loop'''
        self.before_train()
        ed_calculator = EditDistanceCalculator()
        while self.iter < self.args.train.max_iters:
            # train step begin
            self.call_hook('before_train_iter')
            i = 0
            while i < self.grad_accum_step:
                batch_data = self.next_train_batch()
                try:
                    # get las nbest on the fly for mwer train
                    with torch.no_grad():
                        self.solution.eval()  # eval mode
                        batch_data = self.prepare_nbest_data(batch_data, ed_calculator)
                    self.solution.train()
                    self.solution_out = self.solution(batch_data)
                    self.loss = self.solution_out['backward_loss'] / self.grad_accum_step
                    self.dist_handler.backward(self.loss, unscale=(i + 1 == self.grad_accum_step))
                except RuntimeError as e:
                    # destroy auto grad graph through delete loss,
                    # so the gpu memory could be free.
                    self.handle_error(e, batch_data)
                    continue
                i = i + 1
            # clip the grad
            if self.opt_util_cfg.grad_clip:
                self.solution_out['gnorm'] = self.clip_grads()
            # optimizer step
            self.dist_handler.step(iters=self.iter)
            if (self.iter + 1) % self.args.valid.interval == 0:
                self.validation()
            self._iter += 1
            self._inner_iter += 1
        self.after_train()

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

        ed_calculator = EditDistanceCalculator()
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                # get las nbest on the fly for mwer train
                with torch.no_grad():
                    batch_data = self.prepare_nbest_data(batch_data, ed_calculator)

                validation_out = self.solution(batch_data)
                self._val_iter += 1
                batch_data = self.valid_data_loader.next()
                self.call_hook('after_val_iter')
                self.valid_log_buffer.update(validation_out)
            except RuntimeError as e:
                clear_cuda_error()
                torch.cuda.empty_cache()
                logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                continue
        self.call_hook('after_val_epoch')
        self.log_metric(self.valid_log_buffer)
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()
