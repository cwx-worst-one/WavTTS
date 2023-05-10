''' RNNTMBRRunner '''

import numpy as np
import torch
from core.utils.cer.cer_metric import EditDistanceCalculator
from core.utils import logging
from ..base_runner import RUNNERS
from .rnnt_runner import RNNTRunner


@RUNNERS.register_module()
class RNNTMBRRunner(RNNTRunner):
    '''RNNT SDT with MBR criterion'''

    # pylint: disable=line-too-long
    @staticmethod
    def load_nbest_align_info_from_map(nbest_hyp_info):
        '''format nbest align info'''
        max_hyp_len = 0
        max_u_t_len = 0
        nbest_hyp_len = []
        nbest_hyps = []

        # obtain hyps length
        for key, val in nbest_hyp_info.items():
            nbest_hyps.append([int(v) for v in key.split() if v != ''])
            # blank at the head of nbest_hyps
            nbest_hyp_len.append(len(nbest_hyps[-1]) - 1)
            max_hyp_len = max(max_hyp_len, len(nbest_hyps[-1]))
            max_u_t_len = max(max_u_t_len, val['u_t_len'])

        beam_size = len(nbest_hyp_info)
        np_nbest_hyps = np.zeros((beam_size, max_hyp_len))
        np_nbest_t_path_idx = np.zeros((beam_size, max_u_t_len))
        np_nbest_u_path_idx = np.zeros((beam_size, max_u_t_len))
        np_nbest_hyp_path_idx = np.zeros((beam_size, max_u_t_len))
        np_nbest_t_u_path_idx_mask = np.zeros((beam_size, max_u_t_len))

        for idx, (key, val) in enumerate(nbest_hyp_info.items()):
            np_nbest_hyps[idx][0 : (nbest_hyp_len[idx] + 1)] = nbest_hyps[idx]
            np_nbest_t_path_idx[idx][0 : val['u_t_len']] = val['t_steps'][0 : val['u_t_len']]
            np_nbest_u_path_idx[idx][0 : val['u_t_len']] = val['u_steps'][0 : val['u_t_len']]
            np_nbest_hyp_path_idx[idx][0 : val['u_t_len']] = val['hyp_steps'][0 : val['u_t_len']]
            np_nbest_t_u_path_idx_mask[idx][0 : val['u_t_len']] = 1.0

        np_nbest_hyp_len = np.array(nbest_hyp_len)
        return (
            np_nbest_hyps,
            np_nbest_t_path_idx,
            np_nbest_u_path_idx,
            np_nbest_hyp_path_idx,
            np_nbest_hyp_len,
            np_nbest_t_u_path_idx_mask,
        )

    def generate_data_from_alignment(self, batch_data, nbest_hyps_info):
        '''generate nbest alignment data'''

        mbr_beam_size = self.solution_cfg.get('mbr_beam_size', 10)
        max_nbest_t_u_path_len = 0
        max_nbest_hyp_len = 0
        max_len_tmp = batch_data['src'].shape[1] * 2
        bsz = batch_data['src'].shape[0]
        np_nbest_hyps = np.zeros((bsz, mbr_beam_size, max_len_tmp), dtype='float32')
        np_nbest_hyps_mask = np.zeros((bsz, mbr_beam_size, max_len_tmp), dtype='float32')
        np_nbest_t_path_idx = np.zeros((bsz, mbr_beam_size, 2 * max_len_tmp), dtype='float32')
        np_nbest_u_path_idx = np.zeros((bsz, mbr_beam_size, 2 * max_len_tmp), dtype='float32')
        np_nbest_hyp_path_idx = np.zeros((bsz, mbr_beam_size, 2 * max_len_tmp), dtype='float32')
        np_nbest_t_u_path_idx_mask = np.zeros(
            (bsz, mbr_beam_size, 2 * max_len_tmp), dtype='float32'
        )

        char_cpu = batch_data['char'].cpu().numpy().astype('int')
        all_refs = []
        all_hyps = []
        for idx, nbest_hyp_info in enumerate(nbest_hyps_info):
            (
                nbest_hyps,
                nbest_t_path_idx,
                nbest_u_path_idx,
                nbest_hyp_path_idx,
                nbest_hyp_real_len,
                nbest_t_u_path_idx_mask,
            ) = self.load_nbest_align_info_from_map(nbest_hyp_info)

            all_refs.append(char_cpu[idx][: batch_data['target_lengths'][idx]].tolist())

            nbest_hyp_len = nbest_hyps.shape[1]
            max_nbest_hyp_len = max(max_nbest_hyp_len, nbest_hyp_len)
            np_nbest_hyps[idx, :, 0:nbest_hyp_len] = nbest_hyps

            nbest_t_u_path_len = nbest_t_path_idx.shape[1]
            max_nbest_t_u_path_len = max(max_nbest_t_u_path_len, nbest_t_u_path_len)
            np_nbest_t_path_idx[idx, :, 0:nbest_t_u_path_len] = nbest_t_path_idx
            np_nbest_u_path_idx[idx, :, 0:nbest_t_u_path_len] = nbest_u_path_idx
            np_nbest_hyp_path_idx[idx, :, 0:nbest_t_u_path_len] = nbest_hyp_path_idx
            np_nbest_t_u_path_idx_mask[idx, :, 0:nbest_t_u_path_len] = nbest_t_u_path_idx_mask

            beam_hyps = []
            for beam_idx in range(nbest_hyps.shape[0]):
                np_nbest_hyps_mask[idx][beam_idx][0 : nbest_hyp_real_len[beam_idx] + 1] = 1
                # ignore blank
                hyp = (
                    nbest_hyps[beam_idx, 1 : nbest_hyp_real_len[beam_idx] + 1]
                    .astype('int')
                    .tolist()
                )
                beam_hyps.append(hyp)
            all_hyps.append(beam_hyps)

        batch_data['nbest_hyps'] = (
            torch.as_tensor(np_nbest_hyps[:, :, 0:max_nbest_hyp_len]).long().cuda()
        )
        batch_data['nbest_hyps_mask'] = torch.as_tensor(
            np_nbest_hyps_mask[:, :, 0:max_nbest_hyp_len]
        ).cuda()
        batch_data['nbest_t_path_idx'] = (
            torch.as_tensor(np_nbest_t_path_idx[:, :, 0:max_nbest_t_u_path_len]).long().cuda()
        )
        batch_data['nbest_u_path_idx'] = (
            torch.as_tensor(np_nbest_u_path_idx[:, :, 0:max_nbest_t_u_path_len]).long().cuda()
        )
        batch_data['nbest_hyp_path_idx'] = (
            torch.as_tensor(np_nbest_hyp_path_idx[:, :, 0:max_nbest_t_u_path_len]).long().cuda()
        )
        batch_data['nbest_t_u_path_idx_mask'] = torch.as_tensor(
            np_nbest_t_u_path_idx_mask[:, :, 0:max_nbest_t_u_path_len]
        ).cuda()
        return batch_data, all_refs, all_hyps

    def prepare_nbest_data(self, batch_data, ed_calculator):
        '''
        generate nbest sequence on-the-fly for mbr loss
        '''
        # pylint:disable=too-many-locals
        # inference config
        solution_cfg = self.solution_cfg
        mbr_beam_size = solution_cfg.get('mbr_beam_size', 10)
        wer_filter = solution_cfg.get('wer_filter', None)

        assert mbr_beam_size > 1, "mbr_beam_size must be larger than 1"
        nbest_hyps_info, _, _ = self.solution.beam_inference(batch_data, nbest_align_info=True)

        batch_data, all_refs, all_hyps = self.generate_data_from_alignment(
            batch_data, nbest_hyps_info
        )

        # get wer
        bsz = batch_data['src'].shape[0]
        mbr_beam_size = batch_data['nbest_hyps'].shape[1]
        # update mbr_beam_size with beam-search results
        if mbr_beam_size == 1:
            return False
        word_errors = np.zeros((bsz, mbr_beam_size), dtype='float32')
        word_nums = np.zeros((bsz,), dtype='float32')
        for idx, ref in enumerate(all_refs):
            ref = ' '.join(map(str, ref))
            if wer_filter:
                bad_sentence = False
            for beam_idx, hyp in enumerate(all_hyps[idx]):
                hyp = ' '.join(map(str, hyp))
                ed_info, _ = ed_calculator.show_alignment(batch_data['uttid'][idx], ref, hyp)
                word_errors[idx, beam_idx] = (
                    ed_info['sub_err'] + ed_info['ins_err'] + ed_info['del_err']
                )
                if wer_filter and word_errors[idx, beam_idx] / ed_info['ref_word_num'] > wer_filter:
                    bad_sentence = True
                    break
            if wer_filter and bad_sentence:
                word_errors[idx, :] = 0.0
            word_nums[idx] = ed_info['ref_word_num']
        batch_data['word_errors'] = torch.as_tensor(word_errors).cuda()
        batch_data['total_words'] = torch.as_tensor(word_nums).cuda()
        # prepare nbest acoustic and predictor out for mbr loss
        # B, N_sample, T+U
        acoustic_out = batch_data['encoder_out']
        nbest_t_path_idx = batch_data['nbest_t_path_idx']
        nbest_u_path_idx = batch_data['nbest_u_path_idx']
        nbest_t_u_path_idx_mask = batch_data['nbest_t_u_path_idx_mask']
        nbest_t_u_length = (nbest_t_u_path_idx_mask.sum(dim=-1)).int()
        t_u_shape = nbest_t_path_idx.shape

        nbest_hyps = batch_data['nbest_hyps']
        nbest_hyps_for_predicter = nbest_hyps.view(-1, nbest_hyps.shape[2])
        hyp_predicter_out, _ = self.solution.predictor_module(nbest_hyps_for_predicter)

        hyp_predicter_out = hyp_predicter_out.view(
            bsz, mbr_beam_size, hyp_predicter_out.shape[1], hyp_predicter_out.shape[2]
        )

        # expand the acoustic out T -> T+U (B, beam_size, T+U, h)
        # expand the predicter out T -> T+U
        # do jointer
        acoustic_out_expand = torch.zeros(
            [t_u_shape[0], t_u_shape[1], t_u_shape[2], acoustic_out.shape[2]]
        ).type_as(acoustic_out)
        hyp_predicter_out_expand = torch.zeros(
            [t_u_shape[0], t_u_shape[1], t_u_shape[2], acoustic_out.shape[2]]
        ).type_as(acoustic_out)
        for i in range(t_u_shape[0]):
            for j in range(t_u_shape[1]):
                u_t_len = nbest_t_u_length[i][j]
                acoustic_out_expand[i][j][0:u_t_len] = acoustic_out[i].index_select(
                    0, nbest_t_path_idx[i][j][0:u_t_len]
                )
                hyp_predicter_out_expand[i][j][0:u_t_len] = hyp_predicter_out[i][j].index_select(
                    0, nbest_u_path_idx[i][j][0:u_t_len]
                )
        nbest_jointer_out = self.solution.jointer_module.forward_step(
            acoustic_out_expand, hyp_predicter_out_expand
        )
        batch_data['nbest_jointer_out'] = nbest_jointer_out
        return True

    def precompute_encoder_out(self, batch_data):
        '''precompute encoder out'''
        (
            encoder_out,
            backbone_mask,
            trainable,
            mtl_logits,
            encoder_backbone_out,
        ) = self.solution.encoder(batch_data)
        batch_data['encoder_out'] = encoder_out
        batch_data['backbone_mask'] = backbone_mask
        batch_data['trainable'] = trainable
        batch_data['mtl_logits'] = mtl_logits
        batch_data['encoder_backbone_out'] = encoder_backbone_out

    def train(self):
        '''train loop'''
        self.build_beam_search(key='solution')
        self.before_train()
        ed_calculator = EditDistanceCalculator()
        while self.iter < self.args.train.max_iters:
            # train step begin
            self.call_hook('before_train_iter')
            i = 0
            while i < self.grad_accum_step:
                batch_data = self.next_train_batch()
                try:
                    # precompute encoder_out
                    self.precompute_encoder_out(batch_data)
                    # get rnnt nbest on the fly for mbr train
                    with torch.no_grad():
                        self.solution.eval()  # eval mode
                        if not self.prepare_nbest_data(batch_data, ed_calculator):
                            continue
                    self.solution.train()
                    self.solution_out = self.solution(batch_data)
                    self.loss = self.solution_out['backward_loss'] / self.grad_accum_step
                    self.dist_handler.backward(self.loss, unscale=(i + 1 == self.grad_accum_step))
                except RuntimeError as e:
                    # destroy auto grad graph through delete loss,
                    # so the gpu memory could be free.
                    self.handle_error(e, batch_data)
                    continue
                self.train_log_buffer.update(self.solution_out)
                i = i + 1
            # clip the grad
            if self.opt_util_cfg.grad_clip:
                gnorm = self.clip_grads()
                self.train_log_buffer.update({'gnorm': gnorm})
            # optimizer step
            self.dist_handler.step(iters=self.iter)
            self.call_hook('after_train_iter')
            self.log_metric(self.train_log_buffer)
            # validation
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
        self.solution.criterion_module.combine_weight()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0

        ed_calculator = EditDistanceCalculator()
        self.valid_log_buffer.reset()
        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                # precompute encoder_out
                (
                    encoder_out,
                    backbone_mask,
                    trainable,
                    mtl_logits,
                    encoder_backbone_out,
                ) = self.solution.encoder(batch_data)
                batch_data['encoder_out'] = encoder_out
                batch_data['backbone_mask'] = backbone_mask
                batch_data['trainable'] = trainable
                batch_data['mtl_logits'] = mtl_logits
                batch_data['encoder_backbone_out'] = encoder_backbone_out

                self.solution.eval()
                # get rnnt nbest on the fly for mbr train
                with torch.no_grad():
                    if not self.prepare_nbest_data(batch_data, ed_calculator):
                        batch_data = None

                validation_out = self.solution(batch_data)
                self.valid_log_buffer.update(validation_out)
                self._val_iter += 1
                batch_data = self.valid_data_loader.next()
                self.call_hook('after_val_iter')
            except RuntimeError as e:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                continue
        self.call_hook('after_val_epoch')
        self.log_metric(self.valid_log_buffer)
        self.solution.criterion_module.clear_combined_weight()
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()

    def build_beam_search(self, key='inference'):
        '''build beam search'''
        cfg = self.args.get(key, None)
        if key == 'solution':
            cfg.setdefault('len_penalty_scale', 0.05)
            cfg.setdefault('use_batch_beam', True)
            cfg.nbest_align_info = True
            cfg.recombine_sum = False
            cfg.beam_size = cfg.get('mbr_beam_size', 10)
        super().build_beam_search(key=key)
