"""RnntIlmModel"""
# pylint: disable=abstract-method
from core.criterions import *
from core.solutions.base_solution import BaseSolution, register_solution
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *


@register_solution("RnntIlmModel")
class RnntIlmModel(BaseSolution):
    '''
    ilm in RNN-T.
    - Predictor
    - Jointer
    - criterion
    '''

    def __init__(self, args):
        '''
        init function for RNN-T skeleton model.
        '''
        super().__init__()
        self.args = args
        self.predictor_module = eval(args.predictor_type)(args)
        self.jointer_module = eval(args.jointer_type)(args)
        self.criterion_module = eval(args.criterion_type)(args)
        self.limited_context = args.get('limited_context', None)

    def predictor(self, batch_data, key='prev_char'):
        '''
        Predictor Module in RNN-T
        '''
        prev_char = batch_data[key]
        if self.limited_context is not None:
            limited_context = self.limited_context
            batch, time = prev_char.size()
            pad = F.pad(batch_data['char'], (limited_context - 1, 0, 0, 0)).type_as(
                batch_data['char']
            )
            prev_char_k = (
                F.unfold(pad.unsqueeze(1).unsqueeze(2), (1, limited_context - 1))
                .transpose(1, 2)
                .contiguous()
            )
            prev_char_k = F.pad(prev_char_k, (1, 0, 0, 0, 0, 0)).type_as(batch_data['char'])
            prev_char_k = prev_char_k.view(-1, limited_context).contiguous()
            predictor_out_k, _ = self.predictor_module(prev_char_k)
            predictor_out = predictor_out_k.view(batch, time, -1).contiguous()
        else:
            unk_idx = self.args.tgt_dict.index('<unk>')
            predictor_out, _ = self.predictor_module(prev_char, prev_char_drop=0.0, unk_idx=unk_idx)
        return predictor_out

    def jointer(self, encoder, predictor, input_lengths, target_lengths):
        '''
        Jointer Module in RNN-T.
        '''
        jointer_out = self.jointer_module(encoder, predictor, input_lengths, target_lengths)
        return jointer_out

    @torch.no_grad()
    def get_log_prob(self, batch_data):
        '''
        get log prob
        '''
        ############ Encoder #############
        ilm_encoder_out = (
            torch.zeros(batch_data['prev_char'].shape[0], 1, self.args.head_hidden_size)
            .cuda()
            .float()
        )
        ############ Predictor ############
        predictor_out = self.predictor(batch_data)
        target_lengths = batch_data['target_lengths']
        target = batch_data['char']
        target_mask = batch_data['char_mask']
        ############ Jointer ############
        ilm_jointer_out = self.jointer(ilm_encoder_out, predictor_out, None, target_lengths)
        input_lengths = torch.ones_like(target_mask[:, 0]).int()  # (B)
        ilm_prob = self.criterion_module.log_softmax_fc(
            ilm_jointer_out,
            target,
            input_lengths,
            target_mask.sum(1).int(),
            blank_setto_zero=True,
            adaptive_tgt_indices=batch_data['adaptive_tgt_indices'],
        )  # (B, 1, U, 2)
        log_probs = ilm_prob[:, 0, :-1, 1].contiguous()
        log_probs_out = log_probs.masked_fill(~target_mask.bool(), 0)
        return log_probs_out
