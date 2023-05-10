''' BaseRnntModel '''
import os.path as osp
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from core.models.kws.rnnt_encoder import *
from core.models.kws.rnnt_predictor import *
from core.models.kws.rnnt_jointer import *
from core.models.kws.criterion_head import *
from core.criterions import *
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.inference import (
    BaseInfer,
    INFERS,
)
from core.utils import dist_hdfs_get, FalconDict


class KwsRnntEncoderExporter(BaseInfer):
    '''export kws rnnt encoder model to onnx'''

    NAME = 'kws_rnnt_encoder'
    INPUTS = [
        FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1]),
    ]
    OUTPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1]),
    ]

    def __init__(self, encoder, **kwargs):
        super().__init__(kwargs)
        self.encoder = encoder
        fbank_dim = kwargs.get('fbank_dim')
        self._inputs[0].shape[2] = fbank_dim
        if kwargs.get('selected_dfsmn_layer', -1) != -1:
            self._outputs.append(
                FalconDict(name='vad_out', type=torch.float32, shape=['B', 'T', -1])
            )
        self.with_slim = kwargs.get('slim_quant_encoder')
        self._convert_stream_flag = kwargs.get('encoder_convert_stream', True)

    def forward(self, fbank):
        '''
        forward for kws rnnt encoder model
        '''
        encoder_out, selected_encoder_out, _, _ = self.encoder(fbank)
        if selected_encoder_out is not None:
            return encoder_out, selected_encoder_out

        return encoder_out

    def sample_inputs(self):
        fbank = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        return fbank


class KwsRnntPredictorExporter(BaseInfer):
    '''export kws rnnt predictor model to onnx'''

    NAME = 'kws_rnnt_predictor'
    INPUTS = [FalconDict(name='prev_tgt', type=torch.long, shape=['B', 'T'])]
    OUTPUTS = [FalconDict(name='predictor_out', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, predictor, **kwargs):
        super().__init__(kwargs)
        self.predictor = predictor

    def forward(self, prev_tgt):
        '''
        forward for kws rnnt predictor model
        '''
        predictor_out = self.predictor(prev_tgt)
        return predictor_out

    def sample_inputs(self):
        prev_tgt = self._generate_input_data(0, method='ones', dynamic_axis=[1, 1])
        return prev_tgt


class KwsRnntJointerExporter(BaseInfer):
    '''export kws rnnt jointer model to onnx'''

    NAME = 'kws_rnnt_jointer'
    INPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='predictor_out', type=torch.float32, shape=['B', 'T', -1]),
    ]
    OUTPUTS = [
        FalconDict(name='jointer_out', type=torch.float32, shape=['B', 'T', 'T', -1]),
    ]

    def __init__(
        self,
        joint_module,
        adaptive_module=None,
        adaptive_head_module=None,
        output_layer_type='log_softmax',
        **kwargs,
    ):
        super().__init__(kwargs)
        self.joint_module = joint_module
        self.adaptive_module = adaptive_module
        self.adaptive_head_module = adaptive_head_module
        self.output_layer_type = output_layer_type
        self._inputs[0].shape[2] = kwargs.get('rnnt_hidden_size')
        self._inputs[1].shape[2] = kwargs.get('rnnt_hidden_size')
        self.with_slim = kwargs.get('slim_quant_jointer')

    def forward(self, encoder_out, predictor_out):
        '''
        forward for kws rnnt jointer model
        '''
        bsz, tgt_num, _ = predictor_out.shape
        adaptive_tgt_indices = [torch.ones(bsz, tgt_num + 1, 1)]
        encoder_out = encoder_out.unsqueeze(2)
        predictor_out = predictor_out.unsqueeze(1)
        jointer_in = encoder_out + predictor_out
        jointer_in = torch.tanh(jointer_in)
        jointer_out = self.joint_module(jointer_in)
        if self.adaptive_head_module is not None:
            jointer_out = self.adaptive_head_module(jointer_out)
        elif self.adaptive_module is not None:
            target = torch.ones(predictor_out.shape[0], predictor_out.shape[2] - 1).long().cuda()
            input_lengths = torch.tensor([[encoder_out.shape[1]]]).long().cuda()
            target_lengths = torch.tensor([[predictor_out.shape[2]]]).long().cuda()
            jointer_out = self.adaptive_module(
                jointer_out,
                target,
                input_lengths,
                target_lengths,
                adaptive_tgt_indices=adaptive_tgt_indices,
            )
        if self.output_layer_type == 'softmax':
            jointer_out = F.softmax(jointer_out.float(), dim=-1)
        elif self.output_layer_type == 'log_softmax':
            jointer_out = F.log_softmax(jointer_out.float(), dim=-1)
        return jointer_out

    def sample_inputs(self):
        encoder_out = self._generate_input_data(0, method='ones', dynamic_axis=[1, 1, -1])
        predictor_out = self._generate_input_data(1, method='ones', dynamic_axis=[1, 1, -1])
        data = [encoder_out, predictor_out]
        return tuple(data)


@register_solution("KwsRnntModel")
class KwsRnntModel(BaseSolution):
    '''
    KWS model for RNN-T.
    - Encoder
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
        self.encoder = self.build_encoder(args)
        self.predictor = self.build_predictor(args)
        self.jointer = self.build_jointer(args)
        self.criterion = self.build_criterion(args)
        if args.mtl_type is not None:
            if 'ctc' in args.mtl_type:
                args.head_type = args.ctc_head_type
                args.head_hidden_layer_num = args.ctc_head_hidden_layer_num
                args.head_input_size = args.ctc_head_input_size
                args.head_hidden_size = args.ctc_head_hidden_size
                args.head_layer_norm = args.ctc_head_layer_norm
                args.tgt_vocab_size = args.ctc_tgt_vocab_size
                self.ctc_head = self.build_head(args)
            if 'ce' in args.mtl_type:
                args.head_type = args.ce_head_type
                args.head_hidden_layer_num = args.ce_head_hidden_layer_num
                args.head_input_size = args.ce_head_input_size
                args.head_hidden_size = args.ce_head_hidden_size
                args.tgt_vocab_size = args.ce_tgt_vocab_size
                self.ce_head = self.build_head(args)
        self.update_steps = 0
        self.args['slim_quant_encoder'] = False
        self.args['slim_quant_jointer'] = False

    @staticmethod
    def build_encoder(args):
        '''
        build encoder
        '''
        encoder = eval(args.encoder_type)(args)
        if args.pretrain_encoder_model_path is not None:
            pretrain_encoder_model_path = args.pretrain_encoder_model_path
            local_dir = '/tmp/'
            tmp_model = 'pretrain_encoder_model.pt'
            dist_hdfs_get(pretrain_encoder_model_path, local_dir, tmp_model)
            tmp_model = osp.join(local_dir, tmp_model)
            pretrain_encoder_model_path = tmp_model
            pretrain_encoder_model = torch.load(pretrain_encoder_model_path, map_location='cpu')
            pretrain_encoder_model = pretrain_encoder_model['model']
            for name, param in pretrain_encoder_model.named_parameters():
                name = args.pretrain_encoder_model_prefix + name
                if name in pretrain_encoder_model:
                    if param.shape == pretrain_encoder_model[name].shape:
                        print("load encoder param {}".format(name))
                        param.data.copy_(pretrain_encoder_model[name])
                        if args.freeze_pretrain_encoder_model:
                            print('freeze param {}'.format(name))
                            param.requires_grad = False
                    else:
                        print('param {} shape mismatch!'.format(name))
                else:
                    print('param {} is not in pretrain encoder model'.format(name))
        return encoder

    @staticmethod
    def build_predictor(args):
        '''
        build predictor
        '''
        predictor = eval(args.predictor_type)(args)
        if args.pretrain_predictor_model_path is not None:
            pretrain_predictor_model_path = args.pretrain_predictor_model_path
            local_dir = '/tmp/'
            tmp_model = 'pretrain_predictor_model.pt'
            dist_hdfs_get(pretrain_predictor_model_path, local_dir, tmp_model)
            tmp_model = osp.join(local_dir, tmp_model)
            pretrain_predictor_model_path = tmp_model
            pretrain_predictor_model = torch.load(pretrain_predictor_model_path, map_location='cpu')
            pretrain_predictor_model = pretrain_predictor_model['model']
            for name, param in predictor.named_parameters():
                name = 'predictor.' + name
                if name in pretrain_predictor_model:
                    if param.shape == pretrain_predictor_model[name].shape:
                        print("load predictor param {}".format(name))
                        param.data.copy_(pretrain_predictor_model[name])
                        if args.freeze_pretrain_predictor_model:
                            print("freeze param {}".format(name))
                            param.requires_grad = False
                    else:
                        print("param {} shape mismatch!".format(name))
                else:
                    print("param {} is not in pretrain predictor model".format(name))
        return predictor

    @staticmethod
    def build_jointer(args):
        '''
        build jointer
        '''
        jointer = eval(args.jointer_type)(args)
        if args.pretrain_jointer_model_path is not None:
            pretrain_jointer_model_path = args.pretrain_jointer_model_path
            local_dir = '/tmp/'
            tmp_model = 'pretrain_jointer_model.pt'
            dist_hdfs_get(pretrain_jointer_model_path, local_dir, tmp_model)
            tmp_model = osp.join(local_dir, tmp_model)
            pretrain_jointer_model_path = tmp_model
            pretrain_jointer_model = torch.load(pretrain_jointer_model_path, map_location='cpu')
            pretrain_jointer_model = pretrain_jointer_model['model']
            for name, param in jointer.named_parameters():
                name = 'jointer.' + name
                if name in pretrain_jointer_model:
                    if param.shape == pretrain_jointer_model[name].shape:
                        print("load jointer param {}".format(name))
                        param.data.copy_(pretrain_jointer_model[name])
                        if args.freeze_pretrain_jointer_model:
                            print("freeze param {}".format(name))
                            param.requires_grad = False
                    else:
                        print("param {} shape mismatch!".format(name))
                else:
                    print("param {} is not in pretrain jointer model".format(name))
        return jointer

    @staticmethod
    def build_criterion(args):
        '''
        build criterion
        '''
        criterion = eval(args.criterion_type)(args)
        return criterion

    @staticmethod
    def build_head(args):
        '''
        build head
        '''
        head = eval(args.head_type)(args)
        return head

    def forward(self, batch_data):
        '''
        Forward for RNN-T
        '''
        # pylint:disable=too-many-branches,too-many-locals
        if self.training:
            self.update_steps += 1
        # encoder
        fbank = batch_data['src']
        fbank_mask = batch_data['src_mask']
        ce_label = None
        if 'ce_label' in batch_data:
            ce_label = batch_data['ce_label']

        encoder_out, selected_encoder_out, encoder_mask, ce_target = self.encoder(
            fbank, fbank_mask, ce_label
        )
        mtl_type = self.args.mtl_type
        if mtl_type is not None:
            if 'ctc' in mtl_type:
                if self.args.ctc_use_selected_encoder_out and selected_encoder_out is not None:
                    ctc_input = selected_encoder_out
                else:
                    ctc_input = encoder_out
                ctc_logits = self.ctc_head(ctc_input)
                ctc_target = batch_data['char']
            else:
                ctc_logits = None
                ctc_target = None
            if 'ce' in mtl_type:
                if self.args.ce_use_selected_encoder_out and selected_encoder_out is not None:
                    ce_input = selected_encoder_out
                else:
                    ce_input = encoder_out
                ce_logits = self.ce_head(ce_input)
            else:
                ce_logits = None
                ce_target = None

        # predictor
        if self.args.predictor_label_dropout_factor > 0 and (encoder_out.requires_grad):
            with torch.no_grad():
                prev_tgt = batch_data['prev_char']
                random_uniform_tensor = (
                    torch.empty(prev_tgt.size())
                    .uniform_(0, self.args.predictor_label_dropout_factor + 1)
                    .cuda()
                )
                mask = (random_uniform_tensor < 1).long()
                # print(mask.sum())
                eos_idx = 2
                batch_data['prev_char'] = batch_data['prev_char'] * mask + (1 - mask) * eos_idx
        prev_tgt = batch_data['prev_char']
        predictor_out = self.predictor(prev_tgt)

        # jointer
        target = batch_data['char']
        target_mask = batch_data['char_mask']
        adaptive_tgt_indices = batch_data.get('adaptive_tgt_indices', None)
        input_lengths = (encoder_mask.sum(dim=1)).int()
        max_input_lengths = input_lengths.max().item()
        target_lengths = (target_mask.sum(dim=1)).int()
        encoder_out = encoder_out[:, 0:max_input_lengths, :].contiguous()
        logits = self.jointer(
            encoder_out.unsqueeze(2),
            predictor_out.unsqueeze(1),
            target=target,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            adaptive_tgt_indices=adaptive_tgt_indices,
        )

        # criterion
        mtl_type = self.args.mtl_type
        rnnt_exclude_data_tag = self.args.rnnt_exclude_data_tag
        utt_ids = batch_data['uttid']
        src_mask = batch_data['src_mask']
        forward_out = self.criterion(
            logits,
            target,
            src_mask,
            encoder_mask,
            target_mask,
            utt_ids,
            ce_logits,
            ce_target,
            ctc_logits,
            ctc_target,
            rnnt_exclude_data_tag,
        )

        # cer stats
        error_dist = 0.0
        total_dist = 0.0
        ctc_error_dist = 0.0
        if (self.update_steps % self.args.cer_update_freq == 0) or (not logits.requires_grad):
            with torch.no_grad():
                error_dist, total_dist = self.cal_rnnt_cer_stats(encoder_out, target, target_mask)
                if self.args.mtl_type is not None:
                    if 'ctc' in self.args.mtl_type:
                        ctc_error_dist, _ = self.cal_ctc_cer_stats(ctc_logits, target, target_mask)
                        forward_out['ctc_cer'] = ctc_error_dist
                    if 'ce' in self.args.mtl_type:
                        hyp_acc = self.cal_frame_acc(ce_logits, ce_target, encoder_mask)
                        forward_out['acc'] = hyp_acc

        forward_out['cer'] = error_dist
        forward_out['dist'] = total_dist
        return forward_out

    def greedy_decode(self, encoder_out, argmax=True):
        '''
        greedy decode
        '''
        with torch.no_grad():
            (bsz, frame, _) = encoder_out.size()
            if argmax:
                tmp_output = encoder_out.data.new(bsz, frame).zero_().int()
            else:
                tmp_output = encoder_out.data.new(bsz, frame, self.args.tgt_vocab_size).zero_()
            prev_token = (
                encoder_out.data.new(
                    bsz,
                )
                .zero_()
                .long()
            )
            predictor_states = None
            predictor_feat, predictor_states = self.predictor.step(prev_token, predictor_states)
            for frame_idx in range(frame):
                logits = self.jointer.search(encoder_out[:, frame_idx, :], predictor_feat)
                lprobs = logits.log_softmax(dim=-1)

                hyp = torch.argmax(lprobs, dim=-1)

                if argmax:
                    tmp_output[:, frame_idx] = hyp
                else:
                    tmp_output[:, frame_idx] = lprobs

                hyp = hyp.view(bsz)

                mask = hyp == 0
                mask = mask.unsqueeze(1)

                predictor_feat_next, predictor_states_next = self.predictor.step(
                    hyp, predictor_states
                )
                predictor_feat = torch.where(mask, predictor_feat, predictor_feat_next)

                layer_idx = 0
                for predictor_state, predictor_state_next in zip(
                    predictor_states, predictor_states_next
                ):
                    if self.args.predictor_type == 'LSTMPredictor':
                        predictor_h = torch.where(mask, predictor_state[0], predictor_state_next[0])
                        predictor_c = torch.where(mask, predictor_state[1], predictor_state_next[1])
                        predictor_states[layer_idx] = [predictor_h, predictor_c]
                    layer_idx += 1
            return tmp_output

    def prob_beam_forward(self, batch_data, tokens, keep_valid_probs=False):
        '''
        beam decode for probabilities
        '''
        fbank = batch_data['src']
        fbank_mask = batch_data['src_mask']
        encoder_out, _, encoder_mask, _ = self.encoder(fbank, fbank_mask, None)

        ntoken = tokens.shape[0]
        (bsz, frame, ndim) = encoder_out.size()
        predictor_out = encoder_out.data.new(bsz, ntoken, ndim).zero_()
        predictor_states = None
        for token_idx in range(ntoken):
            prev_token = torch.full((bsz,), tokens[token_idx], dtype=torch.long).cuda()
            predictor_feat, predictor_states = self.predictor.step(prev_token, predictor_states)
            predictor_out[:, token_idx, :] = predictor_feat

        if keep_valid_probs:
            # only keep blank and current token probs
            output_probs = encoder_out.data.new(bsz, ntoken, frame, 2).zero_()
        else:
            if not self.args.use_head_log_prob:
                output_probs = encoder_out.data.new(
                    bsz, ntoken, frame, self.args.tgt_vocab_size
                ).zero_()
            else:
                output_probs = encoder_out.data.new(
                    bsz, ntoken, frame, self.args.adaptive_head_size + 1
                ).zero_()

        for frame_idx in range(frame):
            for token_idx in range(ntoken):
                logits = self.jointer.search(
                    encoder_out[:, frame_idx, :], predictor_out[:, token_idx, :]
                )
                lprobs = logits.log_softmax(dim=-1)
                if keep_valid_probs:
                    output_probs[:, token_idx, frame_idx, 0] = lprobs[:, 0]
                    if token_idx == ntoken - 1:
                        output_probs[:, token_idx, frame_idx, 1] = lprobs[:, tokens[token_idx]]
                    else:
                        output_probs[:, token_idx, frame_idx, 1] = lprobs[:, tokens[token_idx + 1]]
                else:
                    output_probs[:, token_idx, frame_idx, :] = lprobs
        return output_probs, encoder_mask

    @staticmethod
    def edit_distance(ref, hyp):
        '''
        edit distance
        '''
        # pylint:disable=too-many-branches
        # pylint:disable=no-else-break
        assert isinstance(ref, list) and isinstance(hyp, list)

        dist = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.uint32)
        for i in range(len(ref) + 1):
            for j in range(len(hyp) + 1):
                if i == 0:
                    dist[0][j] = j
                elif j == 0:
                    dist[i][0] = i
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dist[i][j] = dist[i - 1][j - 1]
                else:
                    substitute = dist[i - 1][j - 1] + 1
                    insert = dist[i][j - 1] + 1
                    delete = dist[i - 1][j] + 1
                    dist[i][j] = min(substitute, insert, delete)
        i = len(ref)
        j = len(hyp)
        steps = []
        while True:
            if i == 0 and j == 0:
                break
            elif (
                i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] and ref[i - 1] == hyp[j - 1]
            ):
                steps.append('corr')
                i, j = i - 1, j - 1
            elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] + 1:
                assert ref[i - 1] != hyp[j - 1]
                steps.append('sub')
                i, j = i - 1, j - 1
            elif j >= 1 and dist[i][j] == dist[i][j - 1] + 1:
                steps.append('ins')
                j = j - 1
            else:
                assert i >= 1 and dist[i][j] == dist[i - 1][j] + 1
                steps.append('del')
                i = i - 1
        steps = steps[::-1]

        counter = Counter({'words': len(ref), 'corr': 0, 'sub': 0, 'ins': 0, 'del': 0})
        counter.update(steps)

        return dist, steps, counter

    def cal_rnnt_edit_dist(self, hyp_tgt, target, target_mask):
        '''
        calculate rnnt edit distance
        '''
        error_dist = 0.0
        total_dist = 0.0
        bsz = hyp_tgt.shape[0]
        for bid in range(bsz):
            curr_hyp_tgt = hyp_tgt[bid]
            neaten_hyp_tgt = []
            prev_hyp = -1
            for idx in range(curr_hyp_tgt.size(0)):
                item = (curr_hyp_tgt[idx]).item()
                if item != prev_hyp:
                    prev_hyp = item
                if item != 0:
                    neaten_hyp_tgt.append(item)
            curr_tgt = target[bid]
            curr_mask = (target_mask[bid]).bool()
            # rnnt mask is more than 1 with respect to target shape
            neaten_tgt = torch.masked_select(curr_tgt, curr_mask)
            neaten_tgt = [item.item() for item in neaten_tgt]
            _, _, counter = self.edit_distance(neaten_tgt, neaten_hyp_tgt)
            error_dist += counter['sub'] + counter['ins'] + counter['del']
            total_dist += counter['words']
        return error_dist, total_dist

    def cal_ctc_edit_dist(self, hyp_tgt, target, target_mask):
        '''
        calculate ctc edit distance
        '''
        error_dist = 0.0
        total_dist = 0.0
        bsz = hyp_tgt.shape[0]
        for bid in range(bsz):
            curr_hyp_tgt = hyp_tgt[bid]
            neaten_hyp_tgt = []
            prev_hyp = -1
            for idx in range(curr_hyp_tgt.size(0)):
                item = (curr_hyp_tgt[idx]).item()
                if item != prev_hyp:
                    prev_hyp = item
                    if item != 0:
                        neaten_hyp_tgt.append(item)
            curr_tgt = target[bid]
            curr_mask = (target_mask[bid]).bool()
            # rnnt mask is more than 1 with respect to target shape
            neaten_tgt = torch.masked_select(curr_tgt, curr_mask)
            neaten_tgt = [item.item() for item in neaten_tgt]
            _, _, counter = self.edit_distance(neaten_tgt, neaten_hyp_tgt)
            error_dist += counter['sub'] + counter['ins'] + counter['del']
            total_dist += counter['words']
        return error_dist, total_dist

    @staticmethod
    def cal_frame_acc(logits, target, target_mask):
        '''
        calculate ce frame acc
        '''
        hyp_tgt = logits.max(dim=-1)[1]
        hyp_acc = ((hyp_tgt == target).float() * (target_mask.float())).sum() / (
            target_mask.float().sum() + 1e-6
        )
        return hyp_acc

    def cal_rnnt_cer_stats(self, encoder_out, target, target_mask):
        '''
        calculate rnnt cer statistics
        '''
        with torch.no_grad():
            hyp_tgt = self.greedy_decode(encoder_out)
            error_dist, total_dist = self.cal_rnnt_edit_dist(hyp_tgt, target, target_mask)
            return error_dist, total_dist

    def cal_ctc_cer_stats(self, logits, target, target_mask):
        '''
        calculate ctc cer statistics
        '''
        with torch.no_grad():
            hyp_tgt = logits.max(dim=-1)[1]
            error_dist, total_dist = self.cal_ctc_edit_dist(hyp_tgt, target, target_mask)
            return error_dist, total_dist

    def load_cmvn(self, mean, inv_std):
        '''
        Load cmvn for model export
        '''
        self.eval()
        mvn_concat_size = self.args.get('mvn_concat_size', 1)
        with torch.no_grad():
            self.encoder.mean.copy_(torch.tensor(np.tile(mean, mvn_concat_size)))
            self.encoder.inv_std.copy_(torch.tensor(np.tile(inv_std, mvn_concat_size)))

    def register_infers(self):
        infers = [
            KwsRnntEncoderExporter(self.encoder, **self.args),
        ]

        # add predictor
        if self.args.get('export_predictor_out', True):
            default_predictor_out_num = self.args.tgt_vocab_size
            if 'Adaptive' in self.args.jointer_type:
                default_predictor_out_num = self.args.adaptive_head_size + 1
            predictor_out_num = self.args.get('predictor_out_num', default_predictor_out_num)
            prev_tgt = torch.tensor(range(predictor_out_num)).long().cuda()
            with torch.no_grad():
                # pylint:disable=unsubscriptable-object
                outputs = self.predictor(prev_tgt)
                embed_tokens = nn.Embedding(outputs.shape[0], outputs.shape[1]).cuda()
                embed_tokens.state_dict()['weight'].copy_(outputs)
                predictor = KwsRnntPredictorExporter(embed_tokens, **self.args)
        else:
            predictor = KwsRnntPredictorExporter(self.predictor, **self.args)
        infers.append(predictor)

        # add jointer
        if 'Adaptive' in self.args.jointer_type:
            if self.args.get('export_jointer_adaptive_head', True):
                jointer = KwsRnntJointerExporter(
                    self.jointer.joint_module,
                    adaptive_head_module=self.jointer.log_softmax_fc.head,
                    **self.args,
                )
            else:
                jointer = KwsRnntJointerExporter(
                    self.jointer.joint_module,
                    adaptive_module=self.jointer.log_softmax_fc,
                    **self.args,
                )

        else:
            jointer = KwsRnntJointerExporter(self.jointer.joint_module, **self.args)
        infers.append(jointer)

        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
