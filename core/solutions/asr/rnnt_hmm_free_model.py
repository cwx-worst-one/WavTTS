''' RNN-T HMM-Free Training solution '''
# pylint: disable=invalid-name
import torch
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.layers.time_reduce_layer import *
from core.solutions.base_solution import register_solution
from core.solutions.inference.rnnt_greedy_search import GreedySearch
from .base_rnnt_model import BaseRnntModel


@register_solution("RnntHmmFreeModel")
class RnntHmmFreeModel(BaseRnntModel):
    '''RNN-T HMM-Free'''

    def __init__(self, args):
        '''init function for RNN-T HMM-Free model.'''
        super().__init__(args)
        self.args = args
        self.teacher_args = args
        # stage can be: teacher_rnnt_ctc, teacher_rnnt, teacher_ctc,
        #               student_ce, student_rnnt, student_rnnt_ctc, student_ctc
        self.stage = args.get('stage', 'teacher_ctc')
        self.acoustic_front_end_module = None
        self.acoustic_backbone_module = None
        self.acoustic_head_module = None
        self.predictor_module = None
        self.jointer_module = None
        self.ctc_module = None
        self.ce_module = None
        self.teacher_acoustic_front_end_module = None
        self.teacher_acoustic_backbone_module = None
        self.teacher_ctc_module = None
        self.teacher_acoustic_head_module = None

        if 'student' in self.stage:
            self._init_student_modules(args)
        elif 'teacher' in self.stage:
            self._init_teacher_modules(args)
        else:
            raise ValueError("student or teacher must in stage")
        self.criterion_module = eval(args.criterion_type)(args)
        self.greedy_searcher = GreedySearch(
            self.args, self.criterion_module, self.predictor_module, self.jointer_module
        )

    def _init_student_modules(self, args):
        '''init modules for student case'''
        self.acoustic_front_end_module = eval(args.front_end_type)(args)
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)
        if '_rnnt' in self.stage:
            self.acoustic_head_module = eval(args.head_type)(args)
            self.predictor_module = eval(args.predictor_type)(args)
            self.jointer_module = eval(args.jointer_type)(args)
        if '_ctc' in self.stage:
            self.ctc_module = eval(args.mtl_head)(args)
        if '_ce' in self.stage:
            self.ce_module = eval(args.mtl_head)(args)
            if args.teacher_front_end_type == 'TimeReduceLSTMP':
                self.teacher_args.backbone_bilstm = True
                self.teacher_args.backbone_hidden_size //= 2
            self.teacher_acoustic_front_end_module = eval(args.teacher_front_end_type)(
                self.teacher_args
            )
            self.teacher_acoustic_backbone_module = eval(args.teacher_acoustic_backbone_type)(
                self.teacher_args
            )
            self.teacher_ctc_module = eval(args.mtl_head)(args)

    def _init_teacher_modules(self, args):
        '''init modules for teacher case'''
        if args.teacher_front_end_type == 'TimeReduceLSTMP':
            self.teacher_args.backbone_bilstm = True
            self.teacher_args.backbone_hidden_size //= 2
        self.teacher_acoustic_front_end_module = eval(args.teacher_front_end_type)(
            self.teacher_args
        )
        self.teacher_acoustic_backbone_module = eval(args.teacher_acoustic_backbone_type)(
            self.teacher_args
        )
        if '_rnnt' in self.stage:
            self.teacher_acoustic_head_module = eval(args.teacher_head_type)(args)
            self.predictor_module = eval(args.predictor_type)(args)
            self.jointer_module = eval(args.jointer_type)(args)
        if '_ctc' in self.stage:
            self.teacher_ctc_module = eval(args.mtl_head)(args)

    def encoder_front_end_backbone(self, fbank, mask, stage='teacher'):
        '''frontend and backbone of encoder'''
        if stage == 'student':
            acoustic_front_end_module = self.acoustic_front_end_module
            acoustic_backbone_module = self.acoustic_backbone_module
        elif stage == 'teacher':
            acoustic_front_end_module = self.teacher_acoustic_front_end_module
            acoustic_backbone_module = self.teacher_acoustic_backbone_module
        else:
            raise ValueError("student or teacher must in stage")
        x, m, frontend_shape = acoustic_front_end_module(fbank, mask)
        x = acoustic_backbone_module(x, m, frontend_shape)
        # backbone_out shape should be (B, T, N)
        return x, m

    def encoder(self, batch_data):
        '''encoder forward'''
        fbank = batch_data['src']
        mask = batch_data['src_mask']
        ce_logits = None
        teacher_ctc_logits = None
        ctc_logits = None
        rnnt_acoustic = None
        if 'student' in self.stage:
            if not self.training or self.update_steps <= self.encoder_fix_steps:
                with torch.no_grad():
                    x, m = self.encoder_front_end_backbone(fbank, mask, stage='student')
            else:
                with torch.enable_grad():
                    x, m = self.encoder_front_end_backbone(fbank, mask, stage='student')
            if '_ce' in self.stage:
                ce_logits = self.ce_module(x)
                with torch.no_grad():
                    fbank_bak = batch_data['src_bak']
                    y, _ = self.encoder_front_end_backbone(fbank_bak, mask, stage='teacher')
                    teacher_ctc_logits = self.teacher_ctc_module(y)
            if '_ctc' in self.stage:
                ctc_logits = self.ctc_module(x)
            if '_rnnt' in self.stage:
                rnnt_acoustic = self.acoustic_head_module(x)
        elif 'teacher' in self.stage:
            x, m = self.encoder_front_end_backbone(fbank, mask, stage='teacher')
            if '_rnnt' in self.stage:
                rnnt_acoustic = self.teacher_acoustic_head_module(x)
            if '_ctc' in self.stage:
                ctc_logits = self.teacher_ctc_module(x)
        return rnnt_acoustic, m, ctc_logits, ce_logits, teacher_ctc_logits

    def forward(self, batch_data, inference=False):
        '''Forward for RNN-T HMM-Free module'''
        if self.training:
            self.update_steps += 1
        ############ Encoder #############
        encoder_out, backbone_mask, ctc_logits, ce_logits, teacher_ctc_logits = self.encoder(
            batch_data
        )
        batch_data['backbone_mask'] = backbone_mask
        if inference:
            return encoder_out, backbone_mask, ctc_logits, None
        ############ Predictor and Jointer ############
        jointer_out = None
        if self.predictor_module is not None and self.jointer_module is not None:
            predictor_out = self.predictor(batch_data, self.training)
            target_lengths = batch_data['target_lengths']
            jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)

        # criterion for loss computation
        forward_out = self.criterion_module(
            jointer_out,
            batch_data,
            ctc_logits=ctc_logits,
            ce_logits=ce_logits,
            teacher_ctc_logits=teacher_ctc_logits,
        )
        if (
            encoder_out is not None
            and self.predictor_module is not None
            and (
                self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad
            )
        ):
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out, target, batch_data
            )
        else:
            forward_out['cer'], forward_out['dist'] = 0.0, 0.0
        return forward_out
