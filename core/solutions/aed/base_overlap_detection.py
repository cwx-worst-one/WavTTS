''' BaseOverlapDetectionModel '''
import torch
from torch import nn
from core.solutions.base_solution import BaseSolution, register_solution
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.layers.pooling import *
from core.criterions import *
from core.solutions.inference import BaseInfer, INFERS
from core.utils import FalconDict


class BaseOverlapDetectionExporter(BaseInfer):
    '''export rnnt encoder to onnx'''

    NAME = 'base_overlap_detecetion'
    INPUTS = [FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['B', -1]),
        FalconDict(name='frame_encoder_out', type=torch.float32, shape=['B', 'T', -1]),
    ]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.acoustic_front_end_module = model.acoustic_front_end_module
        self.acoustic_backbone_module = model.acoustic_backbone_module
        self.acoustic_head_module = model.acoustic_head_module
        self.backbone_pool_module = model.backbone_pool_module
        self.final_fc_module = model.final_fc_module
        self.frame_fc_module = model.frame_fc_module
        self.args = model.args
        self.frame_soft_mask = self.args.get("frame_soft_mask", False)
        self.eval()
        onnx_stack_frame = kwargs.get('downsampling_size') * kwargs.get('onnx_stack_frame', 60)
        self._inputs[0].shape[2] = onnx_stack_frame

    def forward(self, fbank):
        '''
        Forward for RNN-T base module
        '''
        bsz, frames = fbank.size(0), fbank.size(1)
        fbank = fbank.view(bsz, self.args.downsampling_size * frames, -1)
        encoder_out, _, frontend_shape = self.acoustic_front_end_module(fbank, None)
        encoder_out = self.acoustic_backbone_module(
            encoder_out, None, frontend_shape=frontend_shape
        )
        encoder_out = self.acoustic_head_module(encoder_out)
        frame_encoder_out = self.frame_fc_module(encoder_out)
        encoder_out = self.backbone_pool_module(encoder_out)
        encoder_out = self.final_fc_module(encoder_out)
        return encoder_out.softmax(dim=-1), frame_encoder_out.softmax(dim=-1)

    def sample_inputs(self):
        fbank = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        return fbank


@register_solution("BaseOverlapDetectionModel")
class BaseOverlapDetectionModel(BaseSolution):
    '''overlap detection'''

    def __init__(self, args):
        '''init of overlap detection model'''
        super().__init__()
        self.args = args
        self.acoustic_front_end_module = eval(args.front_end_type)(args)
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)
        self.acoustic_head_module = eval(args.head_type)(args)
        self.backbone_pool_module = eval(args.backbone_pool_type)()
        self.final_fc_module = nn.Linear(args.jointer_hidden_size, args.tgt_size)
        self.frame_mtl = args.get("frame_mtl", False)
        if self.frame_mtl:
            self.frame_fc_module = nn.Linear(args.jointer_hidden_size, args.tgt_size)
            self.frame_mtl_scale = args.get("frame_mtl_scale", 1.0)
        self.frame_soft_mask = args.get("frame_soft_mask", False)
        self.criterion_module = eval(args.criterion_type)(args)
        self.update_steps = 0

    def forward(self, batch_data, inference=False):
        """forward"""
        if self.training:
            self.update_steps += 1
        fbank = batch_data['src']
        target = batch_data['target']
        fbank_mask = batch_data['src_mask']
        frame_target = batch_data['frame_target'] if 'frame_target' in batch_data else None
        encoder_out, encoder_mask, frontend_shape = self.acoustic_front_end_module(
            fbank, fbank_mask
        )
        encoder_out = self.acoustic_backbone_module(
            encoder_out, encoder_mask, frontend_shape=frontend_shape
        )
        encoder_out = self.acoustic_head_module(encoder_out)
        frame_loss = 0.0
        frame_encoder_out = None
        if self.frame_mtl and (frame_target is not None or inference):
            t_size = encoder_out.size(1)
            frame_encoder_out = self.frame_fc_module(encoder_out)
            if not inference:
                frame_out = self.criterion_module(
                    frame_encoder_out, encoder_mask, frame_target[:, :t_size], encoder_mask
                )
                frame_loss = self.frame_mtl_scale * frame_out['loss']
        if self.frame_soft_mask:
            soft_mask = frame_encoder_out.softmax(dim=-1)[:, :, -1]
            if self.args.backbone_pool_type == 'AvgPoolLayer':
                encoder_mask = encoder_mask * soft_mask
            else:
                encoder_out = encoder_out * soft_mask.unsqueeze(2)
        encoder_out = self.backbone_pool_module(encoder_out, encoder_mask)
        encoder_out = self.final_fc_module(encoder_out)
        if inference:
            return encoder_out, frame_encoder_out, encoder_mask
        tmp_mask = torch.ones((encoder_out.size(0), 1), device=encoder_out.device)
        forward_out = self.criterion_module(
            encoder_out.unsqueeze(1), tmp_mask, target.unsqueeze(1), tmp_mask
        )
        forward_out["frame_size"] = encoder_mask.sum().float()
        forward_out["backward_loss"] += frame_loss
        forward_out["loss"] += frame_loss
        return forward_out

    @torch.no_grad()
    def inference(self, batch_data):
        '''inference with frame-level logits'''
        encoder_out, frame_encoder_out, encoder_mask = self.forward(batch_data, inference=True)
        predict_score = encoder_out.softmax(dim=-1)[:, 1].tolist()
        predict_frame_score = None
        if frame_encoder_out is not None:
            predict_frame_score = frame_encoder_out.softmax(dim=-1)[:, :, 1]
            predict_frame_score.masked_fill((encoder_mask > 0), 0.0)
        return predict_score, predict_frame_score

    def register_infers(self):
        infers = [BaseOverlapDetectionExporter(self, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
