''' RNN-T solution for dual-channel ASR '''
import torch
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.layers.time_reduce_layer import *
from core.solutions.base_solution import register_solution
from core.solutions.inference import INFERS, BaseInfer
from core.solutions.inference.utils import rnnt_rlt_neaten
from core.utils import FalconDict
from .base_rnnt_model import (
    BaseRnntModel,
    RnntPredictorExporter,
    RnntJointerExporter,
)


class RnntDualChannelEncoderExporter(BaseInfer):
    '''export rnnt encoder to onnx'''

    NAME = 'encoder'
    INPUTS = [
        FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='fbank_mask', type=torch.float32, shape=['B', 'T']),
    ]
    OUTPUTS = [
        FalconDict(name='output1', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='output2', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='backbone_mask', type=torch.float32, shape=['B', 'T']),
    ]

    ORIGIN_TEST_ATOL = 5e-4
    ORIGIN_TEST_RTOL = 1e-3
    OPTIMIZED_TEST_ATOL = 5e-4
    OPTIMIZED_TEST_RTOL = 1e-3

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.acoustic_front_end_module = model.acoustic_front_end_module
        self.acoustic_head_module = model.acoustic_head_module
        self.backbone_pool_module = model.backbone_pool_module
        self.acoustic_backbone_module = model.acoustic_backbone_module
        self.acoustic_backbone_sd1_module = model.acoustic_backbone_sd1_module
        self.acoustic_backbone_sd2_module = model.acoustic_backbone_sd2_module
        self.jointer_hidden_size = kwargs.get('jointer_hidden_size')
        self._convert_stream_flag = False
        self._stack_frame = kwargs.get('onnx_stack_frame', 80)
        self._inputs[0].shape[2] = self._stack_frame

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        fbank = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        fbank_mask = self._generate_input_data(1, method='ones', dynamic_axis=[1, 512])
        datas = [fbank, fbank_mask]
        return tuple(datas)

    def forward(self, fbank, fbank_mask):
        '''
        Forward for RNN-T base module
        '''
        bsz = fbank.size(0)
        front_end_out, backbone_mask, frontend_shape = self.acoustic_front_end_module(
            fbank, fbank_mask
        )
        backbone_out1 = self.acoustic_backbone_sd1_module(
            front_end_out, backbone_mask, frontend_shape=frontend_shape
        )
        backbone_out2 = self.acoustic_backbone_sd2_module(
            front_end_out, backbone_mask, frontend_shape=frontend_shape
        )
        backbone_out = torch.cat((backbone_out1, backbone_out2), dim=0)
        backbone_mask_repeat = backbone_mask.repeat(2, 1)
        backbone_out = self.acoustic_backbone_module(
            backbone_out, mask=backbone_mask_repeat, frontend_shape="BTN"
        )
        encoder_out = self.acoustic_head_module(backbone_out)
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
        encoder_out = encoder_out.view(2, bsz, -1, self.jointer_hidden_size).contiguous()
        return encoder_out[0, :, :, :], encoder_out[1, :, :, :], backbone_mask


@register_solution("RnntDualChannelModel")
class RnntDualChannelModel(BaseRnntModel):
    '''RNN-T Dual Channel Model'''

    def __init__(self, args):
        '''init function for RNN-T skeleton model.'''
        super().__init__(args)
        self.args = args
        self.update_steps = 0
        self.acoustic_backbone_sd1_module = eval(args.acoustic_backbone_sd_type)(args)
        self.acoustic_backbone_sd2_module = eval(args.acoustic_backbone_sd_type)(args)
        self.stage = args.get('stage', 'train')
        self.ctc_lower_layer = args.get("ctc_lower_layer", False)

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''
        speaker shared model for RNN-T encoder for recognition
        '''
        backbone_out = self.acoustic_backbone_module(inputs, mask, frontend_shape=frontend_shape)
        return backbone_out

    def separate(self, batch_data):
        '''separate'''
        if self.stage in ('train', 'pretrain_1ch'):
            fbank = batch_data['src']  # (B, T, ndim)
            fbank_mask = batch_data['src_mask']
            # front-end
            front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
            # backbone
            backbone_out1 = self.acoustic_backbone_sd1_module(
                front_end_out, backbone_mask, frontend_shape=frontend_shape
            )
            backbone_out2 = self.acoustic_backbone_sd2_module(
                front_end_out, backbone_mask, frontend_shape=frontend_shape
            )
            if self.stage == 'pretrain_1ch':
                batch_data['char1'] = batch_data['char']
                batch_data['char2'] = batch_data['char']
                batch_data['prev_char1'] = batch_data['prev_char']
                batch_data['prev_char2'] = batch_data['prev_char']
                batch_data['char1_mask'] = batch_data['char_mask']
                batch_data['char2_mask'] = batch_data['char_mask']
        elif self.stage == 'pretrain_2ch':
            fbank1 = batch_data['src1']  # (B, T, ndim)
            fbank2 = batch_data['src2']  # (B, T, ndim)
            fbank_mask = batch_data['src_mask']
            front_end_out1, backbone_mask, frontend_shape1 = self.frontend(fbank1, fbank_mask)
            front_end_out2, _, frontend_shape2 = self.frontend(fbank2, fbank_mask)
            backbone_out1 = self.acoustic_backbone_sd1_module(
                front_end_out1, backbone_mask, frontend_shape=frontend_shape1
            )
            backbone_out2 = self.acoustic_backbone_sd2_module(
                front_end_out2, backbone_mask, frontend_shape=frontend_shape2
            )
        else:
            raise RuntimeError("stage {} does not supported".format(self.stage))
        return backbone_out1, backbone_out2, backbone_mask

    def encoder(self, backbone_in, backbone_mask):
        # backbone
        backbone_out = self.encoder_backbone(backbone_in, backbone_mask)
        # ctc mtl branch
        ctc_logits = None
        if self.mtl_module is not None and self.training:
            if self.ctc_lower_layer:
                ctc_logits = self.mtl_module(backbone_in)
            else:
                ctc_logits = self.mtl_module(backbone_out)
        # head
        encoder_out = self.acoustic_head_module(backbone_out)
        return encoder_out, ctc_logits

    def forward(self, batch_data, inference=False):
        '''
        Forward for RNN-T dual-channel module
        '''
        if self.training:
            self.update_steps += 1
        ############ Encoder #############
        backbone_out1, backbone_out2, backbone_mask = self.separate(batch_data)
        encoder_out1, ctc_logits1 = self.encoder(backbone_out1, backbone_mask)
        encoder_out2, ctc_logits2 = self.encoder(backbone_out2, backbone_mask)
        if self.backbone_pool_module is not None:
            if backbone_mask is not None:
                batch_data['mtl_mask'] = backbone_mask.clone()
                backbone_mask = backbone_mask[:, :-1][:, ::2]
            encoder_out1 = self.backbone_pool_module(encoder_out1)
            encoder_out2 = self.backbone_pool_module(encoder_out2)
        batch_data['backbone_mask'] = backbone_mask
        if inference:
            return encoder_out1, encoder_out2, backbone_mask
        ############ Predictor ############
        predictor_out1 = self.predictor(batch_data, trainable=self.training, key='prev_char1')
        predictor_out2 = self.predictor(batch_data, trainable=self.training, key='prev_char2')
        target_lengths1 = batch_data['char1_mask'].sum(dim=1).int()
        target_lengths2 = batch_data['char2_mask'].sum(dim=1).int()
        ############ Jointer ############
        jointer_out1 = self.jointer(encoder_out1, predictor_out1, None, target_lengths1)
        jointer_out2 = self.jointer(encoder_out2, predictor_out2, None, target_lengths2)

        # criterion for loss computation
        forward_out = self.criterion_module(
            jointer_out1,
            jointer_out2,
            batch_data,
            encoder_out1,
            encoder_out2,
            ctc_logits1=ctc_logits1,
            ctc_logits2=ctc_logits2,
        )
        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out1.requires_grad:
            target = batch_data['char1']
            error_dist1, total_dist1 = self.greedy_searcher(encoder_out1, target, batch_data)
            target = batch_data['char2']
            error_dist2, total_dist2 = self.greedy_searcher(encoder_out2, target, batch_data)
            forward_out['cer'] = error_dist1 + error_dist2
            forward_out['dist'] = total_dist1 + total_dist2
        return forward_out

    @torch.no_grad()
    def greedy_inference(self, batch_data):
        '''
        Greedy inference
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        ############ Acoustic ############
        acoustic_out1, acoustic_out2, _ = self.forward(batch_data, inference=True)

        greedy_infer_rlt1 = self.criterion_module.greedy_infer(
            acoustic_out1, self.predictor_module, self.jointer_module
        )
        greedy_infer_rlt2 = self.criterion_module.greedy_infer(
            acoustic_out2, self.predictor_module, self.jointer_module
        )
        out_rlt_list1 = []
        out_rlt_list2 = []
        for bid in range(bsz):
            hyp_token_list = greedy_infer_rlt1[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
            out_rlt_list1.append(hyp_token_list)
            hyp_token_list = greedy_infer_rlt2[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
            out_rlt_list2.append(hyp_token_list)
        return out_rlt_list1, out_rlt_list2

    @torch.no_grad()
    def beam_inference(self, batch_data, **kwargs):
        '''
        Beam inference
        '''
        # pylint: disable=unused-argument
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        ############ Acoustic ############
        acoustic_out1, acoustic_out2, backbone_mask = self.forward(batch_data, inference=True)
        # engine beam search has been deleted
        beam_infer_rlt1, beam_infer_rlt1_nbest = self.beam_searcher(acoustic_out1, backbone_mask)
        beam_infer_rlt2, beam_infer_rlt2_nbest = self.beam_searcher(acoustic_out2, backbone_mask)
        out_rlt_list1 = []
        out_rlt_list2 = []
        if kwargs.get("output_timestamp", False):
            for bid_rlt in beam_infer_rlt1_nbest['nbest']:
                out_rlt_list1.append((bid_rlt[0][0][1:], bid_rlt[0][3][1:]))
            for bid_rlt in beam_infer_rlt2_nbest['nbest']:
                out_rlt_list2.append((bid_rlt[0][0][1:], bid_rlt[0][3][1:]))
        else:
            for bid in range(bsz):
                hyp_token_list = beam_infer_rlt1[bid]
                hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
                out_rlt_list1.append(hyp_token_list)
                hyp_token_list = beam_infer_rlt2[bid]
                hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
                out_rlt_list2.append(hyp_token_list)
        return out_rlt_list1, out_rlt_list2

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        infers = [
            RnntPredictorExporter(self.predictor_module, **self.args),
            RnntJointerExporter(self.jointer_module, self.criterion_module, **self.args),
            RnntDualChannelEncoderExporter(self, **self.args),
        ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)


class RnntDualChannelMaskEncoderExporter(BaseInfer):
    '''export rnnt encoder to onnx'''

    NAME = 'encoder'
    INPUTS = [
        FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='fbank_mask', type=torch.float32, shape=['B', 'T']),
    ]
    OUTPUTS = [
        FalconDict(name='output1', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='output2', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='backbone_mask', type=torch.float32, shape=['B', 'T']),
    ]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.acoustic_front_end_module = model.acoustic_front_end_module
        self.acoustic_head_module = model.acoustic_head_module
        self.backbone_pool_module = model.backbone_pool_module
        self.acoustic_backbone_module = model.acoustic_backbone_module
        self.acoustic_backbone_sd1_module = model.acoustic_backbone_sd1_module
        self.acoustic_backbone_sd2_module = model.acoustic_backbone_sd2_module
        self.mask_projection_module = model.mask_projection_module
        self.mask_layer_norm = model.mask_layer_norm
        self.jointer_hidden_size = kwargs.get('jointer_hidden_size')
        self._convert_stream_flag = False
        self._stack_frame = kwargs.get('onnx_stack_frame', 80)
        self._inputs[0].shape[2] = self._stack_frame

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        fbank = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        fbank_mask = self._generate_input_data(1, method='ones', dynamic_axis=[1, 512])
        datas = [fbank, fbank_mask]
        return tuple(datas)

    def forward(self, fbank, fbank_mask):
        '''
        Forward for RNN-T base module
        '''
        bsz = fbank.size(0)
        front_end_out, backbone_mask, frontend_shape = self.acoustic_front_end_module(
            fbank, fbank_mask
        )
        mask_out = self.acoustic_backbone_sd1_module(
            front_end_out, acoustic_mask=backbone_mask, frontend_shape=frontend_shape
        )
        embedding_out = self.acoustic_backbone_sd2_module(
            front_end_out, acoustic_mask=backbone_mask, frontend_shape=frontend_shape
        )
        if self.mask_layer_norm is not None:
            mask_out = self.mask_layer_norm(mask_out)
        mask_out = torch.sigmoid(self.mask_projection_module(mask_out))
        backbone_out1 = mask_out * embedding_out
        backbone_out2 = (1 - mask_out) * embedding_out
        backbone_out = torch.cat((backbone_out1, backbone_out2), dim=0)
        backbone_mask_repeat = backbone_mask.repeat(2, 1)
        backbone_out = self.acoustic_backbone_module(
            backbone_out, acoustic_mask=backbone_mask_repeat, frontend_shape="BTN"
        )
        encoder_out = self.acoustic_head_module(backbone_out)
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
        encoder_out = encoder_out.view(2, bsz, -1, self.jointer_hidden_size).contiguous()
        return encoder_out[0, :, :, :], encoder_out[1, :, :, :], backbone_mask


class RnntDualChannelMaskSeparateExporter(BaseInfer):
    '''export rnnt encoder to onnx'''

    NAME = 'encoder_separate'
    INPUTS = [
        FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='fbank_mask', type=torch.float32, shape=['B', 'T']),
    ]
    OUTPUTS = [
        FalconDict(name='output1', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='output2', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='backbone_mask', type=torch.float32, shape=['B', 'T']),
    ]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.acoustic_front_end_module = model.acoustic_front_end_module
        self.acoustic_backbone_sd1_module = model.acoustic_backbone_sd1_module
        self.acoustic_backbone_sd2_module = model.acoustic_backbone_sd2_module
        self.mask_projection_module = model.mask_projection_module
        self.mask_layer_norm = model.mask_layer_norm
        self._convert_stream_flag = False
        self._stack_frame = kwargs.get('onnx_stack_frame', 80)
        self._inputs[0].shape[2] = self._stack_frame

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        fbank = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        fbank_mask = self._generate_input_data(1, method='ones', dynamic_axis=[1, 512])
        datas = [fbank, fbank_mask]
        return tuple(datas)

    def forward(self, fbank, fbank_mask):
        '''
        Forward for RNN-T base module
        '''
        front_end_out, backbone_mask, frontend_shape = self.acoustic_front_end_module(
            fbank, fbank_mask
        )
        mask_out = self.acoustic_backbone_sd1_module(
            front_end_out, backbone_mask, frontend_shape=frontend_shape
        )
        embedding_out = self.acoustic_backbone_sd2_module(
            front_end_out, backbone_mask, frontend_shape=frontend_shape
        )
        if self.mask_layer_norm is not None:
            mask_out = self.mask_layer_norm(mask_out)
        mask_out = torch.sigmoid(self.mask_projection_module(mask_out))
        backbone_out1 = mask_out * embedding_out
        backbone_out2 = (1 - mask_out) * embedding_out
        return backbone_out1, backbone_out2, backbone_mask


class RnntDualChannelMaskRecognitionExporter(BaseInfer):
    '''export rnnt encoder to onnx'''

    NAME = 'encoder_recognition'
    INPUTS = [
        FalconDict(name='encoder', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='encoder_mask', type=torch.float32, shape=['B', 'T']),
    ]
    OUTPUTS = [
        FalconDict(name='output', type=torch.float32, shape=['B', 'T', -1]),
    ]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.acoustic_head_module = model.acoustic_head_module
        self.backbone_pool_module = model.backbone_pool_module
        self.acoustic_backbone_module = model.acoustic_backbone_module
        self._inputs[0].shape[2] = kwargs.get("backbone_memory_size")
        self._convert_stream_flag = False

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        encoder = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        encoder_mask = self._generate_input_data(1, method='ones', dynamic_axis=[1, 512])
        datas = [encoder, encoder_mask]
        return tuple(datas)

    def forward(self, encoder, encoder_mask):
        '''
        Forward for RNN-T base module
        '''
        backbone_out = self.acoustic_backbone_module(encoder, encoder_mask, frontend_shape="BTN")
        encoder_out = self.acoustic_head_module(backbone_out)
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
        return encoder_out


@register_solution("RnntDualChannelMaskModel")
class RnntDualChannelMaskModel(RnntDualChannelModel):
    '''RnntDualChannelMaskModel'''

    def __init__(self, args):
        '''init function for RNN-T skeleton model.'''
        super().__init__(args)
        self.mask_layer_norm = None
        if args.get("mask_layer_norm", False):
            self.mask_layer_norm = nn.LayerNorm(args.backbone_memory_size)
        self.mask_projection_module = nn.Linear(
            args.backbone_memory_size, args.backbone_memory_size
        )

    def separate(self, batch_data):
        if self.stage == 'train':
            fbank = batch_data['src']  # (B, T, ndim)
            fbank_mask = batch_data['src_mask']
            # front-end
            front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
            # backbone
            mask_out = self.acoustic_backbone_sd1_module(
                front_end_out, backbone_mask, frontend_shape=frontend_shape
            )
            embedding_out = self.acoustic_backbone_sd2_module(
                front_end_out, backbone_mask, frontend_shape=frontend_shape
            )
            if self.mask_layer_norm is not None:
                mask_out = self.mask_layer_norm(mask_out)
            mask_out = torch.sigmoid(self.mask_projection_module(mask_out))
            backbone_out1 = mask_out * embedding_out
            backbone_out2 = (1 - mask_out) * embedding_out
        else:
            raise RuntimeError("stage {} does not supported".format(self.stage))
        return backbone_out1, backbone_out2, backbone_mask

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        infers = [
            RnntPredictorExporter(self.predictor_module, **self.args),
            RnntJointerExporter(self.jointer_module, self.criterion_module, **self.args),
            RnntDualChannelMaskEncoderExporter(self, **self.args),
            RnntDualChannelMaskSeparateExporter(self, **self.args),
            RnntDualChannelMaskRecognitionExporter(self, **self.args),
        ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
