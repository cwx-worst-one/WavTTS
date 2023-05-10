''' BaseRnntModel and BaseRnntSolution '''
# pylint: disable=abstract-method
import copy
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.solutions.base_solution import register_solution
from core.solutions.inference import INFERS, BaseInfer
from core.utils import FalconDict
from .base_rnnt_model import (
    BaseRnntModel,
    RnntEncoderExporter,
    RnntPredictorExporter,
    RnntJointerExporterForLM,
)


class RnntCeEncoderExporter(BaseInfer):
    '''export rnnt ce_encoder to onnx'''

    NAME = 'ce_encoder'
    INPUTS = [
        FalconDict(name='ce_input', type=torch.float32, shape=['B', 'T', 'D']),
    ]
    OUTPUTS = [FalconDict(name='ce_output', type=torch.float32, shape=['B', 'T', -1])]
    ORIGIN_TEST_RTOL = 1e1
    OPTIMIZED_TEST_ATOL = 1e1

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.align_head_module = model.align_head_module.cuda()
        self.extra_ce_encoder = model.extra_ce_encoder.cuda()
        self.args = model.args
        self.eval()

    def forward(self, ce_input):
        '''
        Forward for RNN-T base module
        '''
        extra_encoder_out = self.extra_ce_encoder(ce_input)
        ce_out = self.align_head_module(extra_encoder_out)
        ce_logits = F.log_softmax(ce_out.float(), dim=-1)
        return ce_logits

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        fbank = self._generate_input_data(0, method='rand', dynamic_axis=[1, 128, 512])
        return fbank


@register_solution("BaseRnntModelAddCe")
class BaseRnntModelAddCe(BaseRnntModel):
    '''
    Base DFSMN RNN-T add CE task.
    '''

    def __init__(self, args):
        self.args = args
        self.args.encoder_fix = True
        super().__init__(args)
        self.align_head_module = eval(args.align_head_type)(args)
        self.criterion_module_align = eval(args.align_criterion_type)(args)
        extra_args = copy.deepcopy(args)
        if extra_args.extra_ce_encoder == 'MaskedConformerBackbone':
            extra_args.conformer_mask_topology = args.extra_conformer_mask_topology
            extra_args.conformer_num_blocks = args.extra_conformer_num_blocks
        elif extra_args.extra_ce_encoder == 'DFSMNBackboneLN':
            extra_args.backbone_topology = args.extra_ce_topology
        self.extra_ce_encoder = eval(args.extra_ce_encoder)(extra_args)

    def forward(self, batch_data, inference=False):
        self.acoustic_backbone_module.eval()
        if inference:
            encoder_out, backbone_mask, mtl_logits, encoder_backbone_out = super().forward(
                batch_data, inference
            )
            return encoder_out, backbone_mask, mtl_logits, encoder_backbone_out
        # encoder's param is freezed by setting encoder_fix to True
        encoder_out, backbone_mask, _, mtl_logits, encoder_backbone_out = super().encoder(
            batch_data
        )
        extra_encoder_out = self.extra_ce_encoder(encoder_backbone_out, backbone_mask)
        encoder_out = self.align_head_module(extra_encoder_out)
        batch_data['backbone_mask'] = backbone_mask

        src_mask = batch_data['src_mask']
        bsz = src_mask.shape[0]
        target = batch_data['ce_label']
        target = target.view(bsz, -1, self.args.downsampling_size)[:, :, 0]
        target_mask = batch_data['backbone_mask']
        forward_out = self.criterion_module_align(encoder_out, src_mask, target, target_mask)
        return forward_out

    @torch.no_grad()
    def beam_inference(
        self,
        batch_data,
        nbest=1,
        nbest_align_info=False,
        prefetch=False,
        enable_las_rescore=False,
        output_timestamp=False,
        **_kwargs,
    ):
        encoder_out, backbone_mask, mtl_logits, encoder_backbone_out = self.forward(
            batch_data, inference=True
        )
        batch_data['encoder_out'] = encoder_out
        batch_data['backbone_mask'] = backbone_mask
        batch_data['mtl_logits'] = mtl_logits
        batch_data['encoder_backbone_out'] = encoder_backbone_out
        inf_res, _, frames = super().beam_inference(
            batch_data,
            nbest=nbest,
            nbest_align_info=nbest_align_info,
            enable_las_rescore=enable_las_rescore,
            output_timestamp=output_timestamp,
            prefetch=prefetch,
        )
        extra_encoder_out = self.extra_ce_encoder(encoder_backbone_out, backbone_mask)
        encoder_out = self.align_head_module(extra_encoder_out)
        lprobs = F.log_softmax(encoder_out.float(), dim=-1)
        return inf_res, lprobs, frames

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        infers = [
            RnntEncoderExporter(self, return_backbone=True, **self.args),
            RnntCeEncoderExporter(self, **self.args),
            RnntPredictorExporter(self.predictor_module, **self.args),
            RnntJointerExporterForLM(self.jointer_module, self.criterion_module, **self.args),
        ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
