''' BaseRnntModel and BaseRnntSolution '''
# pylint: disable=abstract-method
import copy
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *

from core.solutions.base_solution import register_solution
from core.solutions.inference import INFERS, BaseInfer
from core.utils import FalconDict
from .base_cif_model import (
    BaseCifModel,
    CifEncoderExporter,
    CifDecoderLogitsExporter,
)


class CifCeEncoderExporter(BaseInfer):
    '''export cif ce_encoder to onnx'''

    NAME = 'ce_encoder'
    INPUTS = [
        FalconDict(name='ce_input', type=torch.float32, shape=['B', 'T', 'D']),
    ]
    OUTPUTS = [FalconDict(name='ce_output', type=torch.float32, shape=['B', 'T', -1])]

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
        fbank = self._generate_input_data(0, dynamic_axis=[1, 128, 512])
        ce_inputs = [fbank]
        return tuple(ce_inputs)


@register_solution("BaseCifModelAddCe")
class BaseCifModelAddCe(BaseCifModel):
    '''
    Base CIF add CE task.
    '''

    def __init__(self, args):
        '''
        init function for CIF pretrain model.
        '''
        self.args = args
        super().__init__(args)
        self.align_head_module = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False)
        self.criterion_module_align = eval(args.align_criterion_type)(args)
        extra_args = copy.deepcopy(args)
        if extra_args.extra_ce_encoder == 'CifSelfAttentionEncoder':
            extra_args.num_encoder_layers = args.extra_num_encoder_layers
        elif self.args.front_end_type in ('Conv2dPooling') and self.args.extra_ce_encoder in (
            'ConformerBackbone',
            'MaskedConformerBackbone',
        ):
            extra_args.conformer_mask_topology = args.extra_conformer_mask_topology
            extra_args.conformer_num_blocks = args.extra_conformer_num_blocks
        self.extra_ce_encoder = eval(args.extra_ce_encoder)(extra_args)
        self.train()

    def train(self, mode: bool = True):
        '''
        Param will really be freezed by set param.requires_grad_(False).
        `with no_grad` may case param be updated by optimizer' weight_decay.
        What's more, we need to set module.eval() to fix some module buffer,
        such as BatchNorm.running_mean.
        '''
        super().train(mode)

        # encoder
        if self.args.front_end_type in ('Conv2dPooling') and self.args.extra_ce_encoder in (
            'ConformerBackbone',
            'MaskedConformerBackbone',
        ):
            self.front_end.requires_grad_(False)
            self.encoder_backbone.requires_grad_(False)
            self.front_end.train(False)
            self.encoder_backbone.train(False)
        elif self.args.extra_ce_encoder == 'CifSelfAttentionEncoder':
            self.front_end.requires_grad_(False)
            self.encoder_backbone.requires_grad_(False)
            self.endocer_dropout.requires_grad_(False)
            self.front_end.train(False)
            self.encoder_backbone.train(False)
            self.endocer_dropout.train(False)

    def forward(self, batch_data, inference=False):
        inputs = batch_data['src'].unsqueeze(-1)
        inputs_mask = batch_data['src_mask']
        extra_encoder_out, _, not_padding = self.encoder(inputs, inputs_mask)
        encoder_out = self.align_head_module(extra_encoder_out)

        if inference:
            return encoder_out, not_padding
        src_mask = batch_data['src_mask']
        bsz = src_mask.shape[0]
        target = batch_data['ce_label']
        target = target.view(bsz, -1, self.args.downsampling_size)[:, :, 0]
        forward_out = self.criterion_module_align(encoder_out, src_mask, target, not_padding)
        return forward_out

    def encoder(self, inputs, inputs_mask):
        if self.args.front_end_type in ('Conv2dPooling') and self.args.extra_ce_encoder in (
            'ConformerBackbone',
            'MaskedConformerBackbone',
        ):
            front_end_out, backbone_mask, frontend_shape = self.front_end(
                inputs.squeeze(-1), inputs_mask
            )
            encoder_backbone_out = self.encoder_backbone(
                front_end_out, backbone_mask, frontend_shape
            )
            encoder_outputs, not_padding = encoder_backbone_out, backbone_mask.int()
            extra_encoder_out = self.extra_ce_encoder(
                encoder_outputs, backbone_mask, frontend_shape
            )
        elif self.args.extra_ce_encoder == 'CifSelfAttentionEncoder':

            encoder_input, ignore_padding = self.front_end(inputs, inputs_mask)
            encoder_input = self.encoder_dropout(encoder_input)
            encoder_outputs, ignore_padding, _, _ = self.encoder_backbone(
                encoder_input, ignore_padding=ignore_padding, prev_kv_cache=None
            )
            extra_encoder_out, _, not_padding, _ = self.extra_ce_encoder(
                encoder_outputs, ignore_padding=ignore_padding, prev_kv_cache=None
            )
        return extra_encoder_out, encoder_outputs, not_padding

    @torch.no_grad()
    def fast_decode(self, inputs, inputs_mask):
        """
        fast_decode
        """
        hparams = self.args
        # Encoder part
        _, encoder_outputs, not_padding = self.encoder(inputs, inputs_mask)
        a = self.cif_weight_estimator(encoder_outputs, not_padding)

        if hparams.use_tail_handling:
            encoder_outputs = nn.functional.pad(encoder_outputs, [0, 0, 0, 1, 0, 0])
            not_padding = nn.functional.pad(not_padding, [0, 1, 0, 0])
            a = nn.functional.pad(a, [0, 1, 0, 0])

        # CIF part
        (
            cif_outputs,
            not_padding_after_cif,
            sum_a,  # pylint: disable=unused-variable
            _,
            cif_dict,  # pylint: disable=unused-variable
        ) = self.cif_calculator(encoder_outputs, not_padding, a)

        if self.beam_size > 1:
            tokens = self.beam_searcher(cif_outputs, not_padding_after_cif)
        else:
            # Decoder part
            logits = self.decoder(cif_outputs)

            # argmax
            tokens = torch.argmax(logits, dim=-1)
            tokens = tokens * not_padding_after_cif
        return tokens, not_padding_after_cif

    @torch.no_grad()
    def inference(
        self,
        batch_data,
        mode='test',
        beam_size=10,
        nbest=1,
        language=None,
        filter_list=None,
        concate_en_letters=None,
        **_kwargs,
    ):
        '''
        inference for cif add ce model.
        '''
        # pylint: disable=too-many-locals
        inputs = batch_data['src'].unsqueeze(-1)
        targets = batch_data['char']
        targets_mask = batch_data['char_mask']
        inputs_mask = batch_data['src_mask']

        self.beam_size = beam_size
        self.nbest = nbest

        token_ids, output_padding = self.fast_decode(inputs, inputs_mask)
        res_lengths = output_padding.sum(dim=1).int().tolist()
        target_lengths = targets_mask.sum(-1).int()

        encoder_out, padding_mask = self.forward(batch_data, inference=True)
        encoder_frames = padding_mask.sum(dim=1).int().tolist()
        lprobs = F.log_softmax(encoder_out.float(), dim=-1)
        hyp_strs, tgt_strs, ed_info_list, align_info_list = self.cal_edit_distance(
            batch_data['uttid'],
            token_ids,
            targets,
            target_lengths,
            language,
            filter_list,
            concate_en_letters,
        )
        out_rlt_dict = {}
        out_rlt_dict["logits"] = lprobs
        out_rlt_dict["frames"] = encoder_frames
        out_rlt_dict["res_lengths"] = res_lengths
        out_rlt_dict["tokens"] = token_ids
        if mode == 'test':
            return (
                hyp_strs,
                tgt_strs,
                ed_info_list,
                align_info_list,
                out_rlt_dict,
            )
        (
            total_error_dist,
            ins_error_dist,
            del_error_dist,
            sub_error_dist,
            total_dist,
        ) = self.rlt_post_process(ed_info_list)
        return total_error_dist, ins_error_dist, del_error_dist, sub_error_dist, total_dist

    def register_infers(self):
        '''register infer object for export.'''
        if self.args.front_end_type in ('Conv2dPooling') and self.args.acoustic_backbone_type in (
            'ConformerBackbone',
            'MaskedConformerBackbone',
        ):
            self.acoustic_front_end_module = self.front_end
            self.acoustic_backbone_module = self.encoder_backbone
            self.acoustic_head_module = None
            self.backbone_pool_module = None
            # self.encoder_backbone = model.encoder_backbone
            infers = [
                CifEncoderExporter(self, self.cif_weight_estimator, self.encoder_proj, **self.args),
                CifCeEncoderExporter(self, **self.args),
                CifDecoderLogitsExporter(self.decoder, **self.args),
            ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
