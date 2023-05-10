''' Context-aware RNN-T model '''
# pylint: disable=unused-wildcard-import
import os
import torch
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.asr.context_encoder import *
from core.models.layers.time_reduce_layer import *
from core.solutions.base_solution import register_solution
from core.solutions.inference.utils import rnnt_rlt_neaten
from core.utils import logging, dist_hdfs_get, get_local_rank
from .base_rnnt_model import BaseRnntModel


@register_solution("ContextAwareRnntModel")
class ContextAwareRnntModel(BaseRnntModel):
    '''Context Aware RNN-T Model'''

    def __init__(self, args):
        '''init function for RNN-T skeleton model.'''
        super().__init__(args)
        self.args = args
        self.update_steps = 0
        self.is_inference = args.is_inference
        self.context_off = args.get('context_off', False)
        self.context_aware_method = args.get('context_aware_method')

        self.build_context_encoder(args)

        if self.context_aware_method == 'CAE':  # context-aware encoder
            self.predictor = super().predictor
            self.greedy_inference = super().greedy_inference
            self.beam_inference = super().beam_inference
        elif self.context_aware_method == 'CAP':  # context-aware predictor
            self.encoder = super().encoder
        else:
            raise ValueError("supported context_aware_method is in {CAE, CAP, CAE&CAP, CLM}")
        # TODO(chenjinkun): CAE&CAP, CLM

    def build_context_encoder(self, args):
        '''build the context-encoder, and resume it if needed'''
        context_encoder_type = args.get('context_encoder_type', 'LSTM')  # LSTM, BERT
        context_encoder_ckpt = args.get('context_encoder_ckpt')
        context_encoder_config = args.get('context_encoder_config')

        if context_encoder_config:
            context_encoder_cfg_local = dist_hdfs_get(context_encoder_config, './tmp')
            args['context_encoder_cfg_local'] = context_encoder_cfg_local
        self.context_encoder = eval(args.context_encoder_module)(args)

        if self.is_inference or not context_encoder_ckpt:
            return
        local_ckpt = dist_hdfs_get(context_encoder_ckpt, './tmp', 'pretrain_context_encoder.pth')

        if context_encoder_type == 'BERT':
            self.context_encoder.text_encoder = BertModel_huggingface.from_pretrained(
                local_ckpt, config=context_encoder_cfg_local
            )

        else:  # resume lstm text-encoder
            context_enc_pname_prefix = args.get('context_encoder_pname_prefix', '')
            state_dict = {}
            if os.path.exists(local_ckpt):
                pretrained_state_dict = torch.load(local_ckpt, map_location='cpu')
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']
                for name, _ in self.context_encoder.text_encoder.named_parameters():
                    old_name = context_enc_pname_prefix + name
                    if old_name in pretrained_state_dict:
                        state_dict[name] = pretrained_state_dict[old_name]
            missing_keys, _ = self.context_encoder.text_encoder.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                logging.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
        logging.info(
            'rank %d, resume pretrained context encoder from %s',
            get_local_rank(),
            context_encoder_ckpt,
        )

    def attend_acoustics_and_context(self, acoustic_feat, context, context_mask):
        '''apply the attention mechanism over the acoustics and context'''
        if not self.context_off and self.context_aware_method == 'CAE':
            context_out, _ = self.context_encoder(
                acoustic_feat.permute(0, 2, 1), context, context_mask
            )
            context_out = context_out.permute(0, 2, 1)
            acoustic_feat = acoustic_feat + context_out
        return acoustic_feat

    def encoder(self, batch_data):
        '''
        Encoder Module in RNN-T
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        context = batch_data['context_text']
        context_mask = batch_data['context_text_mask']
        ############ Audio Encoder ############
        if self.is_inference or self.update_steps <= self.encoder_fix_steps:
            with torch.no_grad():
                # front-end
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                acoustic_in = self.attend_acoustics_and_context(
                    front_end_out, context, context_mask
                )
                # output (B, T, N)
                encoder_backbone_out = self.encoder_backbone(
                    acoustic_in, backbone_mask, frontend_shape
                )
        else:
            with torch.enable_grad():
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                acoustic_in = self.attend_acoustics_and_context(
                    front_end_out, context, context_mask
                )
                encoder_backbone_out = self.encoder_backbone(
                    acoustic_in, backbone_mask, frontend_shape
                )

        # ctc mtl branch
        if self.mtl_module is not None:
            mtl_logits = self.mtl_module(encoder_backbone_out)
        else:
            mtl_logits = None
        # head
        encoder_out = self.acoustic_head_module(encoder_backbone_out)
        # backbone_pool_module
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
            batch_data["mtl_mask"] = backbone_mask.clone()
            backbone_mask = backbone_mask[:, :-1][:, ::2] if backbone_mask is not None else None
        trainable = encoder_out.requires_grad
        return encoder_out, backbone_mask, trainable, mtl_logits, encoder_backbone_out

    def predictor(self, batch_data, trainable=True, key='prev_char'):
        '''
        Predictor Module in RNN-T,
        typically for context-aware predictor
        '''
        prev_char = batch_data[key]
        context_embd = batch_data['context_embd']
        context_mask = batch_data['context_text_mask']
        if trainable and self.args.predictor_dropout_factor > 0.0:
            with torch.no_grad():
                random_uniform_tensor = (
                    torch.empty(prev_char.size())
                    .uniform_(0, self.args.predictor_dropout_factor + 1)
                    .cuda()
                )
                mask = (random_uniform_tensor < 1).long()
                unk_idx = self.args.tgt_dict.index('<unk>')
                prev_char = prev_char * mask + (1 - mask) * unk_idx

        predictor_out, _ = self.predictor_module(
            prev_char,
            context_embd=context_embd,
            context_mask=context_mask,
            context_encoder=self.context_encoder,
        )
        return predictor_out

    def forward(self, batch_data, inference=False):
        '''
        Forward for RNN-T base module,
        typically for context-aware predictor
        '''
        if self.training:
            self.update_steps += 1

        ############ Context Encoder #############
        context = batch_data['context_text']
        context_mask = batch_data['context_text_mask']
        context_embd = self.context_encoder.get_context_embd(context, context_mask)
        batch_data['context_embd'] = context_embd

        ############ Audio Encoder #############
        if 'encoder_out' in batch_data:
            encoder_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            trainable = batch_data['trainable']
            mtl_logits = batch_data['mtl_logits']
            encoder_backbone_out = batch_data['encoder_backbone_out']
        else:
            (
                encoder_out,
                backbone_mask,
                trainable,
                mtl_logits,
                encoder_backbone_out,
            ) = self.encoder(batch_data)

        ############ Context Encoder #############
        context = batch_data['context_text']
        context_mask = batch_data['context_text_mask']
        context_embd = self.context_encoder.get_context_embd(context, context_mask)
        batch_data['context_embd'] = context_embd

        if inference:
            return encoder_out, backbone_mask, mtl_logits, encoder_backbone_out

        ############ Predictor ############
        predictor_out = self.predictor(batch_data, trainable)
        target_lengths = batch_data['target_lengths']

        ############ Jointer ############
        jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)
        batch_data['backbone_mask'] = backbone_mask

        # criterion for loss computation
        forward_out = self.criterion_module(
            jointer_out,
            batch_data,
            mtl_logits=mtl_logits,
            mtl_type=self.mtl_type,
        )
        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad:
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out,
                target,
                batch_data,
                context_embd=context_embd,
                context_mask=context_mask,
                context_encoder=self.context_encoder,
            )
        return forward_out

    @torch.no_grad()
    def greedy_inference(self, batch_data):
        '''
        Greedy inference,
        typically for context-aware predictor
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        ############ Acoustic ############
        acoustic_out, _, _, _ = self.forward(batch_data, inference=True)

        greedy_infer_rlt = self.criterion_module.greedy_infer(
            acoustic_out,
            self.predictor_module,
            self.jointer_module,
            context_embd=batch_data['context_embd'],
            context_mask=batch_data['context_text_mask'],
            context_encoder=self.context_encoder,
        )
        out_rlt_list = []
        for bid in range(bsz):
            hyp_token_list = greedy_infer_rlt[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
            out_rlt_list.append(hyp_token_list)
        return out_rlt_list

    @torch.no_grad()
    def beam_inference(
        self,
        batch_data,
        nbest=1,
        nbest_align_info=False,
        enable_las_rescore=False,
        output_timestamp=False,
        prefetch=False,
        fixed_prefix=False,
        output_rnnt_confidence=False,
        **_kwargs,
    ):
        '''
        Beam inference,
        typically for context-aware predictor
        '''
        # pylint:disable=too-many-branches,too-many-locals
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        ############ Acoustic ############
        if 'encoder_out' in batch_data:
            # no more repetitive computation for efficiency
            acoustic_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            mtl_logits = batch_data['mtl_logits']
            encoder_backbone_out = batch_data['encoder_backbone_out']
        else:
            acoustic_out, backbone_mask, mtl_logits, encoder_backbone_out = self.forward(
                batch_data, inference=True
            )

        beam_infer_rlt, beam_infer_rlt_nbest = self.beam_searcher(
            acoustic_out,
            backbone_mask,
            context_embd=batch_data['context_embd'],
            context_mask=batch_data['context_text_mask'],
            context_encoder=self.context_encoder,
        )

        # return nbest align info
        if nbest_align_info:
            return beam_infer_rlt_nbest["nbest"], None, None
        out_rlt_list = {}
        # return timestamp or confidence
        if output_timestamp or output_rnnt_confidence:
            out_rlt_list["timestamp"] = []
            for bid_rlt in beam_infer_rlt_nbest["nbest"]:
                out_rlt_list["timestamp"].append(bid_rlt[0][3][1:])

        out_rlt_list_nbest = []
        out_rlt_confidence_list = []
        out_rlt_list["inf_res"] = []
        for bid in range(bsz):
            hyp_token_list = beam_infer_rlt[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
            out_rlt_list["inf_res"].append(hyp_token_list)
            bid_rlt = beam_infer_rlt_nbest['nbest'][bid]
            if output_rnnt_confidence:
                out_rlt_confidence_list.append(bid_rlt[0][3][1:])
            if nbest > 1 and beam_infer_rlt_nbest is not None:
                cur_nbest = []
                if isinstance(bid_rlt, torch.Tensor):
                    num_beam = bid_rlt.size(0)
                else:
                    num_beam = len(bid_rlt)
                for beam_idx in range(num_beam):
                    beam_rlt = bid_rlt[beam_idx]
                    if isinstance(beam_rlt, tuple):
                        rlt_score = beam_rlt[1]
                        rlt_confidence = '|'.join([str(p) for p in beam_rlt[3][1:]])
                        beam_rlt = beam_rlt[0]  # ([xxx, xxx], score)
                    else:
                        rlt_score = 0.0
                        rlt_confidence = '0.0'
                    cur_nbest.append(
                        (
                            rnnt_rlt_neaten(beam_rlt),
                            '{}|{}'.format(str(rlt_score), rlt_confidence),
                        )
                    )
                out_rlt_list_nbest.append(cur_nbest)
        out_rlt_list["nbest"] = out_rlt_list_nbest
        if enable_las_rescore:
            assert nbest > 1
            return out_rlt_list, encoder_backbone_out, backbone_mask
        encoder_frames = backbone_mask.sum(dim=1).int().tolist()
        if fixed_prefix:
            out_rlt_list["fixed_prefix"] = beam_infer_rlt_nbest["fixed_prefix"]
        if prefetch:
            out_rlt_list["prefetch"] = beam_infer_rlt_nbest["prefetch"]
        if output_rnnt_confidence:
            out_rlt_list["confidence"] = out_rlt_confidence_list
        return out_rlt_list, mtl_logits, encoder_frames
