''' RnntDeepBiasModel '''
import torch
import torch.nn.functional as F
from core.models.asr.rare_word import *
from core.models.asr.tog_frontend import *
from core.solutions.base_solution import register_solution
from core.solutions.inference.rnnt_beam_search import NonBatchBeamSearch, BatchBeamSearch
from .base_rnnt_model import BaseRnntModel
from .rnnt_tog_model import RnntTogModel


@register_solution("RnntDeepBiasModel")
class RnntDeepBiasModel(BaseRnntModel):
    '''RnntDeepBiasModel'''

    def __init__(self, args):
        super().__init__(args)
        args.rw_embed_args.setdefault("tgt_vocab_size", args.tgt_vocab_size)
        args.rw_embed_args.setdefault("jointer_hidden_size", args.jointer_hidden_size)
        self.rw_embed_moudule = eval(args.rw_embed_type)(args.rw_embed_args)
        args.rw_bias_args.setdefault("embed_dim", args.jointer_hidden_size)
        args.rw_bias_args.setdefault("predictor_emb_size", args.predictor_emb_size)
        self.rw_bias_moudule = eval(args.rw_bias_type)(args.rw_bias_args)
        self.rare_word_bias_dropout = args.get("rare_word_bias_dropout", 0.0)

    def init_beam_search(self, inference_cfg, lm_solution):
        '''beam search init'''
        if inference_cfg.get('output_timestamp', False):
            inference_cfg.nbest = 1
        if inference_cfg.get('output_streaming_stable_metric', False) and inference_cfg.get(
            'use_batch_beam', False
        ):
            raise RuntimeError(
                'output_streaming_stable_metric only works in non batch beam search!'
            )
        rw_bias_moudule = self.rw_bias_moudule
        if not inference_cfg.get('use_batch_beam', False):
            self.beam_searcher = NonBatchBeamSearch(
                inference_cfg,
                self.predictor_module,
                self.jointer_module,
                self.criterion_module,
                lm_solution,
                rw_bias_moudule,
            )
        else:
            self.beam_searcher = BatchBeamSearch(
                inference_cfg,
                self.predictor_module,
                self.jointer_module,
                self.criterion_module,
                lm_solution,
                rw_bias_moudule,
            )

    def predictor(self, batch_data, trainable=True, key='prev_char'):
        '''
        Predictor Module in RNN-T
        '''
        prev_char = batch_data[key]
        if self.limited_context is not None:
            limited_context = self.limited_context
            batch, time = prev_char.size()
            pad = F.pad(batch_data['char'], (limited_context - 1, 0, 0, 0)).type_as(
                batch_data['src']
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
            prev_char_drop = self.args.predictor_dropout_factor if trainable else 0
            prev_embed, predictor_out, _ = self.predictor_module(
                prev_char, prev_char_drop=prev_char_drop, unk_idx=unk_idx, return_embed=True
            )

        return prev_embed, predictor_out

    def forward(self, batch_data, inference=False):
        '''
        Forward for RNN-T base module
        '''
        if self.training:
            self.update_steps += 1
        ############ Encoder #############
        if 'encoder_out' in batch_data:
            encoder_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            trainable = batch_data['trainable']
            mtl_logits = batch_data['mtl_logits']
            encoder_backbone_out = batch_data['encoder_backbone_out']
        else:
            encoder_out, backbone_mask, trainable, mtl_logits, encoder_backbone_out = self.encoder(
                batch_data
            )
            batch_data['backbone_mask'] = backbone_mask
        assert "bias_words" in batch_data
        if inference:
            rw_embed = self.rw_embed_moudule(batch_data["bias_words"])
            batch_data["rw_embed"] = rw_embed
            return encoder_out, backbone_mask, mtl_logits, encoder_backbone_out
        ############ Predictor ############
        predictor_out = self.predictor(batch_data, trainable)
        prev_embed = predictor_out[0]
        predictor_out = predictor_out[1]
        rw_embed = self.rw_embed_moudule(batch_data["bias_words"])
        rw_bias, _ = self.rw_bias_moudule(predictor_out, rw_embed, query2=prev_embed)
        if self.training and self.rare_word_bias_dropout > 0.0:
            drop_prob = torch.rand(prev_embed.size(0), 1, 1, device=prev_embed.device)
            drop_mask = (drop_prob > self.rare_word_bias_dropout).float()
            rw_bias *= drop_mask
        predictor_out += rw_bias

        target_lengths = batch_data['target_lengths']

        ############ Jointer ############
        # special jointer for large batch support.
        if self.args.get('jointer_split_num', 1) > 1:
            return self.splitting_jointer_forward(
                batch_data, encoder_out, predictor_out, mtl_logits
            )

        jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)
        if self.ilmt_weight > 0.0:
            ilmt_encoder_out = torch.zeros_like(encoder_out[:, 0:1, :])
            ilmt_jointer_out = self.jointer(ilmt_encoder_out, predictor_out, None, target_lengths)
            batch_data["ilmt_jointer_out"] = ilmt_jointer_out

        if self.joint_las_weight > 0.0:
            las_logits = self.las_decoder(batch_data, encoder_backbone_out, backbone_mask)
            batch_data["las_logits"] = las_logits

        # criterion for loss computation
        forward_out = self.criterion_module(
            jointer_out, batch_data, mtl_logits=mtl_logits, mtl_type=self.mtl_type
        )

        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad:
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out, target, batch_data
            )
        return forward_out

    @torch.no_grad()
    def beam_inference(
        self,
        batch_data,
        nbest=1,
        nbest_align_info=False,
        output_timestamp=False,
        prefetch=False,
        fixed_prefix=False,
        output_rnnt_confidence=False,
        endpoint=False,
        stable_metric_list=None,
        **_kwargs,
    ):
        '''
        Beam inference
        '''
        ############ Acoustic ############
        if 'encoder_out' in batch_data:
            # no more repetitive computation for efficiency
            acoustic_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            mtl_logits = batch_data['mtl_logits']
        else:
            acoustic_out, backbone_mask, mtl_logits, _ = self.forward(batch_data, inference=True)

        rw_embed = batch_data.get("rw_embed", None)
        beam_infer_rlt, beam_infer_rlt_nbest = self.beam_searcher(
            acoustic_out, backbone_mask, stable_metric_list=stable_metric_list, rw_embed=rw_embed
        )
        out_rlt_list, encoder_frames = self.process_beam_infer_rlt(
            batch_data,
            beam_infer_rlt,
            beam_infer_rlt_nbest,
            backbone_mask,
            nbest,
            nbest_align_info,
            output_timestamp,
            prefetch,
            fixed_prefix,
            output_rnnt_confidence,
            endpoint,
        )
        return out_rlt_list, mtl_logits, encoder_frames


@register_solution('RnntBiasTogModel')
class RnntBiasTogModel(RnntTogModel, RnntDeepBiasModel):
    '''RnntBiasTogModel'''

    def __init__(self, args):
        '''init'''
        # pylint: disable=super-init-not-called
        RnntDeepBiasModel.__init__(self, args)
        self.merge_by_concat = False
        tog_args = args.tog_front_end_args
        assert not tog_args.get("merge_by_concat", False)
        self.token_frontend = eval(tog_args.token_frontend_type)(tog_args)
        if args.get('tog_fix', False):
            self.token_frontend.requires_grad_(False)
