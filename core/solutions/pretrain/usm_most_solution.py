'''USM MOST solution'''
# pylint:disable=too-many-branches,too-many-locals
import copy
import torch
import torch.nn.functional as F
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.pretrained.data2vec_model import SharedEncoder
from core.models.pretrained.text_encoder import TextEncoder
from core.models.pretrained.utils import *
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.inference.rnnt_greedy_search import GreedySearch
from core.solutions.inference.utils import rnnt_rlt_neaten
from core.solutions.inference.rnnt_beam_search import (
    BatchBeamSearch,
    NonBatchBeamSearch,
)
from core.models.layers.ema_module import EMAModule


@register_solution("USMMostModel")
class USMMostModel(BaseSolution):
    '''USMMostModel'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        # shared encoder, (feature encoder = speech encoder)
        self.shared_encoder = SharedEncoder(args)
        # text encoder
        self.text_token_type = args.text_encoder_args.get('text_token_type', None)
        vocab_size = args.text_encoder_args.get('tgt_vocab_size', args.tgt_vocab_size)
        args.text_encoder_args.tgt_vocab_size = vocab_size
        self.text_encoder = TextEncoder(args.text_encoder_args)
        # reuse RNN-T jointer & predictor
        self.acoustic_head_module = eval(args.head_type)(args)
        self.predictor_module = eval(args.predictor_type)(args)
        self.jointer_module = eval(args.jointer_type)(args)
        self.criterion_module = eval(args.criterion_type)(args)
        # multitask head
        self.mtl_type = args.get('mtl_type', None)
        self.mtl_module = eval(args.mtl_head)(args) if self.mtl_type else None

        self.beam_searcher = None
        self.greedy_searcher = GreedySearch(
            self.args, self.criterion_module, self.predictor_module, self.jointer_module
        )
        self.update_steps = 0
        self.loss_norm_type = args.get('loss_norm_type', 'batch')
        self.loss_scale_paired = args.get('loss_scale_paired', 1)
        # unsupervised speech pretraining
        self.loss_scale_speech = args.get('loss_scale_speech', 1)
        # supervised alignment & duration
        self.loss_scale_match = args.get('loss_scale_match', 1)
        # text reconstruction
        self.loss_scale_text = args.get('loss_scale_text', 1)
        # normalizations
        args.ema_duration = args.get('ema_duration', True)  # default
        args.text_loss_skip_encoder = args.get('text_loss_skip_encoder', False)
        args.instance_norm_target = args.get('instance_norm_target', False)
        args.layer_norm_target = args.get('layer_norm_target', False)

        # build ema
        self.cuda()
        self.ema = None
        self.enable_ema = args.get('enable_ema', False)
        args.ema_anneal_start_step = args.get('ema_anneal_start_step', 0)
        self.make_ema_model()

        self._register_load_state_dict_pre_hook(self._model_load_hook)

    @property
    def is_downsample_head(self):
        '''whether downsample head'''
        return self.args.head_type in ('RnntDownsampleHead',)

    def acoustic_head(self, backbone_out, backbone_mask):
        '''Acoustic head'''
        if self.is_downsample_head:
            out, backbone_mask = self.acoustic_head_module(backbone_out, backbone_mask)
        else:
            out = self.acoustic_head_module(backbone_out)
        return out, backbone_mask

    def predictor(self, batch_data, trainable=True, key='prev_char'):
        '''
        Predictor Module in RNN-T
        '''
        prev_char = batch_data[key]
        unk_idx = self.args.tgt_dict.index('<unk>')
        prev_char_drop = self.args.predictor_dropout_factor if trainable else 0
        predictor_out, _ = self.predictor_module(
            prev_char, prev_char_drop=prev_char_drop, unk_idx=unk_idx
        )
        return predictor_out

    def jointer(self, encoder, predictor, input_lengths, target_lengths):
        '''
        Jointer Module in RNN-T.
        '''
        jointer_out = self.jointer_module(encoder, predictor, input_lengths, target_lengths)
        return jointer_out

    def forward_speech(self, batch_data):
        '''
        unsupervised speech training:
            speech encoder -> shared encoder
            Time mask only.
        '''
        self.shared_encoder.set_ema(self.ema.model.shared_encoder)
        forward_out = self.shared_encoder.forward_pretrained(batch_data)
        self.shared_encoder.del_ema()
        return forward_out

    def forward_text(self, batch_data):
        '''
        text reconstruction training:
            text encoder -> shared encoder -> RNN-T decoder
            Time & Freq mask.
        '''
        if self.text_token_type:
            mel_out, _, _, mel_masks = self.text_encoder(
                batch_data['phone'].clone(), ~batch_data['phone_mask'].bool(), mask=self.training
            )
        else:
            mel_out, _, _, mel_masks = self.text_encoder(
                batch_data['char'].clone(), ~batch_data['char_mask'].bool(), mask=self.training
            )
        # pass to shared_encoder
        encoder_backbone_out, encoder_mask = self.shared_encoder.forward_text(
            mel_out, mel_masks, mask=self.training
        )
        backbone_mask = ~encoder_mask
        encoder_out, backbone_mask = self.acoustic_head(encoder_backbone_out, backbone_mask)
        # NOTE inplace change may cause side-effects
        batch_data['src_mask'] = backbone_mask
        batch_data['backbone_mask'] = backbone_mask
        # mtl head
        mtl_logits = self.mtl_module(encoder_backbone_out) if self.mtl_type else None
        # RNN-T decoder
        predictor_out = self.predictor(batch_data)
        target_lengths = batch_data['target_lengths']
        jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)
        # RNN-T loss
        forward_out = self.criterion_module(
            jointer_out, batch_data, mtl_logits=mtl_logits, mtl_type=self.mtl_type
        )
        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad:
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out, target, batch_data
            )

        return forward_out

    def forward_text_ema(self, batch_data, text_out=None):
        '''
        text reconstruction training:
            text encoder -> shared encoder -> RNN-T decoder
            Time & Freq mask.
        '''
        if text_out is None:
            texts, masks = batch_data['char'], ~batch_data['char_mask'].bool()
            if self.args.text_loss_skip_encoder:
                with torch.no_grad():
                    mel_out, _, _, mel_masks = self.text_encoder(texts, masks)
            else:
                with torch.no_grad():
                    if self.args.ema_duration:
                        self.ema.model.text_encoder.eval()
                        duration = self.ema.model.text_encoder.predict_duration(texts, masks)
                    else:
                        self.text_encoder.eval()
                        duration = self.text_encoder.predict_duration(texts, masks)
                        self.text_encoder.train()
                mel_out, _, _, mel_masks = self.text_encoder(
                    texts.clone(), masks, d_targets=duration, mask=self.training
                )
        else:
            mel_out, mel_masks = text_out
        # pass to shared_encoder
        encoder_backbone_out, encoder_mask = self.shared_encoder.forward_text(
            mel_out, mel_masks, mask=self.training
        )
        backbone_mask = ~encoder_mask
        encoder_out, backbone_mask = self.acoustic_head(encoder_backbone_out, backbone_mask)
        # NOTE inplace change may cause side-effects
        batch_data['src_mask'] = backbone_mask
        batch_data['backbone_mask'] = backbone_mask
        # mtl head
        mtl_logits = self.mtl_module(encoder_backbone_out) if self.mtl_type else None
        # RNN-T decoder
        predictor_out = self.predictor(batch_data)
        target_lengths = batch_data['target_lengths']
        jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)
        # RNN-T loss
        forward_out = self.criterion_module(
            jointer_out, batch_data, mtl_logits=mtl_logits, mtl_type=self.mtl_type
        )
        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad:
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out, target, batch_data
            )

        return forward_out

    # pylint:disable=too-many-locals
    def forward_speech_text(self, batch_data):
        '''
        modality matching & duration prediction training
            eval, no_grad, no mask
            only train text encoder
        '''
        # RNN-T alignment, eval, no_grad
        self.eval()
        with torch.no_grad():
            # speech encoder
            encoder_backbone_out, padding_mask = self.shared_encoder(batch_data)
            backbone_mask = ~padding_mask
            encoder_out, backbone_mask = self.acoustic_head(encoder_backbone_out, backbone_mask)
            batch_data['backbone_mask'] = backbone_mask

            # RNN-T decoder
            predictor_out = self.predictor(batch_data, trainable=False)
            target_lengths = batch_data['target_lengths']
            jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)

            # RNN-T loss
            input_lengths = (batch_data['backbone_mask'].sum(dim=1)).int()
            target = batch_data['char']
            target_mask = batch_data['char_mask']
            target_lengths = (target_mask.sum(dim=1)).int()
            log_probs = self.criterion_module.log_softmax_fc(
                jointer_out,
                target,
                input_lengths,
                target_lengths,
                adaptive_tgt_indices=batch_data['adaptive_tgt_indices'],
            )
            align = self.criterion_module.rnnt_force_alignment(
                log_probs,
                input_lengths,
                target_lengths + 1,
            )

            align_nxt = torch.cat([align[:, 1:], align.new_zeros(align.size(0), 1)], -1)
            lengths = batch_data['target_lengths']
            for b, lens in enumerate(lengths):
                align_nxt[b, lens] = input_lengths[b]
            duration = align_nxt - align
            duration = duration[:, 1:]  # remove <blk>
            if self.is_downsample_head:
                duration *= 2

        # modality matching: get speech representation from speech encoder, eval, no_grad
        with torch.no_grad():
            speech_feature, speech_mask = self.shared_encoder.extract_features(batch_data)

        # enable grad for text encoder
        self.train()

        # duration prediction: get duration from alignment
        texts, masks = batch_data['char'], ~batch_data['char_mask'].bool()
        mel_out, log_d_predictions, _, mel_mask = self.text_encoder(
            texts.clone(), masks, speech_mask, duration, mask=self.training
        )
        src_mask = ~masks
        mel_mask = ~mel_mask  # NOTE should be the same as ~speech_mask

        log_duration_targets = torch.log(duration.float() + 1)
        log_duration_predictions = log_d_predictions.masked_select(src_mask).float()
        log_duration_targets = log_duration_targets.masked_select(src_mask).float()
        duration_loss = F.mse_loss(log_duration_predictions, log_duration_targets)

        mel_out = mel_out.masked_select(mel_mask.unsqueeze(-1)).float()
        speech_feature = speech_feature.masked_select(mel_mask.unsqueeze(-1)).float()
        modality_matching_loss = F.mse_loss(mel_out, speech_feature)

        forward_out = {}
        forward_out['dur_loss'] = duration_loss
        forward_out['dur_denorm'] = src_mask.sum()
        forward_out['dur_prd'] = torch.clamp(torch.round(torch.exp(log_d_predictions) - 1), min=0)
        forward_out['dur_tgt'] = duration
        forward_out['mm_loss'] = modality_matching_loss
        forward_out['mm_denorm'] = mel_mask.sum()
        forward_out['backward_loss'] = duration_loss + modality_matching_loss

        return forward_out

    # pylint:disable=too-many-locals
    def forward_speech_text_ema(self, batch_data):
        '''
        modality matching & duration prediction training
            eval, no_grad, no mask
            only train text encoder
        '''
        # RNN-T alignment, eval, no_grad
        self.ema.model.eval()
        with torch.no_grad():
            # speech encoder
            encoder_backbone_out, padding_mask = self.ema.model.shared_encoder(batch_data)
            backbone_mask = ~padding_mask
            encoder_out, backbone_mask = self.ema.model.acoustic_head(
                encoder_backbone_out, backbone_mask
            )
            batch_data['backbone_mask'] = backbone_mask

            # RNN-T decoder
            predictor_out = self.ema.model.predictor(batch_data, trainable=False)
            target_lengths = batch_data['target_lengths']
            jointer_out = self.ema.model.jointer(encoder_out, predictor_out, None, target_lengths)

            # RNN-T loss
            input_lengths = (batch_data['backbone_mask'].sum(dim=1)).int()
            target = batch_data['char']
            target_mask = batch_data['char_mask']
            target_lengths = (target_mask.sum(dim=1)).int()
            log_probs = self.ema.model.criterion_module.log_softmax_fc(
                jointer_out,
                target,
                input_lengths,
                target_lengths,
                adaptive_tgt_indices=batch_data['adaptive_tgt_indices'],
            )
            align = self.ema.model.criterion_module.rnnt_force_alignment(
                log_probs,
                input_lengths,
                target_lengths + 1,
            )

            align_nxt = torch.cat([align[:, 1:], align.new_zeros(align.size(0), 1)], -1)
            lengths = batch_data['target_lengths']
            for b, lens in enumerate(lengths):
                align_nxt[b, lens] = input_lengths[b]
            duration = align_nxt - align
            duration = duration[:, 1:]  # remove <blk>
            if self.is_downsample_head:
                duration *= 2

        # modality matching: get speech representation from speech encoder, eval, no_grad
        with torch.no_grad():
            speech_feature, speech_mask = self.ema.model.shared_encoder.extract_features(batch_data)
            if self.args.instance_norm_target:
                speech_feature = F.instance_norm(speech_feature.transpose(1, 2).float()).transpose(
                    1, 2
                )
            if self.args.layer_norm_target:
                speech_feature = F.layer_norm(speech_feature.float(), speech_feature.shape[-1:])

        # duration prediction: get duration from alignment
        texts, masks = batch_data['char'], ~batch_data['char_mask'].bool()
        mel_out, log_d_predictions, _, mel_mask = self.text_encoder(
            texts.clone(), masks, speech_mask, duration, mask=self.training
        )
        text_out = (mel_out, mel_mask)
        src_mask = ~masks
        mel_mask = ~mel_mask  # NOTE should be the same as ~speech_mask

        log_duration_targets = torch.log(duration.float() + 1)
        log_duration_predictions = log_d_predictions.masked_select(src_mask).float()
        log_duration_targets = log_duration_targets.masked_select(src_mask).float()
        duration_loss = F.mse_loss(log_duration_predictions, log_duration_targets)

        mel_out = mel_out.masked_select(mel_mask.unsqueeze(-1)).float()
        speech_feature = speech_feature.masked_select(mel_mask.unsqueeze(-1)).float()
        modality_matching_loss = F.mse_loss(mel_out, speech_feature)

        forward_out = {}
        forward_out['dur_loss'] = duration_loss
        forward_out['dur_denorm'] = src_mask.sum()
        forward_out['dur_prd'] = torch.clamp(torch.round(torch.exp(log_d_predictions) - 1), min=0)
        forward_out['dur_tgt'] = duration
        forward_out['mm_loss'] = modality_matching_loss
        forward_out['mm_denorm'] = mel_mask.sum()
        forward_out['backward_loss'] = duration_loss + modality_matching_loss
        forward_out['text_out'] = text_out

        return forward_out

    def forward_paired(self, batch_data, inference=False):
        '''
        RNN-T training:
            shared encoder -> RNN-T decoder
            Time & Freq mask.
        '''
        # speech encoder
        encoder_backbone_out, padding_mask = self.shared_encoder(batch_data)
        backbone_mask = ~padding_mask
        encoder_out, backbone_mask = self.acoustic_head(encoder_backbone_out, backbone_mask)
        trainable = encoder_out.requires_grad
        batch_data['backbone_mask'] = backbone_mask
        # mtl head
        mtl_logits = self.mtl_module(encoder_backbone_out) if self.mtl_type else None
        if inference:
            return encoder_out, backbone_mask, mtl_logits, encoder_backbone_out

        # RNN-T decoder
        predictor_out = self.predictor(batch_data, trainable)
        target_lengths = batch_data['target_lengths']
        jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)

        # RNN-T loss
        forward_out = self.criterion_module(
            jointer_out, batch_data, mtl_logits=mtl_logits, mtl_type=self.mtl_type
        )
        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad:
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out, target, batch_data
            )

        return forward_out

    def forward(self, batch_data, batch_list=None, inference=False):
        '''
        forward multiple times
        '''
        if batch_list is None:
            return self.forward_paired(batch_data, inference)

        loss = 0.0
        solution_out = {
            'text_loss': 0,
            'text_denorm': 0,
            'text_cer': 0,
            'text_dist': 0,
            'speech_loss': 0,
            'speech_denorm': 0,
        }
        for key in batch_list:
            batch = batch_data[key]
            if key == 'paired':
                solution_out.update(self.forward_paired(batch))
                # loss / B, nll_loss / N
                solution_out['paired_loss'] = solution_out['backward_loss']
                solution_out['paired_denorm'] = solution_out['utt_num']
                loss += solution_out['backward_loss'] * self.loss_scale_paired
            if key == 'speech-text':
                if batch_data.get('use_paired_speech', False):
                    solution_out_speech = self.forward_speech(copy.deepcopy(batch))
                    # loss / (N * sqrt(D))
                    solution_out['speech_loss'] += solution_out_speech['loss']  # speech only loss
                    solution_out['speech_denorm'] += solution_out_speech['tgt_size']
                    loss += solution_out_speech['loss'] * self.loss_scale_speech
                if self.enable_ema:
                    solution_out_mm = self.forward_speech_text_ema(batch)
                else:
                    solution_out_mm = self.forward_speech_text(batch)
                # dur_loss / (N * D)
                solution_out['dur_loss'] = solution_out_mm['dur_loss']
                solution_out['dur_denorm'] = solution_out_mm['dur_denorm']
                solution_out['dur_prd'] = solution_out_mm['dur_prd']
                solution_out['dur_tgt'] = solution_out_mm['dur_tgt']
                solution_out['mm_loss'] = solution_out_mm['mm_loss']
                solution_out['mm_denorm'] = solution_out_mm['mm_denorm']
                loss += solution_out_mm['backward_loss'] * self.loss_scale_match
                if batch_data.get('use_paired_text', False):
                    if self.enable_ema:
                        text_out = solution_out_mm['text_out']
                        solution_out_text = self.forward_text_ema(batch, text_out)
                    else:
                        solution_out_text = self.forward_text(batch)
                    # loss / B
                    solution_out['text_loss'] += solution_out_text['backward_loss']
                    solution_out['text_denorm'] += solution_out_text['utt_num']
                    solution_out['text_cer'] += solution_out_text.get('cer', 0)
                    solution_out['text_dist'] += solution_out_text.get('dist', 0)
                    loss += solution_out_text['backward_loss'] * self.loss_scale_text
            if key == 'speech':
                solution_out_speech = self.forward_speech(batch)
                # loss / (N * sqrt(D))
                solution_out['speech_loss'] += solution_out_speech['loss']  # speech only loss
                solution_out['speech_denorm'] += solution_out_speech['tgt_size']
                loss += solution_out_speech['loss'] * self.loss_scale_speech
            if key == 'text':
                if self.enable_ema:
                    solution_out_text = self.forward_text_ema(batch)
                else:
                    solution_out_text = self.forward_text(batch)
                # loss / B
                solution_out['text_loss'] += solution_out_text['backward_loss']
                solution_out['text_denorm'] += solution_out_text['utt_num']
                solution_out['text_cer'] += solution_out_text.get('cer', 0)
                solution_out['text_dist'] += solution_out_text.get('dist', 0)
                loss += solution_out_text['backward_loss'] * self.loss_scale_text
        solution_out['loss'] = loss
        solution_out['loss_denorm'] = 1.0
        solution_out['backward_loss'] = loss
        return solution_out

    def make_ema_model(self):
        '''make eam model'''
        args = self.args
        skip_keys, skip_list = set(), []
        if args.get('ema_skip_prefix', None):
            skip_list = args.ema_skip_prefix.split(',')
        # e.g. skip shared_encoder.d2v_model.{feature_extractor, encoder.pos_conv}
        for k, _ in self.named_parameters():
            for prefix in skip_list:
                if k.startswith(prefix):
                    skip_keys.add(k)
        self.ema = EMAModule(
            self,
            ema_decay=self.args.ema_decay,
            ema_fp32=True,
            skip_keys=skip_keys,
        )
        self.ema.model.eval()

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.update_steps = num_updates
        if hasattr(self.shared_encoder, 'set_num_updates'):
            self.shared_encoder.set_num_updates(num_updates)
        if hasattr(self.text_encoder, 'set_num_updates'):
            self.text_encoder.set_num_updates(num_updates)
        if self.training and self.ema is not None:
            if num_updates < self.args.ema_anneal_start_step:
                decay = 0
            elif num_updates < self.args.ema_anneal_end_step:
                cur_steps = num_updates - self.args.ema_anneal_start_step
                total_steps = self.args.ema_anneal_end_step - self.args.ema_anneal_start_step
                decay = get_annealed_rate(
                    self.args.ema_decay,
                    self.args.ema_end_decay,
                    cur_steps,
                    total_steps,
                )
            else:
                decay = self.args.ema_end_decay
            self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        # pylint: disable=unsupported-assignment-operation
        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            if k in state_dict:
                self.ema.restore(state_dict[k], True)
                del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _model_load_hook(
        self,
        state_dict,
        _prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        """init from pretrained model"""
        old_state_dict = state_dict.copy()
        state_dict.clear()
        skip_keys = self.args.get('skip_params', '?').split(',')
        keep_keys = self.args.get('keep_params', '').split(',')
        for name, param in old_state_dict.items():
            new_name = name
            keep = True
            for prefix in skip_keys:
                if name.startswith(prefix):
                    print(f'skip {name}')
                    keep = False
            for prefix in keep_keys:
                if not name.startswith(prefix):
                    print(f'skip {name}')
                    keep = False
            if keep:
                state_dict[new_name] = param

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
        if not inference_cfg.get('use_batch_beam', False):
            self.beam_searcher = NonBatchBeamSearch(
                inference_cfg,
                self.predictor_module,
                self.jointer_module,
                self.criterion_module,
                lm_solution,
            )
        else:
            self.beam_searcher = BatchBeamSearch(
                inference_cfg,
                self.predictor_module,
                self.jointer_module,
                self.criterion_module,
                lm_solution,
            )

    @torch.no_grad()
    def greedy_inference(self, batch_data):
        '''
        Greedy inference
        '''
        bsz = batch_data['src_mask'].shape[0]
        ############ Acoustic ############
        acoustic_out, _, _, _ = self.forward(batch_data, inference=True)

        greedy_infer_rlt = self.greedy_searcher.greedy_infer(
            acoustic_out, self.predictor_module, self.jointer_module
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

        beam_infer_rlt, beam_infer_rlt_nbest = self.beam_searcher(
            acoustic_out, backbone_mask, stable_metric_list=stable_metric_list
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

    def process_beam_infer_rlt(
        self,
        batch_data,
        beam_infer_rlt,
        beam_infer_rlt_nbest,
        backbone_mask,
        nbest=1,
        nbest_align_info=False,
        output_timestamp=False,
        prefetch=False,
        fixed_prefix=False,
        output_rnnt_confidence=False,
        endpoint=False,
    ):
        '''
        Beam results process
        '''
        # pylint:disable=too-many-branches,too-many-locals
        bsz = batch_data['src_mask'].shape[0]
        # return nbest align info
        if nbest_align_info:
            return beam_infer_rlt_nbest["nbest"], None
        out_rlt_list = {}
        # return timestamp
        if output_timestamp:
            out_rlt_list["timestamp"] = []
            for bid_rlt in beam_infer_rlt_nbest["nbest"]:
                out_rlt_list["timestamp"].append(bid_rlt[0][3][1:])

        out_rlt_list_nbest = []
        out_rlt_confidence_list = []
        out_rlt_list["inf_res"] = []
        # pylint: disable=too-many-nested-blocks
        for bid in range(bsz):
            hyp_token_list = beam_infer_rlt[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
            out_rlt_list["inf_res"].append(hyp_token_list)
            prefetch_frames = -1
            bid_rlt = beam_infer_rlt_nbest['nbest'][bid]
            if prefetch:
                final_best = beam_infer_rlt_nbest['prefetch'][bid][-1][1]
                if len(beam_infer_rlt_nbest['prefetch'][bid]) > 1:
                    for prefetch_rlt in beam_infer_rlt_nbest['prefetch'][bid][-2::-1]:
                        if prefetch_rlt[1] == final_best:
                            bid_rlt = prefetch_rlt[-1]
                            prefetch_frames = prefetch_rlt[0]
                            break
            if output_rnnt_confidence:
                out_rlt_confidence_list.append(bid_rlt[0][4][1:])
            if nbest > 1 and beam_infer_rlt_nbest is not None:
                cur_nbest = []
                if isinstance(bid_rlt, torch.Tensor):
                    num_beam = bid_rlt.size(0)
                else:
                    num_beam = len(bid_rlt)
                for beam_idx in range(num_beam):
                    beam_rlt = bid_rlt[beam_idx]
                    rlt_confidence = '0.0'
                    fst_score = 0.0
                    rlt_score = 0.0
                    if isinstance(beam_rlt, tuple):
                        rlt_score = beam_rlt[1]
                        if len(beam_rlt) >= 5:
                            fst_score = beam_rlt[2]
                            confidence = beam_rlt[4]
                            if len(confidence) > 1:
                                rlt_confidence = '|'.join([str(p) for p in confidence[1:]])
                        beam_rlt = beam_rlt[0]
                    cur_nbest.append(
                        (
                            rnnt_rlt_neaten(beam_rlt),
                            fst_score,
                            '{}|{}'.format(str(rlt_score), rlt_confidence),
                            prefetch_frames,
                        )
                    )

                out_rlt_list_nbest.append(cur_nbest)
        encoder_frames = backbone_mask.sum(dim=1).int().tolist()
        if nbest > 1 and beam_infer_rlt_nbest is not None:
            out_rlt_list["nbest"] = out_rlt_list_nbest
        if fixed_prefix:
            out_rlt_list["fixed_prefix"] = beam_infer_rlt_nbest["fixed_prefix"]
        if prefetch:
            out_rlt_list["prefetch"] = beam_infer_rlt_nbest["prefetch"]
        if endpoint:
            out_rlt_list["endpoint"] = beam_infer_rlt_nbest["endpoint"]
        if output_rnnt_confidence:
            out_rlt_list["confidence"] = out_rlt_confidence_list

        return out_rlt_list, encoder_frames
