''' BaseLASModel '''
import torch
from torch import nn

from core.solutions.base_solution import BaseSolution, register_solution
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.pretrained.acoustic_tokenizer import *
from core.models.asr.las_decoder import *
from core.criterions.las_criterion import *
from core.solutions.inference.las_beam_search import BaseBeamSearch
from core.utils import logging


@register_solution("BaseLASModel")
class BaseLASModel(BaseSolution):
    '''
    Base model for LAS.
    - Encoder
        - frontend: VGGFrontEnd
        - backbone: TransformerBackbone, DFSMNBackbone
        - head:
    - Attention
    - Decoder
    - criterion
    '''

    def __init__(self, args):
        '''
        init function for LAS skeleton model.
        '''
        super().__init__()
        self.args = args
        if self.args.using_discrete_token:  # for discrete tokenization
            logging.info('Using discrete token for LAS.')
            self.acoustic_tokenizer = eval(args.acoustic_tokenizer_type)(args)
            self.embed_acoustic_tokens = nn.Embedding(
                args.acoustic_vocab_size, args.acoustic_token_embedding_size
            )
            nn.init.normal_(
                self.embed_acoustic_tokens.weight,
                mean=0,
                std=args.acoustic_token_embedding_size**-0.5,
            )
        else:
            if self.args.front_end_type == 'VGGFrontEnd':
                self.args.downsampling_size = 4
            self.acoustic_front_end_module = eval(args.front_end_type)(args)
            self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)
        self.decoder_module = eval(args.las_decoder_type)(args)
        self.criterion_module = eval(args.criterion_type)(args)

        if args.get('mtl_type', None) and self.training:
            self.mtl_type = args.get('mtl_type')
            self.mtl_module = eval(args.mtl_head)(args)
        else:
            self.mtl_type = None
            self.mtl_module = None

        self.frontend_fix = args.get('frontend_fix', False)
        self.backbone_fix = args.get('backbone_fix', False)
        self.mtl_fix = args.get('mtl_fix', False)
        self.train()
        self._register_load_state_dict_pre_hook(self._compatible_load_hook)

    def train(self, mode: bool = True):
        '''
        Param will really be freezed by set param.requires_grad_(False).
        `with no_grad` may case param be updated by optimizer' weight_decay.
        What's more, we need to set module.eval() to fix some module buffer,
        such as BatchNorm.running_mean.
        '''
        super().train(mode)
        if self.frontend_fix:
            self.acoustic_front_end_module.requires_grad_(False)
            self.acoustic_front_end_module.train(False)
        if self.backbone_fix:
            self.acoustic_backbone_module.requires_grad_(False)
            self.acoustic_backbone_module.train(False)
        if self.mtl_fix and self.mtl_module is not None:
            self.mtl_module.requires_grad_(False)
            self.mtl_module.train(False)
        if self.args.using_discrete_token and self.args.acoustic_token_fix:
            self.acoustic_tokenizer.requires_grad_(False)
            self.acoustic_tokenizer.train(False)

    @staticmethod
    def _compatible_load_hook(
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        """Transform samiasr-chkpt to dolphin-chkpt"""
        # pylint:disable=too-many-branches
        old_state_dict = state_dict.copy()
        state_dict.clear()
        for name, param in old_state_dict.items():
            new_name = name
            # frontend
            if name.startswith('encoder.embed.conv.'):
                new_name = name.replace(
                    'encoder.embed.conv.', prefix + 'acoustic_front_end_module.conv.'
                )
            elif name.startswith('encoder.embed.out.0.'):
                new_name = name.replace(
                    'encoder.embed.out.0.', prefix + 'acoustic_front_end_module.out.'
                )
            # backbone
            elif name.startswith('encoder.encoders.'):
                new_name = name.replace(
                    'encoder.encoders.', prefix + 'acoustic_backbone_module.transformers.'
                )
                if '.self_attn.' in new_name:
                    new_name = new_name.replace('linear_q.', 'q_proj.')
                    new_name = new_name.replace('linear_k.', 'k_proj.')
                    new_name = new_name.replace('linear_v.', 'v_proj.')
                    new_name = new_name.replace('linear_out.', 'out_proj.')
                if '.feed_forward.' in new_name:
                    new_name = new_name.replace('.feed_forward.w_1.', '.fc1.')
                    new_name = new_name.replace('.feed_forward.w_2.', '.fc2.')
                new_name = new_name.replace('.norm1.', '.self_attn_layer_norm.')
                new_name = new_name.replace('.norm2.', '.final_layer_norm.')
            elif name.startswith('encoder.after_norm.'):
                new_name = name.replace(
                    'encoder.after_norm.', prefix + 'acoustic_backbone_module.after_norm.'
                )
            # decoder
            elif name.startswith('decoder.embed_tokens.'):
                new_name = name.replace(
                    'decoder.embed_tokens.', prefix + 'decoder_module.embed_tokens.'
                )
            elif name.startswith('decoder.embed_positions.'):
                new_name = name.replace(
                    'decoder.embed_positions.', prefix + 'decoder_module.pos_en.'
                )
            elif name.startswith('decoder.layers.'):
                new_name = name.replace('decoder.layers.', prefix + 'decoder_module.transformers.')
            elif name.startswith('decoder.layer_norm.'):
                new_name = name.replace(
                    'decoder.layer_norm.', prefix + 'decoder_module.final_norm.'
                )
            elif name.startswith('decoder.output_projection.'):
                new_name = name.replace(
                    'decoder.output_projection.', prefix + 'decoder_module.pred_fc.'
                )
            # mtl
            elif name.startswith('ctc.ctc_lo.'):
                new_name = name.replace('ctc.ctc_lo.', prefix + 'mtl_module.head_fc_trans.')
            state_dict[new_name] = param

    def frontend(self, fbank, mask):
        '''
        frontend of LAS encoder
        '''
        if self.acoustic_front_end_module is not None:
            return self.acoustic_front_end_module(fbank, mask)
        # Nothing to do in frontend
        return fbank, mask, None

    def encoder(self, batch_data):
        '''
        Encoder Module in LAS
        '''
        if 'waveform' in batch_data:
            fbank = batch_data['waveform']  # (B, T, ndim)
            fbank_mask = batch_data['wav_mask']
        else:
            fbank = batch_data['src']  # (B, T, ndim)
            fbank_mask = batch_data['src_mask']
        ############ Encoder ############
        if self.args.using_discrete_token:
            acoustic_tokens, backbone_mask = self.acoustic_tokenizer(fbank, fbank_mask)
            encoder_out = self.embed_acoustic_tokens(acoustic_tokens)
        else:
            # front-end
            front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
            # output (B, N, T)
            encoder_out = self.acoustic_backbone_module(
                front_end_out, backbone_mask, frontend_shape=frontend_shape
            )
        trainable = encoder_out.requires_grad
        # ctc mtl branch
        if self.mtl_module is not None:
            mtl_logits = self.mtl_module(encoder_out)
        else:
            mtl_logits = None
        return encoder_out, backbone_mask, trainable, mtl_logits

    # pylint: disable=unused-argument
    def decoder(self, encoder, mask, batch_data, trainable=True):
        '''
        Decoder Module in LAS
        '''
        prev_char = batch_data['prev_char']
        decoder_out = self.decoder_module(encoder, mask, prev_char)
        return decoder_out

    def forward(self, batch_data):
        '''
        Forward for LAS base model
        '''
        encoder_out, backbone_mask, _, mtl_logits = self.encoder(batch_data)
        logits = self.decoder(encoder_out, backbone_mask, batch_data)
        src_mask = batch_data['src_mask']
        target = batch_data['char']
        target_mask = batch_data['char_mask']
        forward_out = self.criterion_module(
            logits, src_mask, target, target_mask, backbone_mask, mtl_logits, self.mtl_type
        )
        return forward_out

    @staticmethod
    def max_decoder_positions():
        '''used for decoding'''
        return 10000

    @staticmethod
    def reorder_encoder_out(encoder_out, new_order):
        '''encoder_out shape (B,T,N) new_order shape (B*beam)'''
        for name, tensor in encoder_out.items():
            encoder_out[name] = tensor.index_select(0, new_order)
        return encoder_out

    def init_beam_search(self, inference_cfg):
        '''beam search init'''
        self.beam_searcher = BaseBeamSearch(
            self.args,
            inference_cfg,
            self.decoder_module,
        )

    @torch.no_grad()
    def beam_inference(
        self,
        batch_data,
        beam_size=5,
        lm_solution=None,
        lm_weight=1.0,
        reorder_dict_map=None,
        max_len_a=0,
        max_len_b=2000,
        normalize_scores=True,
        len_penalty=1.0,
        nbest_out=False,
    ):
        '''
        Beam inference

        Args:
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
        '''

        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        ############ Acoustic ############
        encoder_out, encoder_out_mask, _, mtl_logits = self.encoder(batch_data)

        if lm_solution is None or lm_weight == 0.0:
            beam_infer_rlt = self.beam_searcher(encoder_out, encoder_out_mask, mtl_logits)
        else:
            # TODO (houjunfeng) add beam search with lm
            beam_infer_rlt, _ = self.criterion_module.beam_search_with_lm(
                encoder_out,
                encoder_out_mask,
                self.decoder_module,
                self.jointer_module,
                beam_size=beam_size,
                lm=lm_solution,
                lm_weight=lm_weight,
                reorder_dict_map=reorder_dict_map,
                mtl_logits=mtl_logits,
            )

        out_rlt_list = []
        nbest_rlt_list = []
        for bid in range(bsz):
            hyp_token_list = beam_infer_rlt[bid]
            if len(hyp_token_list) > 0:
                out_rlt_list.append(hyp_token_list[0]['tokens'])  # top1
                nbest_rlt_list.append([t['tokens'] for t in hyp_token_list])  # topK

        if nbest_out:
            return out_rlt_list, nbest_rlt_list
        return out_rlt_list


@register_solution("LASMWERModel")
class LASMWERModel(BaseLASModel):
    '''
    New criterion for LAS MWER training.
    - Encoder
        - frontend: VGGFrontEnd
        - backbone: TransformerBackbone, DFSMNBackbone
        - head:
    - Attention
    - Decoder
    - criterion
    '''

    def forward(self, batch_data):
        '''
        Forward for LAS MWER model
        '''
        encoder_out, backbone_mask, _, _ = self.encoder(batch_data)

        forward_out = self.criterion_module(
            encoder_out, backbone_mask, batch_data, self.decoder_module
        )

        return forward_out


@register_solution("StreamingLASModel")
class StreamingLASModel(BaseLASModel):
    '''
    Streaming LAS implementation based on SAMIASR
    '''

    def __init__(self, args):
        '''
        init function for streaming LAS skeleton model.
        '''
        super().__init__(args)
        self.streaming_decoder = args.get(
            "streaming_decoder", False
        )  # indicate at which training step
        if self.streaming_decoder:
            self.decoder_chunk_size = args.get("decoder_chunk_size", 8)  # chunk size
            self.decoder_left_chunk_num = args.get(
                "decoder_left_chunk_num", 4
            )  # number of encoder frames to look at at current step
            self.decoder_right_peak = args.get(
                "decoder_right_peak", 1
            )  # number of future peaks to look at at current step

    @staticmethod
    def streaming_decoder_mask(
        xs_lens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_hat_ctc: torch.Tensor,
        chunk_size: int = 8,
        left_chunk_num: int = 4,
        right_peak_num: int = 1,
        blank_idx: int = 0,
    ):
        """
        Args:
            xs_lens:    (B, )
            ys_hat_ctc: (B, T)
            ys_in_pad:  (B, L)          L is num of tokens in sentences
        Return:
            bool Tensor  (B, L, T)
        """
        assert chunk_size > 0 and left_chunk_num >= 0 and right_peak_num >= 0, "Invalid params"

        # pylint: disable=invalid-name
        B, T = ys_hat_ctc.size()
        B, L = ys_in_pad.size()
        enc_dec_mask = torch.zeros(B, L, T, dtype=torch.int32).to(ys_in_pad.device)

        for i in range(B):
            # get valid frame
            frame_num = int(xs_lens[i])
            peaks = []
            for frame, token in enumerate(ys_hat_ctc[i][:frame_num]):
                if token != blank_idx:
                    peaks.append(frame)

            for j in range(L):
                start_frame, end_frame = 0, frame_num
                # shrink left boundary
                if len(peaks) > 0:
                    current_peak = j if j < len(peaks) else -1
                    current_chunk = peaks[current_peak] // chunk_size
                    left_chunk = current_chunk - left_chunk_num

                    if peaks[current_peak] % chunk_size == 0:
                        # frame_idx is start from 0
                        # this peak frame is the start frame of a chunk
                        left_chunk -= 1
                    start_frame = left_chunk * chunk_size
                    start_frame = max(start_frame, 0)

                # shrink right boundary
                right_peak = j + right_peak_num
                if right_peak < len(peaks):
                    if (peaks[right_peak] + 1) % chunk_size == 0:
                        # frame_idx is start from 0
                        # this peak frame is the end frame of a chunk
                        end_frame = peaks[right_peak]
                    else:
                        # attention whole chunk
                        end_frame = (peaks[right_peak] // chunk_size + 1) * chunk_size
                end_frame = min(end_frame, frame_num)
                enc_dec_mask[i, j, start_frame:end_frame] = 1

        return enc_dec_mask

    def get_encoder_decoder_mask(self, encoder_out_mask, decoder_in, y_ctc_hat):
        '''get_encoder_decoder_mask'''
        encoder_out_lens = (encoder_out_mask.sum(dim=1)).int()
        return self.streaming_decoder_mask(
            encoder_out_lens,
            decoder_in,
            y_ctc_hat,
            self.decoder_chunk_size,
            self.decoder_left_chunk_num,
            self.decoder_right_peak,
            blank_idx=self.criterion_module.blank,
        )

    def forward(self, batch_data):
        '''
        Forward for streaming LAS model
        '''
        encoder_out, backbone_mask, _, mtl_logits = self.encoder(batch_data)
        # get encoder mask for decoder
        if self.streaming_decoder and mtl_logits is not None:
            y_ctc_hat = torch.argmax(mtl_logits, dim=2)
            encoder_decoder_mask = self.get_encoder_decoder_mask(
                backbone_mask, batch_data['prev_char'], y_ctc_hat
            )
        if self.streaming_decoder:
            logits = self.decoder(encoder_out, encoder_decoder_mask, batch_data)
        else:
            logits = self.decoder(encoder_out, backbone_mask, batch_data)
        src_mask = batch_data['src_mask']
        target = batch_data['char']
        target_mask = batch_data['char_mask']
        forward_out = self.criterion_module(
            logits, src_mask, target, target_mask, backbone_mask, mtl_logits, self.mtl_type
        )
        return forward_out
