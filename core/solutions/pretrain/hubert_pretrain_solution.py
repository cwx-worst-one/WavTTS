"""best-rq pretrain"""
import logging
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from core.models.pretrained.utils import compute_mask_indices
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.pretrained.acoustic_tokenizer import *
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.asr.base_rnnt_model import BaseRnntModel
from core.solutions.asr.base_cif_model import BaseCifModel


@register_solution("HuBERTPretrainModel")
class HuBERTPretrainModel(BaseSolution):
    """HuBERT pretrain model"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args
        self.acoustic_front_end_module = eval(args.front_end_type)(args)
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)
        self.encoder_output_dim = args.encoder_output_dim

        self.logit_temp = getattr(args, 'logit_temp', 0.1)
        self.args.n_softmax = getattr(args, 'n_softmax', 1)  # TODO support multiple softmax
        self.args.mask_prob = getattr(args, 'mask_prob', 0.08)
        # See HuBERT paper https://arxiv.org/pdf/2106.07447.pdf, Sec. IV
        self.args.mask_span_length = getattr(args, 'mask_span_length', 10)  # 100ms
        self.args.mask_noise_std = getattr(args, 'mask_noise_std', 0.1)
        self.args.valid_mask_start_stride = getattr(args, 'valid_mask_start_stride', 1000)
        self.args.valid_mask_span_length = getattr(args, 'valid_mask_span_length', 10)
        # init tokenizer
        tok_args = args.tokenizer_args
        self.tokenizer = eval(tok_args.type)(tok_args)
        self.tokenizer.requires_grad_ = False
        self.tokenizer.eval()
        # pretrain-only params
        self.final_proj = nn.Linear(args.encoder_output_dim, args.final_dim, bias=False)
        self.label_embs = nn.Parameter(
            torch.FloatTensor(self.tokenizer.codebook_size, args.final_dim)
        )
        nn.init.normal_(self.label_embs)

    def frontend(self, fbank, mask):
        '''frontend'''
        if self.acoustic_front_end_module is not None:
            return self.acoustic_front_end_module(fbank, mask)
        # Nothing to do in frontend
        return fbank, mask, "BTN"

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''Backbone'''
        attn_mask = False
        if (
            self.args.acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.acoustic_backbone_module.causal_transformer
        ):
            attn_mask = True
        backbone_out = self.acoustic_backbone_module(
            inputs, mask, attn_mask=attn_mask, frontend_shape=frontend_shape
        )
        return backbone_out

    def _gen_mask_indicators(self, input_masks):
        """Generate mask indicators, where 1 means masked out."""
        device = input_masks.device
        B, T = input_masks.shape
        if self.training:
            mask_indicators = compute_mask_indices(
                shape=(B, T),
                padding_mask=1 - input_masks,
                mask_prob=self.args.mask_prob,
                mask_length=self.args.mask_span_length,
                mask_type="static",  # The same with BEST-RQ paper
                mask_other=0.0,
                mask_minlen_type="crop_end",
                min_masks=1,
                no_overlap=False,
                min_space=0,
                require_same_masks=True,
                mask_dropout=0.0,
            )
            mask_indicators = (
                torch.from_numpy(mask_indicators).to(input_masks.device).long() * input_masks
            )
        else:
            mask_indicators = (torch.zeros_like(input_masks, device=device)).long()
            for i in range(0, T, self.args.valid_mask_start_stride):
                mask_indicators[
                    :, i : i + self.args.valid_mask_span_length
                ] = 1  # deterministic mask
            mask_indicators = mask_indicators * input_masks
        return mask_indicators

    def forward(self, batch_data):
        """forward"""

        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']

        front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
        mask_indicators = self._gen_mask_indicators(backbone_mask)
        # fbank is masked by Gaussian noise
        noises = self.args.mask_noise_std * torch.randn_like(front_end_out, device=fbank.device)
        front_end_out = (
            front_end_out * (1.0 - mask_indicators)[:, :, None]
            + noises * mask_indicators[:, :, None]
        )
        encoder_backbone_out = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)
        # generate targets
        if self.args.tokenizer_args.get('feat') == 'mfcc':
            targets = self.tokenizer(batch_data['mfcc'])
            ratio = targets.size(1) // encoder_backbone_out.size(1)
            targets = targets[:, ::ratio]
        else:
            targets, _ = self.tokenizer(fbank, fbank_mask)
        # compute logits
        proj_x = self.final_proj(encoder_backbone_out)
        proj_x = F.normalize(proj_x, dim=1)
        label_embs = F.normalize(self.label_embs, dim=1)
        logits = F.linear(proj_x, label_embs) / self.logit_temp
        # compute loss
        B, T, C = logits.shape
        loss_all = F.cross_entropy(logits.view(B * T, C), targets.view(-1), reduction='none')
        mask_m = ((mask_indicators.bool()) * backbone_mask.bool()).view(-1)
        mask_u = ((~mask_indicators.bool()) * backbone_mask.bool()).view(-1)
        cnt_m, cnt_u = mask_m.sum(), mask_u.sum()
        masked_loss = loss_all.masked_select(mask_m).sum() / cnt_m
        unmasked_loss = loss_all.masked_select(mask_u).sum() / cnt_u
        loss = (
            self.args.masked_loss_scale * masked_loss
            + self.args.unmasked_loss_scale * unmasked_loss
        )
        cnt = self.args.masked_loss_scale * cnt_m + self.args.unmasked_loss_scale * cnt_u
        # compute accuracy
        acc_all = (logits.argmax(-1) == targets).view(-1)
        acc_m = acc_all.masked_select(mask_m).sum() / cnt_m
        acc_u = acc_all.masked_select(mask_u).sum() / cnt_u
        # collect outputs
        forward_out = {
            'loss': loss,
            'backward_loss': loss,
            'loss_m': masked_loss,
            'loss_u': unmasked_loss,
            'acc_m': acc_m,
            'acc_u': acc_u,
            'cnt_m': cnt_m,
            'cnt_u': cnt_u,
            'cnt': cnt,
            'utt_num': fbank_mask.shape[0],
            'frame_size': fbank_mask.sum(),
        }

        return forward_out

    def extract_encoder_backbone_out(self, batch_data):
        """Extract output of the backbone."""

        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']

        if not self.training:
            with torch.no_grad():
                # front-end
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                # output (B, T, N)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
        else:
            with torch.enable_grad():
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )

        return encoder_backbone_out, backbone_mask

    def extract_codes(self, batch_data):
        """forward"""

        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        if not self.training:
            with torch.no_grad():
                # front-end
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                # output (B, T, N)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
        else:
            with torch.enable_grad():
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )

        batch_size, seq_len_enc, _ = encoder_backbone_out.shape

        logits = self.proj_heads(encoder_backbone_out).view(
            batch_size, seq_len_enc, self.args.n_softmax, self.args.codebook_size
        )

        predicted_codes = torch.argmax(logits, dim=-1).int()  # [batch_size, seq_len_enc, n_softmax]
        # dtype=int to save spaces.
        return predicted_codes, backbone_mask

    def extract_data(self, batch_data, extraction_mode):
        if extraction_mode == "extract_encoder_backbone_out":
            encoder_backbone_out, backbone_mask = self.extract_encoder_backbone_out(batch_data)
            data = encoder_backbone_out
            lengths = torch.sum(backbone_mask, dim=1).long()
        elif extraction_mode == "extract_codes":
            encoder_backbone_out, backbone_mask = self.extract_codes(batch_data)
            data = encoder_backbone_out
            lengths = torch.sum(backbone_mask, dim=1).long()
        else:
            raise ValueError("Unknown extraction_mode: {}".format(extraction_mode))
        return data, lengths

    def set_num_updates(self, num_updates):
        '''set_num_updates'''
        self.num_updates = num_updates


# TODO(baiye): a general finetuned Model from some chkpts
@register_solution("HuBERTPretrainedRnntModel")
class HuBERTPretrainedRnntModel(BaseRnntModel):
    """An RNN-T model which uses HuBERT pretrained encoder"""

    def __init__(self, args):
        """init"""
        super().__init__(args)

        self.encoder_output_dim = args.encoder_output_dim

        self.adaptor = nn.Linear(self.encoder_output_dim, self.encoder_output_dim, bias=False)

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''
        Adding adaptor to the encoder backbone
        '''
        attn_mask = False
        if (
            self.args.acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.causal_transformer
        ):
            attn_mask = True
        backbone_out = self.acoustic_backbone_module(
            inputs, mask, attn_mask=attn_mask, frontend_shape=frontend_shape
        )

        adapted_backbone_out = self.adaptor(backbone_out)

        return adapted_backbone_out


@register_solution("HuBERTPretrainedCifModel")
class HuBERTPretrainedCifModel(BaseCifModel):
    """An CIF model which uses HuBERT pretrained encoder"""

    def __init__(self, args):
        """init"""
        super().__init__(args)
        self._register_load_state_dict_pre_hook(self._model_load_hook)

    @staticmethod
    def _model_load_hook(
        state_dict,
        _prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        old_state_dict = state_dict.copy()
        state_dict.clear()
        for name, param in old_state_dict.items():
            new_name = name
            if name.startswith('acoustic_front_end_module.'):
                new_name = name.replace('acoustic_front_end_module.', 'front_end.')
            elif name.startswith('acoustic_backbone_module.'):
                new_name = name.replace('acoustic_backbone_module.', 'encoder_backbone.')
            state_dict[new_name] = param
