"""best-rq pretrain"""
import logging
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from core.models.pretrained.utils import compute_mask_indices
from core.models.pretrained.quantizer import RandomProjectionQuantizer
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.pretrained.acoustic_tokenizer import *
from core.criterions.criterion import Xentropy
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.asr.base_rnnt_model import BaseRnntModel
from core.solutions.asr.base_cif_model import BaseCifModel
from core.utils import dist_hdfs_get, get_rank


@register_solution("BestRqPretrainModel")
class BestRqPretrainModel(BaseSolution):
    """best-rq pretrain model"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args

        self.acoustic_front_end_module = eval(args.front_end_type)(args)
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)

        self.encoder_output_dim = args.encoder_output_dim
        self.quantizer_input_dim = args.quantizer_input_dim

        self.args.n_softmax = getattr(args, 'n_softmax', 1)
        # the number of softmax. See https://arxiv.org/abs/2303.01037
        self.args.mask_prob = getattr(args, 'mask_prob', 0.08)
        # See HuBERT paper https://arxiv.org/pdf/2106.07447.pdf, Sec. IV
        self.args.mask_span_length = getattr(args, 'mask_span_length', 10)  # 100ms
        self.args.mask_noise_std = getattr(args, 'mask_noise_std', 0.1)
        self.args.valid_mask_start_stride = getattr(args, 'valid_mask_start_stride', 1000)
        self.args.valid_mask_span_length = getattr(args, 'valid_mask_span_length', 10)
        self.args.codebook_size = getattr(args, 'codebook_size', 8192)
        self.args.codebook_dim = getattr(args, 'codebook_dim', 16)
        self.args.prototype_initialization_method = getattr(
            args, 'prototype_initialization_method', 'gaussian'
        )
        self.args.projection_initialization_method = getattr(
            args, 'projection_initialization_method', 'xavier'
        )

        self.proj_heads = nn.Linear(
            args.encoder_output_dim, self.args.n_softmax * self.args.codebook_size, bias=False
        )

        self.quantizer = RandomProjectionQuantizer(
            input_dim=self.quantizer_input_dim,
            codebook_size=self.args.codebook_size,
            codebook_dim=self.args.codebook_dim,
            prototype_initialization_method=self.args.prototype_initialization_method,
            projection_initialization_method=self.args.projection_initialization_method,
            quantizer_num=self.args.n_softmax,
        )

        self.criterion = Xentropy(self.args)

        if self.args.front_end_type == "Conv2dPooling":
            # to be consistent with Conv2dPooling
            self.unfolder = nn.Unfold(
                kernel_size=(3, 1),
                dilation=1,
                padding=(self.args.front_end_padding, 0),
                stride=(2, 1),
            )
        elif self.args.front_end_type == "CifOriginalFrontend":
            raise NotImplementedError('CifOriginalFrontend is not supported now.')
        else:
            raise ValueError("Unsupported front-end type: {}".format(self.args.front_end_type))

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

    def _unfold(self, fbank, fbank_mask):
        """A helper function for unfold fbank"""

        B, T, D = fbank.shape
        assert fbank_mask.shape[1] == T
        assert fbank_mask.shape[0] == B

        if self.args.front_end_type == "Conv2dPooling":
            unfold_fbank = (
                self.unfolder(fbank.unsqueeze(1))
                .reshape(B, 3, -1, D)
                .transpose(1, 2)
                .reshape(B, -1, 3 * D)
            )
            unfold_fbank_mask = (
                self.unfolder(fbank_mask.unsqueeze(1).unsqueeze(3))
                .reshape(B, 3, -1, 1)
                .transpose(1, 2)
                .reshape(B, -1, 3)
                .sum(-1)
                > 0
            ).float()
            # XOR, i.e., if one frame is masked,
            # the stack of frames which incl!ude this frame are all masked
        elif self.args.front_end_type == "CifOriginalFrontend":
            raise NotImplementedError('CifOriginalFrontend is not supported now.')
        else:
            raise ValueError("Unsupported front-end type: {}".format(self.args.front_end_type))
        return unfold_fbank, unfold_fbank_mask

    def _subsample(self, fbank, fbank_mask):
        """Subsample the sequence in terms of the front-end."""

        if self.args.front_end_type == "Conv2dPooling":
            fbank, fbank_mask = self._unfold(fbank, fbank_mask)
            fbank, fbank_mask = self._unfold(fbank, fbank_mask)
        elif self.args.front_end_type == "CifOriginalFrontend":
            raise NotImplementedError('CifOriginalFrontend is not supported now.')
        else:
            raise ValueError("Unsupported front-end type: {}".format(self.args.front_end_type))
        return fbank, fbank_mask

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
            for i in range(0, self.args.valid_mask_start_stride, T):
                mask_indicators[
                    :, i : i + self.args.valid_mask_span_length
                ] = 1  # deterministic mask
            mask_indicators = mask_indicators * input_masks
        return mask_indicators

    def forward(self, batch_data):
        """forward"""

        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']

        mask_indicators = self._gen_mask_indicators(fbank_mask)
        subsampled_fbank, subsampled_mask_indicators = self._subsample(fbank, mask_indicators)
        # subsampled_fbank is not masked
        noises = self.args.mask_noise_std * torch.randn_like(fbank, device=fbank.device)
        fbank = fbank * (1.0 - mask_indicators)[:, :, None] + noises * mask_indicators[:, :, None]
        # fbank is masked by Gaussian noise

        B, T_sub, D_sub = subsampled_fbank.shape
        codes = self.quantizer.forward(subsampled_fbank.view(B * T_sub, D_sub))

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

        B, T_enc, D = encoder_backbone_out.shape
        assert T_sub == T_enc
        logits = self.proj_heads(encoder_backbone_out).view(
            B, T_enc, self.args.n_softmax, self.args.codebook_size
        )
        # (B, T, n_softmax, codebook_size)
        logits = logits.transpose(1, 2).reshape(
            B * self.args.n_softmax, T_enc, self.args.codebook_size
        )
        targets = (
            codes.view(B, T_enc, self.args.n_softmax)
            .transpose(1, 2)
            .reshape(B * self.args.n_softmax, T_enc)
        )
        backbone_mask = (
            backbone_mask.view(B, 1, T_enc)
            .repeat(1, self.args.n_softmax, 1)
            .reshape(B * self.args.n_softmax, T_enc)
        )
        subsampled_mask_indicators = (
            subsampled_mask_indicators.view(B, 1, T_enc)
            .repeat(1, self.args.n_softmax, 1)
            .reshape(B * self.args.n_softmax, T_enc)
        )
        target_mask = subsampled_mask_indicators * backbone_mask

        forward_out = self.criterion(
            logits=logits, src_mask=backbone_mask, target=targets, target_mask=target_mask
        )
        forward_out['frame_num'] = fbank_mask.float().sum()
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
@register_solution("BestRqPretrainedRnntModel")
class BestRqPretrainedRnntModel(BaseRnntModel):
    """An RNN-T model which uses BEST-RQ pretrained encoder"""

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


@register_solution("BestRqPretrainedCifModel")
class BestRqPretrainedCifModel(BaseCifModel):
    """An CIF model which uses BEST-RQ pretrained encoder"""

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


@register_solution("KmeansTrainingModel")
class KmeansTrainingModel(BaseSolution):
    """KmeansTrainingModel"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args
        self.acoustic_tokenizer = eval(args.acoustic_tokenizer_type)(args)

        assert self.args.acoustic_tokenizer_return_distance

        self.fake_param = nn.Parameter(torch.tensor(1.0))

    def set_num_updates(self, num_updates):
        '''set_num_updates'''
        self.num_updates = num_updates

    def forward(self, batch_data):
        """forward"""

        inputs = batch_data['src']  # (B, T, ndim)
        input_masks = batch_data['src_mask']
        dists, masks = self.acoustic_tokenizer.forward(inputs, input_masks)

        forward_out = {}
        forward_out["distance"] = dists.sum() / masks.sum()
        forward_out["frame_size"] = masks.sum()
        forward_out["frame_num"] = input_masks.sum()
        forward_out["utt_num"] = input_masks.shape[0]
        forward_out["loss"] = dists.sum() / masks.sum()  # A fake loss
        forward_out["backward_loss"] = 2 * self.fake_param  # A fake loss
        return forward_out
