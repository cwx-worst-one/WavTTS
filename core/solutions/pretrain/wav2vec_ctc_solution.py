"""wav2vec ctc"""

import contextlib
import re
from argparse import Namespace
import torch
from torch import nn
from core.models.pretrained.wav2vec2_model import *
from core.models.pretrained.data2vec_model import *
from core.criterions.criterion import *
from core.solutions.base_solution import BaseSolution, register_solution
from core.models.pretrained.w2l_decoder import W2lGreedyDecoder


@register_solution("BaseWav2vecCtcModel")
class BaseWav2vecCtcModel(BaseSolution):
    """wav2vec ctc model"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        tgt_dict = args.tgt_dict
        self.apply_mask = args.apply_mask

        d = args.encoder_embed_dim

        self.w2v_model = eval(args.wav2vec_type)(args)

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        self.proj = nn.Linear(d, len(tgt_dict))
        nn.init.xavier_uniform_(self.proj.weight)
        if hasattr(self.proj, 'bias'):
            nn.init.constant_(self.proj.bias, 0.0)

        self.criterion_module = eval(args.ctc_type)(args)
        self.decoder_module = eval(args.decoder_type)(args)
        self.w2l_decoder = None
        self.build_decoder(args)

        self._register_load_state_dict_pre_hook(self._model_load_hook)

    def build_decoder(self, args):
        """build_decoder"""
        dec_args = Namespace()
        dec_args.nbest = args.nbest
        dec_args.criterion = "ctc"
        dec_args.kenlm_model = args.kenlm
        dec_args.lexicon = args.lm_lexicon
        dec_args.beam = args.beam
        dec_args.beam_size_token = min(args.beam_size_token, len(args.tgt_dict))
        dec_args.beam_threshold = args.beam_threshold
        dec_args.lm_weight = args.lm_weight
        dec_args.word_score = args.word_score
        dec_args.unk_weight = -math.inf
        dec_args.sil_weight = args.sil_weight
        dec_args.tgt_dict = args.tgt_dict
        dec_args.w2l_decoder = args.w2l_decoder
        args = dec_args

        decoder = None
        w2l_decoder = getattr(args, "w2l_decoder", None)
        if w2l_decoder == 'greedy':
            decoder = W2lGreedyDecoder(args, args.tgt_dict)
        self.w2l_decoder = decoder

    def hyp_post_process(self, hyp_str):
        """hyp_post_process"""
        unit_type = self.args.modeling_unit_type
        if unit_type == 'bpe':
            hyp_str = hyp_str.replace('^', '')
            hyp_str = hyp_str.replace('@@ ', '')
        if unit_type == 'char':
            hyp_str = hyp_str.replace(' ', '')
            hyp_str = hyp_str.replace('|', ' ')

        hyp_str = re.sub(r"\<[^>]+\>", "", hyp_str)  # NOTE for lid
        return hyp_str

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
        """Transform bytespeech-chkpt to dolphin-chkpt"""
        old_state_dict = state_dict.copy()
        state_dict.clear()
        for name, param in old_state_dict.items():
            new_name = name
            if name.startswith('wav2vec_ctc_model.'):
                # bytespeech finetuned model
                new_name = name.replace('wav2vec_ctc_model.', '')
            elif name.startswith('wav2vec_model.'):
                # bytespeech pretrained model
                new_name = name.replace('wav2vec_model.', 'w2v_model.')
            state_dict[new_name] = param

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.num_updates = num_updates
        self.w2v_model.encoder.num_updates = num_updates

    def forward(self, batch_data, tbc=True):
        """forward"""
        padding_mask = (1 - batch_data['src_mask']).int().bool()

        w2v_args = {
            "batch_data": batch_data,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        net_output = {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "labels": batch_data["char"],
            "src_mask": batch_data["src_mask"],
        }
        forward_out = self.criterion_module(net_output)
        decoder_out = self.decoder_module(
            batch_data,
            self.w2l_decoder,
            forward_out,
        )
        forward_out.update(decoder_out)
        if self.training:
            self.num_updates += 1

        return forward_out

    @torch.no_grad()
    def inference(self, batch_data, tbc=True):
        """inference"""
        w2v_args = {
            "batch_data": batch_data,
            "mask": self.apply_mask and self.training,
        }

        x, padding_mask = self.w2v_model.extract_features(**w2v_args)

        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        net_output = {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }
        lprobs = F.log_softmax(x.float(), dim=-1).contiguous()

        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1).cpu()

        lprobs = lprobs.transpose(0, 1).contiguous().cpu()

        if self.args.w2l_decoder == 'wfst':
            return lprobs, input_lengths

        hyps = self.w2l_decoder.decode(lprobs, input_lengths)
        results = []
        for hyp in hyps:
            hyp = hyp[0]  # top1
            words = hyp.get('words')
            if words is None:
                tokens = self.args.tgt_dict.string(hyp['tokens'].tolist())
                words = self.hyp_post_process(tokens).split()
            results += [' '.join(words)]

        return results
