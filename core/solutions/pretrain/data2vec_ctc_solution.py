"""data2vec ctc"""

import contextlib
import torch
from core.models.pretrained.data2vec_model import *
from core.criterions.criterion import *
from core.solutions.base_solution import register_solution
from core.solutions.pretrain.wav2vec_ctc_solution import BaseWav2vecCtcModel


@register_solution("BaseData2vecCtcModel")
class BaseData2vecCtcModel(BaseWav2vecCtcModel):
    """data2vec ctc model"""

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
            if name.startswith('data2vec_ctc_model.'):
                # bytespeech finetuned model
                new_name = name.replace('data2vec_ctc_model.', '')
            elif name.startswith('data2vec_model.'):
                # bytespeech pretrained model
                new_name = name.replace('data2vec_model.', 'w2v_model.')
            state_dict[new_name] = param

    def forward(self, batch_data, tbc=True):
        """forward"""
        padding_mask = (1 - batch_data['src_mask']).int().bool()

        w2v_args = {
            "batch_data": batch_data,
            "padding_mask": padding_mask,
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
        padding_mask = (1 - batch_data['src_mask']).int().bool()

        w2v_args = {
            "batch_data": batch_data,
            "padding_mask": padding_mask,
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
