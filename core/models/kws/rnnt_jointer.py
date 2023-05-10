'''
rnnt_jointer.py.
'''
import torch
from torch import nn
from core.models.layers.adaptive_softmax import RNNTAdaptiveSoftmax


class ShallowJointer(nn.Module):
    '''Shallow Jointer'''

    def __init__(self, args):
        super().__init__()
        self.joint_module = nn.Sequential(
            *[
                nn.Linear(args.rnnt_hidden_size, args.rnnt_softmax_hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(args.rnnt_softmax_hidden_size, args.tgt_vocab_size),
            ]
        )

    def forward(
        self,
        acoustic_out,
        predicter_out,
        target=None,
        input_lengths=None,
        target_lengths=None,
        **_kwargs
    ):
        """forward"""
        # pylint:disable=unused-argument
        joint_input = acoustic_out + predicter_out
        joint_input = torch.tanh(joint_input)
        logits = self.joint_module(joint_input)
        return logits

    def search(self, acoustic_out, predicter_out):
        '''for decoding'''
        joint_input = acoustic_out + predicter_out
        joint_input = torch.tanh(joint_input)
        logits = self.joint_module(joint_input)
        return logits


class AdaptiveShallowJointer(nn.Module):
    '''Adaptive Shallow Jointer'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.joint_module = nn.Sequential(
            *[nn.Linear(args.rnnt_hidden_size, args.rnnt_softmax_hidden_size), nn.LeakyReLU(0.2)]
        )

        sub_cutoff = list(range(1, args.cutoff_groups + 1))
        cutoff = [self.args.adaptive_head_size] + [
            item * self.args.adaptive_tail_size + self.args.adaptive_head_size
            for item in sub_cutoff
        ]
        if args.tgt_vocab_size > cutoff[-1]:
            cutoff += [args.tgt_vocab_size]
        dropout = 0.0
        self.log_softmax_fc = RNNTAdaptiveSoftmax(
            args.tgt_vocab_size,
            args.rnnt_softmax_hidden_size,
            cutoff,
            dropout,
            concate_U=0,
            use_proj=False,
            tc_align=False,
        )

    def forward(
        self,
        acoustic_out,
        predicter_out,
        target=None,
        input_lengths=None,
        target_lengths=None,
        adaptive_tgt_indices=None,
    ):
        """forward"""
        joint_input = acoustic_out + predicter_out
        joint_input = torch.tanh(joint_input)
        joint_hidden = self.joint_module(joint_input)
        log_probs = self.log_softmax_fc(
            joint_hidden,
            target.long(),
            input_lengths.long(),
            target_lengths.long(),
            adaptive_tgt_indices=adaptive_tgt_indices,
        )
        return log_probs

    def search(self, acoustic_out, predicter_out, adapt_softmax_thresh=1.0, only_head=False):
        '''for decoding search'''
        joint_input = acoustic_out + predicter_out
        joint_input = torch.tanh(joint_input)
        joint_hidden = self.joint_module(joint_input)
        if not self.args.use_head_probs:
            lprobs = self.log_softmax_fc.get_log_prob2(
                joint_hidden, adapt_softmax_thresh=adapt_softmax_thresh, only_head=only_head
            )
        else:
            lprobs = self.log_softmax_fc.get_head_log_prob(joint_hidden)
        return lprobs

    def step(self, acoustic_out, predicter_out):
        '''for decoding step'''
        joint_input = acoustic_out + predicter_out
        joint_input = torch.tanh(joint_input)
        joint_hidden = self.joint_module(joint_input)
        if hasattr(self.log_softmax_fc, 'get_log_prob'):
            lprobs = self.log_softmax_fc.get_log_prob(joint_hidden)
        else:
            logits = self.log_softmax_fc(joint_hidden)
            lprobs = logits.log_softmax(dim=-1)
        return lprobs
