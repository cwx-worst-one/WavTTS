''' rnnt_jointer
'''
import torch
from torch import nn
import torch.nn.functional as F
from core.extensions import ShallowJointFunction


class ShallowJointerSimple(nn.Module):
    '''
    ShallowJointer: simply add acoustic output and predictor output with a tanh
    '''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.args = args

    # pylint: disable=no-self-use
    def forward(self, acoustic_out, predictor_out, _input_lengths, _target_lengths):
        '''forward
        Args:
            acoustic_out: encoder's output, [B, T, H]
            predictor_out: predictor's output, [B, U, H]
            input_lengths: length of encoder output (with frame downsampled), [B]
            target_lengths: length of targets, [B]
        Return:
            joint_output: output of jointer, [B, T, U, H]
        '''
        joint_input = acoustic_out.unsqueeze(2) + predictor_out.unsqueeze(1)
        joint_output = torch.tanh(joint_input)
        return joint_output

    @staticmethod
    def forward_step(acoustic_out, predictor_out):
        '''forward by a step with the encoder output from a frame
            and the predictor output from a token
        Args:
            acoustic_out: encoder output, [B, H] or [H]
            predictor_out: predictor output, [B, H] or [H]
        Return:
            jointer output of a frame and a token, the sampe shape as acoustic_out
        '''
        joint_input = acoustic_out + predictor_out
        joint_output = torch.tanh(joint_input)
        return joint_output


class ShallowJointer(nn.Module):
    '''ShallowJointer:
    If jointer_simple_fusion is true, the softmax values of encoder output and
    predictor output are first multiplied, then activated by tanh,
    and finally passed into a linear layer with ReLU activation.
    Otherwise, there will be one more linear layer with ReLU,
    compared to ShallowJointerSimple
    '''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.args = args
        self.joint_module = nn.Sequential(
            *[nn.Linear(args.jointer_hidden_size, args.jointer_hidden_size), nn.LeakyReLU(0.2)]
        )

    def forward(self, acoustic_out, predictor_out, _input_lengths, target_lengths):
        '''
        forward in two cases, which is depended by concate_U
        When concate_U is true, we can save a lot of GPU memory
        Args:
            acoustic_out: [B, T, H], the output of encoder
            predictor_out: [B, U, H], the output of predictor
            input_lengths: [B], the frame length after downsampling
            target_lengths: [B], the target lengths
        Return:
            joint_output: output of jointer, with the shape of
                          [sum(Ui), T, H] if concate_U, else [B, T, U, H]
        '''
        if self.args.concate_U and not self.args.jointer_simple_fusion:
            joint_input = ShallowJointFunction.apply(
                acoustic_out,
                predictor_out,
                target_lengths,
                self.args.concate_U,
                self.args.jointer_simple_fusion,
            )
        else:
            joint_input = self.join_input(acoustic_out, predictor_out, target_lengths)
        joint_output = self.joint_module(joint_input)
        return joint_output

    def join_input(self, acoustic_out, predictor_out, target_lengths):
        '''join input.'''
        if self.args.concate_U:
            # will save gpu memory from [B, T, U, H] to [sum(Ui), T, H]
            acoustic_out = acoustic_out.unsqueeze(1)  # [B, 1, T, H]
            predictor_out = predictor_out.unsqueeze(2)  # [B, U, 1, H]
            res = []
            for ao, po, tl in zip(acoustic_out, predictor_out, target_lengths):
                if self.args.jointer_simple_fusion:
                    res += [torch.exp(F.log_softmax(ao, -1) + F.log_softmax(po[: tl + 1], -1))]
                    # equal: F.softmax(ao, -1)*F.softmax(po[:tl + 1], -1)
                else:
                    res += [(ao + po[: tl + 1])]  # [Ui, T, H]
            joint_input = torch.cat(res, dim=0)  # [sum(Ui), T, H]
        else:
            if self.args.jointer_simple_fusion:
                joint_input = torch.exp(
                    F.log_softmax(acoustic_out.unsqueeze(2), -1)
                    + F.log_softmax(predictor_out.unsqueeze(1), -1)
                )
            else:
                joint_input = acoustic_out.unsqueeze(2) + predictor_out.unsqueeze(1)
        joint_input = torch.tanh(joint_input)
        return joint_input

    def forward_step(self, acoustic_out, predictor_out):
        '''the same as that in ShallowJointerSimple'''
        if self.args.jointer_simple_fusion:
            joint_input = torch.exp(
                F.log_softmax(acoustic_out, dim=-1) + F.log_softmax(predictor_out, dim=-1)
            )
        else:
            joint_input = acoustic_out + predictor_out
        joint_input = torch.tanh(joint_input)
        joint_output = self.joint_module(joint_input)
        return joint_output
