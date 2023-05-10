'''Large margin softmax layer w/ a fully-connected layer.
'''
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LargeMarginSoftmax(nn.Module):
    '''Large margin softmax
    This module contains different types of large margin softmax, e.g.
    angular softmax, additive margin softmax, additive angular margin softmax, etc.
    It also supports conventional softmax.
    The layer is different from a vanilla softmax because it CONTAINS A
    TRAINABLE FC LAYER.

    Reference:
        https://arxiv.org/abs/1704.08063
        https://arxiv.org/abs/1801.05599
        https://arxiv.org/abs/1801.07698
    '''

    def __init__(self, args, input_dim, tgt_size):
        super().__init__()
        self.softmax_type = args.softmax_type
        self.input_size = input_dim
        self.output_size = tgt_size
        self.softmax_renorm = args.softmax_renorm
        self.weight = nn.Parameter(torch.FloatTensor(tgt_size, input_dim))
        nn.init.xavier_uniform_(self.weight)

        # hyperparamters for large margin softmax
        self.softmax_lambda_base = args.get("softmax_lambda_base", 0)
        self.softmax_lambda_min = args.get("softmax_lambda_min", 0)
        self.softmax_lambda_gamma = args.get("softmax_lambda_gamma", 1)
        self.softmax_lambda_power = args.get("softmax_lambda_power", 1)
        self.softmax_margin_start = args.get("softmax_margin_start", 0.0)
        self.softmax_margin_steps = args.get("softmax_margin_steps", None)
        self.by_epoch = args.get("softmax_margin_by_epoch", False)
        self.iters_per_epoch = args.iters_per_epoch
        assert self.softmax_margin_steps is None or isinstance(self.softmax_margin_steps, list)
        self.margin = args.softmax_margin
        self.cos_margin = math.cos(self.margin)
        self.sin_margin = math.sin(self.margin)
        self.thresh = math.cos(math.pi - self.margin)
        self.sinmm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, input_feat, label, step):
        '''forward
        Args:
            input_feat: features with shape[B, ..., D]
            label: label with shape [B, ...]
            step: the training step. used in lambda annealing.
        '''
        # Compute the norm factor for the logits
        softmax_renorm = (
            self.softmax_renorm
            if self.softmax_renorm > 0
            else input_feat.norm(p=2, dim=-1, keepdim=True)
        )
        cos_theta = F.linear(F.normalize(input_feat), F.normalize(self.weight))

        if not self.training:
            # In validation, no need to apply margin.
            logit = softmax_renorm * cos_theta
            return logit, {"margin_lambda": 0}

        # Margin policy
        this_margin = self.margin
        this_cos_margin = self.cos_margin
        this_sin_margin = self.sin_margin
        this_thresh = self.thresh
        this_sinmm = self.sinmm
        if self.softmax_margin_steps is not None:
            num_increase = sum(
                (step // self.iters_per_epoch if self.by_epoch else step) >= s
                for s in self.softmax_margin_steps
            )
            this_margin = self.softmax_margin_start + (
                (self.margin - self.softmax_margin_start)
                / len(self.softmax_margin_steps)
                * num_increase
            )
            this_cos_margin = math.cos(this_margin)
            this_sin_margin = math.sin(this_margin)
            this_thresh = math.cos(math.pi - this_margin)
            this_sinmm = math.sin(math.pi - this_margin) * this_margin

        if self.softmax_type == "additive_margin_softmax":
            phi = cos_theta - this_margin
        elif self.softmax_type == "additive_angular_margin_softmax":
            cos_theta = cos_theta.clamp(-1, 1)  # improve numerical stability
            sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi = cos_theta * this_cos_margin - sin_theta * this_sin_margin  # cos(target+margin)
            phi = torch.where(
                cos_theta > this_thresh,
                phi.type(torch.float32),
                (cos_theta - this_sinmm).type(torch.float32),
            )
        elif self.softmax_type == "angular_softmax":
            raise NotImplementedError("ASoftmax is not implemented.")
        else:
            # Maybe the combined loss?
            raise NotImplementedError("Unknown softmax type {}".format(self.softmax_type))

        # compute lambda for annealing
        if self.softmax_lambda_base > 0:
            lambda_factor = self.softmax_lambda_base * (
                (1.0 + self.softmax_lambda_gamma * step) ** (-self.softmax_lambda_power)
            )
            lambda_factor = np.maximum(float(self.softmax_lambda_min), lambda_factor)
            phi = (
                1.0 / (1.0 + lambda_factor) * phi
                + lambda_factor / (1.0 + lambda_factor) * cos_theta
            )
        else:
            lambda_factor = 0
        one_hot = torch.zeros(cos_theta.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        logit = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logit = softmax_renorm * logit
        return logit, {"margin": this_margin, "margin_lambda": lambda_factor}


class CurricularFace(nn.Module):
    '''Huang Y, Wang Y, Tai Y, et al. Curricularface: adaptive curriculum learning loss
       for deep face recognition. CVPR 2020.
    Reference: https://github.com/HuangYG123/CurricularFace/blob/master/head/metrics.py
    '''

    def __init__(self, args, input_dim, tgt_size):
        super().__init__()
        self.softmax_type = args.softmax_type
        self.input_size = input_dim
        self.output_size = tgt_size
        self.softmax_renorm = args.softmax_renorm
        self.weight = nn.Parameter(torch.FloatTensor(tgt_size, input_dim))
        # nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.weight, std=0.01)

        self.softmax_margin_steps = args.get("softmax_margin_steps", None)
        self.by_epoch = args.get("softmax_margin_by_epoch", False)
        self.iters_per_epoch = args.iters_per_epoch
        assert self.softmax_margin_steps is None or isinstance(self.softmax_margin_steps, list)
        self.margin = args.softmax_margin
        self.cos_margin = math.cos(self.margin)
        self.sin_margin = math.sin(self.margin)
        self.thresh = math.cos(math.pi - self.margin)
        self.sinmm = math.sin(math.pi - self.margin) * self.margin
        self.register_buffer('cft', torch.zeros(1))

    def forward(self, input_feat, label, step):
        '''forward'''
        softmax_renorm = (
            self.softmax_renorm
            if self.softmax_renorm > 0
            else input_feat.norm(p=2, dim=-1, keepdim=True)
        )
        cos_theta = F.linear(F.normalize(input_feat), F.normalize(self.weight))
        # for numerical stability, also make the cos_theta not optimized
        cos_theta = cos_theta.clamp(-1, 1).type(torch.float32)

        if not self.training:
            # In validation, no need to apply margin.
            logit = softmax_renorm * cos_theta
            return logit, {"margin_lambda": 0}

        # Margin policy
        this_margin = self.margin
        if self.softmax_margin_steps is not None:
            num_increase = sum(
                (step // self.iters_per_epoch if self.by_epoch else step) >= s
                for s in self.softmax_margin_steps
            )
            this_margin = self.margin / len(self.softmax_margin_steps) * num_increase

        target_logit = cos_theta[torch.arange(0, input_feat.size(0)), label].view(-1, 1)

        final_target_logit = target_logit - this_margin
        mask = cos_theta > final_target_logit

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.cft = target_logit.mean() * 0.01 + (1 - 0.01) * self.cft
        cos_theta[mask] = hard_example * (self.cft + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        logit = softmax_renorm * cos_theta
        return logit, {"margin": this_margin}


class ClassificationCircleLoss(nn.Module):
    '''Sun Y, Cheng C, Zhang Y, et al. Circle loss: A unified perspective of pair
    similarity optimization. CVPR 2020.
    '''

    def __init__(self, args, input_dim, tgt_size):
        super().__init__()
        self.gamma = args.circle_loss_gamma
        self.margin = args.circle_loss_margin
        self.o_p = 1.0 + self.margin
        self.o_n = -self.margin
        self.delta_p = 1 - self.margin
        self.delta_n = self.margin
        self.weight = nn.Parameter(torch.FloatTensor(tgt_size, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feat, label, **_kwargs):
        '''forward'''
        # Compute the norm factor for the logits
        cos_theta = F.linear(F.normalize(input_feat), F.normalize(self.weight))

        if not self.training:
            # In experiments, the norm of the input feature seems to shrink during training,
            # making the valid loss increase. To avoid this 'inaccurate' valid loss, we fix the
            # norm to gamma.
            logit = self.gamma * cos_theta
            return logit, {"margin_lambda": 0}

        ap = torch.clamp(self.o_p - cos_theta.detach(), min=0)
        an = torch.clamp(cos_theta.detach() - self.o_n, min=0)

        target_logit = self.gamma * ap * (cos_theta - self.delta_p)
        nontar_logit = self.gamma * an * (cos_theta - self.delta_n)

        one_hot = torch.zeros(cos_theta.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # combine target and nontarget logits
        logit = (one_hot * target_logit) + ((1.0 - one_hot) * nontar_logit)
        return logit, {"margin": self.margin}
