''' loss functions for Model Distillation '''
import torch.nn.functional as F
from core.models.utils import kl_divloss_rnnt


class CrossEntropy:
    """CrossEntropy with Temperature"""

    def __init__(self, kwargs):
        """init func"""
        self.temperature = kwargs.get('temperature', 1)

    def __call__(self, source, target, *args, **kwargs):
        """compute the loss"""
        source = F.log_softmax(source['predicts'] / self.temperature, dim=1)
        target = F.softmax(target['predicts'] / self.temperature, dim=1)
        return F.kl_div(source, target.detach(), reduction='sum')


class RnntKL:
    """Rnnt KL"""

    def __call__(self, source, target, *args, **kwargs):
        """compute the loss"""
        log_probs = source['log_probs']
        teacher_log_probs = target['log_probs']
        distillation_loss = kl_divloss_rnnt(log_probs, teacher_log_probs.detach())
        return distillation_loss


class LasKL:
    """Rnnt KL"""

    def __init__(self, kwargs):
        """init func"""
        self.temperature = kwargs.get('temperature', 1)

    def __call__(self, source, target, *args, **kwargs):
        """compute the loss"""
        source = F.log_softmax(source['las_logits'] / self.temperature, dim=1)
        target = F.softmax(target['las_logits'] / self.temperature, dim=1)
        return F.kl_div(source, target.detach(), reduction='mean')


class LasRescoreKL:
    """Rnnt KL"""

    def __init__(self, kwargs):
        """init func"""
        self.temperature = kwargs.get('temperature', 1)
        self.fw_decoder_weight = kwargs.get('fw_decoder_weight', 0.5)

    def __call__(self, source, target, *args, **kwargs):
        """compute the loss"""
        fw_source = F.log_softmax(source['fw_logits'] / self.temperature, dim=1)
        fw_target = F.softmax(target['fw_logits'] / self.temperature, dim=1)
        fw_loss = F.kl_div(fw_source, fw_target.detach(), reduction='mean')
        bw_source = F.log_softmax(source['bw_logits'] / self.temperature, dim=1)
        bw_target = F.softmax(target['bw_logits'] / self.temperature, dim=1)
        bw_loss = F.kl_div(bw_source, bw_target.detach(), reduction='mean')
        return self.fw_decoder_weight * fw_loss + (1 - self.fw_decoder_weight) * bw_loss


distill_loss_func = {
    'CrossEntropy': CrossEntropy,
    'RnntKL': RnntKL,
    'LasKL': LasKL,
    'LasRescoreKL': LasRescoreKL,
}
