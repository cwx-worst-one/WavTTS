''' test for rnnt_loss. '''
import random

import torch
import torch.nn.functional as F

# pylint: disable=import-error
from warp_rnnt import rnnt_loss as rnnt_loss1
from core.extensions import rnnt_loss as rnnt_loss2


def test_rnnt_loss():
    input_n = 10
    input_t = random.randint(20, 30)
    input_u = random.randint(20, 30)
    input_v = input_u + random.randint(1, 20)

    log_probs = torch.randn(input_n, input_t, input_u, input_v, device='cuda', dtype=torch.float32)
    log_probs = F.log_softmax(log_probs, dim=-1)
    labels = torch.randint(1, input_v, (input_n, input_u - 1), device='cuda', dtype=torch.int32)
    frames_lengths = torch.randint(1, input_t + 1, (input_n,), device='cuda', dtype=torch.int32)
    # label length value in [1, input_u), less than input_u means no last blank
    labels_lengths = torch.randint(1, input_u, (input_n,), device='cuda', dtype=torch.int32)
    # make sure label length less or equal then frames length
    labels_lengths = torch.min(labels_lengths, frames_lengths)

    log_probs_ipt1 = log_probs.clone().detach_().requires_grad_()
    log_probs_ipt2 = log_probs.clone().detach_().requires_grad_()
    for gather in (False, True):
        loss1 = rnnt_loss1(
            log_probs_ipt1,
            labels,
            frames_lengths,
            labels_lengths,
            average_frames=False,
            reduction='mean',
            gather=gather,
            blank=0,
        )
        loss2 = rnnt_loss2(
            log_probs_ipt2,
            labels,
            frames_lengths,
            labels_lengths,
            average_frames=False,
            reduction='mean',
            gather=gather,
            blank=0,
        )
        assert torch.allclose(loss1, loss2)
        loss1.backward()
        loss2.backward()
        assert torch.allclose(log_probs_ipt1.grad.abs().sum(), log_probs_ipt2.grad.abs().sum())
