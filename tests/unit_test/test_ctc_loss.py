'''
test deterministic ctc loss.
'''
import torch
import torch.nn.functional as F
from core.extensions import ctc_loss


def test_deterministic_ctc_loss():
    bsz, time_len, tgt_len, vsz = 92, 104, 23, 12968
    logits = torch.rand([bsz, time_len, vsz], device='cuda')
    target = torch.randint(0, vsz, [bsz, tgt_len], device='cuda')
    input_lens = torch.randint(1, time_len, [bsz], device='cuda')
    target_lens = torch.randint(1, tgt_len, [bsz], device='cuda')

    logits.requires_grad_()
    lprob = F.log_softmax(logits, dim=-1)
    loss = F.ctc_loss(
        lprob.transpose(0, 1).contiguous(),
        target.int(),
        input_lens,
        target_lens,
        blank=0,
        reduction='mean',
        zero_infinity=True,
    )
    loss.backward()
    # print(loss, logits.grad.sum())

    torch.backends.cudnn.deterministic = True
    print('cudnn deterministic', torch.backends.cudnn.deterministic)
    my_logits = logits.clone().detach_().requires_grad_()
    my_lprob = F.log_softmax(my_logits, dim=-1)
    my_loss = ctc_loss(
        my_lprob.transpose(0, 1).contiguous(),
        target.int(),
        input_lens,
        target_lens,
        blank=0,
        reduction='mean',
        zero_infinity=True,
    )
    my_loss.backward()
    # print(my_loss, my_logits.grad.sum())
    assert torch.allclose(loss, my_loss)
    assert torch.allclose(my_logits.grad.data, logits.grad.data, atol=5e-7)
