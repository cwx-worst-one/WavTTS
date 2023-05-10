''' test Fused Shallow jointer. '''

import random
import numpy as np
import torch
import torch.nn.functional as F
from core.extensions import ShallowJointFunction


def _origin_fun(a, p, target_lengths, concate_u=1, jointer_simple_fusion=0):
    '''origin func.'''
    if concate_u:
        # will save gpu memory from [B, T, U, H] to [sum(Ui), T, H]
        acoustic_out = a.unsqueeze(1)  # [B, 1, T, H]
        predicter_out = p.unsqueeze(2)  # [B, U, 1, H]
        res = []
        for ao, po, tl in zip(acoustic_out, predicter_out, target_lengths):
            if jointer_simple_fusion:
                res += [torch.exp(F.log_softmax(ao, -1) + F.log_softmax(po[: tl + 1], -1))]
                # equal: F.softmax(ao, -1)*F.softmax(po[:tl + 1], -1)
            else:
                res += [(ao + po[: tl + 1])]  # [Ui, T, H]
        joint_input = torch.cat(res, dim=0)  # [sum(Ui), T, H]
    else:
        if jointer_simple_fusion:
            joint_input = torch.exp(
                F.log_softmax(acoustic_out.unsqueeze(2), -1)
                + F.log_softmax(predicter_out.unsqueeze(1), -1)
            )
        else:
            joint_input = acoustic_out.unsqueeze(2) + predicter_out.unsqueeze(1)
    joint_input = torch.tanh(joint_input)
    return joint_input


def test_fused_shallow_jointer():
    '''test func.'''
    bsz = 35
    u_num = 56
    t_num = 105
    h_num = 768
    target_lengths = [random.randint(1, u_num - 1) for _ in range(bsz)]
    sum_u = sum(target_lengths) + bsz

    np_a = np.random.rand(bsz, t_num, h_num)
    np_p = np.random.rand(bsz, u_num, h_num)
    np_scale = np.random.rand(sum_u, t_num, h_num)
    tensor_scale = torch.from_numpy(np_scale).cuda().float()
    tensor_scale.requires_grad = False

    std_a = torch.from_numpy(np_a).cuda().float().detach().requires_grad_()
    std_p = torch.from_numpy(np_p).cuda().float().detach().requires_grad_()
    std_out = _origin_fun(std_a, std_p, target_lengths)
    (std_out * tensor_scale).sum().backward()

    my_a = torch.from_numpy(np_a).cuda().float().detach().requires_grad_()
    my_p = torch.from_numpy(np_p).cuda().float().detach().requires_grad_()
    my_out = ShallowJointFunction.apply(my_a, my_p, target_lengths, 1, 0)
    (my_out * tensor_scale).sum().backward()

    assert torch.allclose(std_out, my_out)
    assert torch.allclose(std_a.grad.data, my_a.grad.data)
    assert torch.allclose(std_p.grad.data, my_p.grad.data)
