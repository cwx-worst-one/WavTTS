""" test apex adam and sgd optimizer """

import torch
import torchvision

try:
    from apex.optimizers import FusedAdam, FusedSGD
except Exception:
    FusedAdam, FusedSGD = None, None
from torch.optim import Adam, SGD, AdamW


def test_fused_adam():
    '''test fused adam.'''
    if FusedAdam is None:
        return
    model = torchvision.models.resnet101().cuda()
    params = list(model.parameters())
    grads = [torch.rand_like(p) for p in params]
    params1, params2 = [], []
    for p, g in zip(params, grads):
        p1 = p.clone().detach()
        p1.grad = g.clone().detach()
        params1.append(p1)
        p2 = p.clone().detach()
        p2.grad = g.clone().detach()
        params2.append(p2)

    std_opt = Adam(params1)
    std_opt.step()
    std_opt.step()
    my_opt = FusedAdam(params2, adam_w_mode=False)
    my_opt.step()
    my_opt.step()
    for std_p, my_p in zip(params1, params2):
        st1 = std_opt.state[std_p]
        st2 = my_opt.state[my_p]
        assert torch.allclose(std_p.data, my_p.data, atol=1e-4)
        assert torch.allclose(std_p.grad.data, my_p.grad.data, atol=1e-8)
        assert torch.allclose(st1['exp_avg'], st2['exp_avg'], atol=1e-7)
        assert torch.allclose(st1['exp_avg_sq'], st2['exp_avg_sq'], atol=1e-7)
    my_opt.load_state_dict(std_opt.state_dict())


def test_fused_adamw():
    '''test fused adamw.'''
    model = torchvision.models.resnet101().cuda()
    params = list(model.parameters())
    grads = [torch.rand_like(p) for p in params]
    params1, params2 = [], []
    for p, g in zip(params, grads):
        p1 = p.clone().detach()
        p1.grad = g.clone().detach()
        params1.append(p1)
        p2 = p.clone().detach()
        p2.grad = g.clone().detach()
        params2.append(p2)

    std_opt = AdamW(params1)
    std_opt.step()
    std_opt.step()
    my_opt = FusedAdam(params2, adam_w_mode=True, weight_decay=1e-2)
    my_opt.step()
    my_opt.step()
    for std_p, my_p in zip(params1, params2):
        st1 = std_opt.state[std_p]
        st2 = my_opt.state[my_p]
        assert torch.allclose(std_p.data, my_p.data, atol=1e-4)
        assert torch.allclose(std_p.grad.data, my_p.grad.data, atol=1e-8)
        assert torch.allclose(st1['exp_avg'], st2['exp_avg'], atol=1e-7)
        assert torch.allclose(st1['exp_avg_sq'], st2['exp_avg_sq'], atol=1e-7)
    my_opt.load_state_dict(std_opt.state_dict())


def test_fused_sgd():
    '''test fused sgd.'''
    model = torchvision.models.resnet101().cuda()
    params = list(model.parameters())
    grads = [torch.rand_like(p) for p in params]
    params1, params2 = [], []
    for p, g in zip(params, grads):
        p1 = p.clone().detach()
        p1.grad = g.clone().detach()
        params1.append(p1)
        p2 = p.clone().detach()
        p2.grad = g.clone().detach()
        params2.append(p2)

    std_opt = SGD(params1, lr=0.02, momentum=0.9, weight_decay=0.0001)
    std_opt.step()
    std_opt.step()
    my_opt = FusedSGD(params2, lr=0.02, momentum=0.9, weight_decay=0.0001)
    my_opt.step()
    my_opt.step()
    for std_p, my_p in zip(params1, params2):
        st1 = std_opt.state[std_p]
        st2 = my_opt.state[my_p]
        assert torch.allclose(std_p.data, my_p.data, atol=1e-4)
        assert torch.allclose(std_p.grad.data, my_p.grad.data, atol=1e-8)
        assert torch.allclose(st1['momentum_buffer'], st2['momentum_buffer'], atol=1e-7)
    my_opt.load_state_dict(std_opt.state_dict())


if __name__ == "__main__":
    test_fused_adam()
    test_fused_adamw()
    test_fused_sgd()
