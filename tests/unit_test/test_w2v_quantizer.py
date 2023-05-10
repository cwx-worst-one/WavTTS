'''
test wav2vec quantizer mm
'''
import torch


def test_w2v_quantizer_bmm():
    '''test w2v quantizer bmm'''
    bsz = 10
    tsz = 22
    groups = 2
    num_vars = 32

    x = torch.randn(bsz * tsz, groups * num_vars).cuda().requires_grad_()
    var = torch.randn(1, groups * num_vars, 52).cuda().requires_grad_()

    mx = x.clone().detach_().requires_grad_()
    mvar = var.clone().detach_().requires_grad_()

    # standard process
    x = x.unsqueeze(-1) * var
    x = x.view(bsz * tsz, groups, num_vars, -1)
    x = x.sum(-2)
    x = x.view(bsz, tsz, -1)

    std_loss = x.sum()
    x.retain_grad()

    std_loss.backward()
    std_x_grad = x.grad.data

    # current process
    mx = mx.reshape(bsz * tsz, groups, num_vars).permute(1, 0, 2)
    mvar = mvar.reshape(groups, num_vars, -1)
    mx = torch.bmm(mx, mvar).permute(1, 0, 2).reshape(bsz, tsz, -1)

    my_loss = mx.sum()
    mx.retain_grad()

    my_loss.backward()
    my_x_grad = mx.grad.data

    assert torch.allclose(x, mx, atol=1e-4)
    assert torch.allclose(std_loss, my_loss, atol=1e-5)
    assert torch.allclose(std_x_grad, my_x_grad, atol=1e-5)
