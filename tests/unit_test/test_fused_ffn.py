''' test fused ffn. '''
import torch
from core.models.layers.feed_forward import FFN


def test_fused_ffn():
    '''test fused ffn.'''
    embed = 256
    hidden = 1024
    drop = 0.0
    act = 'gelu'
    std_ffn = FFN(embed, hidden, drop, act).cuda()
    fused_ffn = FFN(embed, hidden, drop, act).cuda()
    fused_ffn.load_state_dict(std_ffn.state_dict())

    std_x = torch.randn(7, 33, embed).cuda().requires_grad_()
    fused_x = std_x.clone().detach().requires_grad_()

    std_y = std_ffn(std_x, fused=False)
    fused_y = fused_ffn(fused_x, fused=True)
    torch.allclose(std_y, fused_y)

    std_y.sum().backward()
    fused_y.sum().backward()
    torch.allclose(std_ffn.w_1.weight.grad, fused_ffn.w_1.weight.grad)
    torch.allclose(std_ffn.w_1.bias.grad, fused_ffn.w_1.bias.grad)
    torch.allclose(std_ffn.w_2.weight.grad, fused_ffn.w_2.weight.grad)
    torch.allclose(std_ffn.w_2.bias.grad, fused_ffn.w_2.bias.grad)
    torch.allclose(std_x.grad, fused_x.grad)
