''' test optmized lstm. '''
import time
import torch
from core.models.layers.lstmp_layer import LSTM
from core.extensions import amp_init


def test_opt_lstm_speed():
    '''test lstm speed.'''
    freq = 10
    for cls_name, module in {'torch LSTM': torch.nn.LSTM, 'My LSTM': LSTM}.items():
        for dtype in ['float', 'half']:
            lstm = module(512, 2048, batch_first=True).cuda()
            x = torch.rand(39, 104, 512).cuda()
            if dtype == 'half':
                lstm.half()
                x = x.half()

            y = lstm(x)  # warmup
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(freq):
                y, _ = lstm(x)
                y.sum().backward()
            torch.cuda.synchronize()
            print('{} {} {:.3f}ms'.format(cls_name, dtype, (time.time() - start) / freq * 1000))


def test_opt_lstm_value():
    '''test lstm value.'''
    atol, rtol = 1e-3, 1e-4
    x1 = torch.rand(39, 104, 512).cuda().detach_().requires_grad_()
    x2 = x1.clone().detach_().requires_grad_()

    lstm = torch.nn.LSTM(512, 1024, num_layers=2, batch_first=True, bidirectional=False).cuda()
    opt0 = torch.optim.SGD(lstm.parameters(), lr=1e-4)
    amp_handler = amp_init(lstm, opt0, opt_level='O1', init_scale=1)

    my_lstm = LSTM(512, 1024, num_layers=2, batch_first=True).cuda()
    opt1 = torch.optim.SGD(my_lstm.parameters(), lr=1e-4)
    my_amp_handler = amp_init(my_lstm, opt1, opt_level='O1', init_scale=1)
    my_lstm.load_state_dict(lstm.state_dict())

    std_h, std_c = lstm(x1)
    std_h.sum().backward()
    my_h, my_c = my_lstm(x2)
    my_h.sum().backward()

    # print(std_h.sum(), my_h.sum())
    assert torch.allclose(std_h, my_h, atol=atol, rtol=rtol)
    assert torch.allclose(std_c[0], my_c[0], atol=atol, rtol=rtol)
    assert torch.allclose(std_c[1], my_c[1], atol=atol, rtol=rtol)
    assert torch.allclose(lstm.weight_ih_l0.grad, my_lstm.weight_ih_l0.grad, atol=1e-1, rtol=1e-2)
    assert torch.allclose(lstm.bias_ih_l0.grad, my_lstm.bias_ih_l0.grad, atol=2e-2, rtol=1e-2)
    assert torch.allclose(lstm.weight_hh_l0.grad, my_lstm.weight_hh_l0.grad, atol=1e-1, rtol=1e-2)
    assert torch.allclose(lstm.bias_hh_l0.grad, my_lstm.bias_hh_l0.grad, atol=2e-2, rtol=1e-2)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-3, rtol=1e-3)
    amp_handler.clear()
    my_amp_handler.clear()
