'''test torch version for rir and unit test for AddRIR and GetRirData'''
import torch
from packaging import version
from core.dataset.preprocess import AddRIR, GetRirData


def new_complex_multiply(in1, in2):
    """for new torch"""
    assert in1.ndim == in2.ndim
    out_real = in1.real * in2.real - in1.imag * in2.imag
    out_imag = in1.real * in2.imag + in1.imag * in2.real
    return torch.complex(out_real, out_imag)


def complex_multiply(in1, in2):
    """complex multiply"""
    assert in1.ndim == in2.ndim
    assert in1.ndim == 3
    out_real = in1[:, :, 0] * in2[:, :, 0] - in1[:, :, 1] * in2[:, :, 1]
    out_imag = in1[:, :, 0] * in2[:, :, 1] + in1[:, :, 1] * in2[:, :, 0]
    return torch.stack([out_real, out_imag], -1)


def test_torch_new():
    '''test for torch1.8.1'''
    if version.parse(torch.__version__) < version.parse('1.7.0'):
        return
    x = torch.tensor(
        [
            [-0.3956, 1.3793, -1.3157, -0.0500, -0.4695],
            [1.6540, -0.1727, 0.5770, -0.3633, -0.2875],
            [0.5534, -0.2503, -1.6930, 0.1228, -1.1333],
        ]
    )
    y = torch.tensor(
        [
            [-3.1566, 0.3886, 0.8590, 0.1631, -0.2441],
            [-0.2281, 1.5672, -3.2595, 0.6775, 0.2452],
            [1.1685, -1.7741, -0.4615, 1.7945, -1.0936],
        ]
    )
    base = torch.tensor(
        [
            [0.4721, -4.5979, 4.2849, 0.8815, 0.6539],
            [0.7049, 3.4640, -6.0773, 2.6001, -2.0960],
            [-0.1638, 1.3206, -3.9576, 5.4950, -1.8152],
        ]
    )
    fft_x = torch.fft.rfft(x)
    fft_h = torch.fft.rfft(y)
    x_conv = torch.fft.irfftn(
        new_complex_multiply(fft_x, fft_h),
        dim=-1,
        s=[5],
    )
    print((base - x_conv).abs().max())
    assert torch.allclose(base, x_conv, atol=1e-4)


def test_torch_old():
    '''test for torch1.5.1'''
    if version.parse(torch.__version__) >= version.parse('1.7.0'):
        return
    x = torch.tensor(
        [
            [-0.3956, 1.3793, -1.3157, -0.0500, -0.4695],
            [1.6540, -0.1727, 0.5770, -0.3633, -0.2875],
            [0.5534, -0.2503, -1.6930, 0.1228, -1.1333],
        ]
    )
    y = torch.tensor(
        [
            [-3.1566, 0.3886, 0.8590, 0.1631, -0.2441],
            [-0.2281, 1.5672, -3.2595, 0.6775, 0.2452],
            [1.1685, -1.7741, -0.4615, 1.7945, -1.0936],
        ]
    )
    base = torch.tensor(
        [
            [0.4721, -4.5979, 4.2849, 0.8815, 0.6539],
            [0.7049, 3.4640, -6.0773, 2.6001, -2.0960],
            [-0.1638, 1.3206, -3.9576, 5.4950, -1.8152],
        ]
    )
    fft_x = torch.rfft(x, signal_ndim=1, normalized=False, onesided=True)
    fft_h = torch.rfft(y, signal_ndim=1, normalized=False, onesided=True)

    x_conv = torch.irfft(
        complex_multiply(fft_x, fft_h),
        signal_ndim=1,
        normalized=False,
        onesided=True,
        signal_sizes=[5],
    )
    print((base - x_conv).abs().max())
    assert torch.allclose(base, x_conv, atol=1e-4)


def get_datas(random_seed=123):
    """get one batch data"""
    torch.random.manual_seed(random_seed)  # get same input
    wav = torch.randn(26, 128240)
    noise = torch.randn(26, 128240)
    rir = torch.randn(26, 4, 2048)
    batch_data = {}
    batch_data['waveform'] = wav.cuda().contiguous()  # to cuda
    batch_data['noise'] = noise.cuda().contiguous()
    batch_data['rir'] = rir.cuda().contiguous()

    return batch_data


def test_add_rir(counter=10):
    '''test addRIR and GetRirData'''
    add_rir = AddRIR()
    print("test AddRiR.addnoise")
    iter_ = 0
    for noise_data in add_rir.get_noise():
        assert noise_data is not None
        iter_ += 1
        if iter_ > counter:
            break
    iter_ = 0
    for rir, rir_direction in add_rir.get_rir():
        assert rir is not None
        assert rir_direction is not None
        iter_ += 1
        if iter_ > counter:
            break
    get_rir_data = GetRirData()
    for _ in range(counter):
        noise_data = get_rir_data.get_noise_data(452310)
        assert noise_data is not None

    for _ in range(counter):
        rir, rir_direction = get_rir_data.get_rir_data()
        assert rir is not None
        assert rir_direction is not None
