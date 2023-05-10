'''
test streaming rnnt.
'''
import copy
import torch

from core.utils import Config
from core.extensions import AmpEnable
from core.solutions import setup_solution

from core.models.layers.unfold import PantherUnFold


def build_solution():
    '''build solution'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.manual_seed(121738)
    # setup solution, and get onnx file
    config = 'configs/asr/dfsmn_rnnt.py'
    cfg = Config.fromfile(config)
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('backend', 'panther')
    solution_cfg.setdefault('fbank_dim', 80)
    solution_cfg['onnx_stack_frame'] = 80
    solution_cfg['onnx_with_mask'] = False
    solution_cfg['backbone_mask'] = False
    tgt_vocab_size = (
        solution_cfg.adaptive_head_size
        + solution_cfg.adaptive_tail_size * solution_cfg.adaptive_tail_groups
    )
    solution_cfg.setdefault('tgt_vocab_size', tgt_vocab_size)
    solution_cfg.setdefault('encoder_convert_stream', True)
    solution_cfg.setdefault('predictor_convert_stream', True)

    # train_cfg = cfg.train
    # onnx_dir = os.path.join(train_cfg.save_root, train_cfg.save_dir, train_cfg.save_name, 'onnx')
    # solution_cfg.setdefault('onnx_dir', onnx_dir)
    # inference are registered when solution init.
    solution = setup_solution(solution_cfg)
    # checkpoint_hdfs = 'step_240000.pth'
    # checkpoint = load_checkpoint(solution, checkpoint_hdfs, map_location='cpu')
    solution.cuda()
    solution.eval()
    return solution


def test_vgg_frontend():
    solution = build_solution()
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, feat_dim, slice_size = 5, 1000, 80, 20
        config = Config.fromfile('configs/asr/dfsmn_rnnt.py')
        assert (config.solution.front_end_type == 'VGGFrontEnd',)
        x = torch.rand([bsz, times, feat_dim]).cuda()
        # x_mask = torch.ones([bsz, times]).cuda().int().cuda()
        model = solution.acoustic_front_end_module
        nonstream_y, _, _ = model(x, None)

        required_right_context = model.fsmn_right_kernel_size * model.downsampling_size
        states = torch.zeros([bsz, model.state_size]).cuda()

        # states = torch.zeros([bsz, model.state_size()]).cuda()
        stream_y = []

        # firtst
        start, end, x_sign = 0, slice_size + required_right_context, 1
        x_sign = torch.tensor([x_sign]).int().cuda()
        xi = x[:, start:end, :]
        yi, states, _, _ = model.forward_step(xi, x_mask=None, x_sign=x_sign, states=states)
        stream_y.append(yi)
        # mid
        for i in range(slice_size + required_right_context, times - slice_size, slice_size):
            start, end, x_sign = i, i + slice_size, 0
            xi = x[:, start:end, :]
            yi, states, _, _ = model.forward_step(xi, x_mask=None, x_sign=x_sign, states=states)
            stream_y.append(yi)
        # last
        start, x_sign = i + slice_size, 2
        xi = x[:, start:, :]
        yi, states, _, _ = model.forward_step(xi, x_mask=None, x_sign=x_sign, states=states)
        stream_y.append(yi)
        stream_y = torch.cat(stream_y, dim=2)
        assert torch.allclose(stream_y, nonstream_y, rtol=1e-5, atol=1e-5)


def test_dfsmn_layer():
    '''test dfsmn'''
    solution = build_solution()
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, feat_dim, slice_size = 5, 100, 512, 20
        x = torch.rand([bsz, times, feat_dim]).cuda()
        x_mask = torch.ones([bsz, times]).cuda().int().cuda()
        model = solution.acoustic_backbone_module.dfsmn_layers[0]

        nonstream_y = model(x, x_mask)

        required_right_context = model.right_kernel_size
        states = torch.zeros([bsz, model.state_size]).cuda()
        stream_y = []
        # firtst
        start, end, x_sign = 0, slice_size + required_right_context, 1
        x_sign = torch.tensor([x_sign]).int().cuda()
        xi = x[:, start:end, :]
        yi, states, _ = model.forward_step(
            xi.transpose(1, 2), x_mask=None, x_sign=x_sign, states=states
        )
        stream_y.append(yi)
        # mid
        for i in range(slice_size + required_right_context, times - slice_size, slice_size):
            start, end, x_sign = i, i + slice_size, 0
            xi = x[:, start:end, :]
            yi, states, _ = model.forward_step(
                xi.transpose(1, 2), x_mask=None, x_sign=x_sign, states=states
            )
            stream_y.append(yi)
        # last
        start, x_sign = i + slice_size, 2
        xi = x[:, start:, :]
        yi, states, _ = model.forward_step(
            xi.transpose(1, 2), x_mask=None, x_sign=x_sign, states=states
        )

        stream_y.append(yi)
        stream_y = torch.cat(stream_y, dim=2)
        assert torch.allclose(stream_y, nonstream_y.transpose(1, 2), rtol=1e-5, atol=1e-5)


def test_dfsmn_backbone():
    '''test dfsmn'''
    solution = build_solution()
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, feat_dim, slice_size = 5, 100, 512, 20
        x = torch.rand([bsz, times, feat_dim]).cuda()
        x_mask = torch.ones([bsz, times]).cuda().int().cuda()
        model = solution.acoustic_backbone_module
        nonstream_y = model(x, x_mask)

        required_right_context = sum(n.right_kernel_size for n in model.dfsmn_layers)
        states = torch.zeros([bsz, model.state_size]).cuda()
        stream_y = []
        # firtst
        start, end, x_sign = 0, slice_size + required_right_context, 1
        x_sign = torch.tensor([x_sign]).int().cuda()
        xi = x[:, start:end, :]

        yi, states = model.forward_step(xi, x_mask=None, x_sign=x_sign, states=states)
        stream_y.append(yi)
        # mid
        for i in range(slice_size + required_right_context, times - slice_size, slice_size):
            start, end, x_sign = i, i + slice_size, 0
            xi = x[:, start:end, :]
            yi, states = model.forward_step(xi, x_mask=None, x_sign=x_sign, states=states)
            stream_y.append(yi)
        # last
        start, x_sign = i + slice_size, 2
        xi = x[:, start:, :]
        yi, states = model.forward_step(xi, x_mask=None, x_sign=x_sign, states=states)

        stream_y.append(yi)
        stream_y = torch.cat(stream_y, dim=1)
        assert torch.allclose(stream_y, nonstream_y, rtol=1e-5, atol=2e-5)


def test_dfsmn_memory():
    '''test dfsmn ln'''
    backbone_memory_size = 512
    left_kernel_size = 40
    right_kernel_size = 1
    dilation = 1

    conv_memory = torch.nn.Conv1d(
        backbone_memory_size,
        backbone_memory_size,
        kernel_size=left_kernel_size + right_kernel_size + 1,
        padding=0,
        stride=1,
        dilation=dilation,
        groups=backbone_memory_size,
        bias=False,
    ).cuda()

    unfold_memory = PantherUnFold(
        kernel_size=(left_kernel_size + right_kernel_size + 1, 1),
        padding=0,
        stride=1,
        dilation=dilation,
    ).cuda()
    unfold_memory_coffe = (
        conv_memory.weight.data.clone().detach().permute(2, 1, 0).unsqueeze(0).contiguous()
    )
    unfold_memory_coffe = torch.nn.Parameter(unfold_memory_coffe)

    bsz = 16
    seq_len = 210
    x0 = torch.randn(bsz, seq_len, backbone_memory_size).cuda().requires_grad_()
    x1 = x0.clone().detach().requires_grad_()

    x0_pad = torch.nn.functional.pad(
        x0,
        (0, 0, left_kernel_size * dilation, right_kernel_size * dilation),
    )
    x1_pad = torch.nn.functional.pad(
        x1,
        (0, 0, left_kernel_size * dilation, right_kernel_size * dilation),
    )

    y0 = conv_memory(x0_pad.transpose(1, 2).contiguous()).transpose(1, 2)
    y1 = (
        unfold_memory(x1_pad.unsqueeze(1)).contiguous().view(bsz, -1, seq_len, backbone_memory_size)
    )
    y1 = (y1 * unfold_memory_coffe).sum(dim=1)

    assert torch.allclose(y0, y1, rtol=1e-5, atol=1e-6), torch.max(torch.abs(y0 - y1))


if __name__ == '__main__':
    # test()
    test_vgg_frontend()
    test_dfsmn_layer()
    test_dfsmn_backbone()
    test_dfsmn_memory()
