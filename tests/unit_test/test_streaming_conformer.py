'''
test streaming rnnt.
'''
import torch
import panther
from core.utils import FalconDict, Config
from core.extensions import AmpEnable

from core.solutions.inference.infer import convert_to_np
from core.solutions.inference.infer import convert_to_tensor

# pylint: disable=unused-import
from core.models.asr.acoustic_frontend import Conv2dPooling, Conv2dPooling4
from core.models.asr.acoustic_backbone import (
    MaskedConformerBackbone,
    ConformerBackbone,
)
from core.models.asr.acoustic_head import *


class MaskStreamConformerEncoderExporter(torch.nn.Module):
    '''Conformer exporter test.'''

    def __init__(self, frontend, backbone, head, downsampling_size):
        super().__init__()
        self.acoustic_front_end_module = frontend
        self.acoustic_backbone_module = backbone
        self.acoustic_head_module = head
        self.downsampling_size = downsampling_size
        self.required_right = (
            sum(
                m.self_attn.right_kernel_size + m.conv_module.right_kernel_size
                for m in self.acoustic_backbone_module.encoders
            )
            * self.downsampling_size
        )

    def forward_(self, fbank, fbank_mask):
        '''forward.'''
        front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
        encoder_backbone_out = self.backbone(front_end_out, backbone_mask, frontend_shape)
        encoder_backbone_out = self.head(encoder_backbone_out)
        return encoder_backbone_out, front_end_out

    def split_cache_list(self, bsz, cache_list):
        '''split Tensor to list.'''
        if isinstance(cache_list, (list, tuple)):
            return cache_list
        new_list, offset = [], 0
        for m in self.backbone.encoders:
            attn_left_kernel_size = m.self_attn.left_kernel_size
            conv_left_kernel_size = m.conv_module.left_kernel_size
            feat_dim = m.attention_dim
            m_list = []
            # 0
            size = attn_left_kernel_size * feat_dim
            m_list.append(
                cache_list[:, offset : offset + size].view(bsz, attn_left_kernel_size, feat_dim)
            )
            offset += size
            # 1
            size = attn_left_kernel_size
            m_list.append(cache_list[:, offset : offset + size].view(bsz, 1, attn_left_kernel_size))
            offset += size
            # 2
            size = conv_left_kernel_size * feat_dim
            m_list.append(
                cache_list[:, offset : offset + size].view(bsz, conv_left_kernel_size, feat_dim)
            )
            offset += size

            new_list.append(m_list)
        return new_list

    def forward_step_(self, x, mask, states, cache_list):
        '''forward step.'''
        bsz, times, _ = x.size()
        assert times % self.downsampling_size == 0
        cache_list = self.split_cache_list(bsz, cache_list)

        required_right_context = torch.Tensor([self.required_right]).long().cuda()
        x_sign = torch.Tensor([0]).int().cuda()
        x_out, states, _, _ = self.frontend.forward_step(
            x, states=states, x_sign=x_sign, required_right_context=required_right_context
        )
        mask = mask.view(bsz, times // self.downsampling_size, self.downsampling_size)[
            :, :, 0
        ].float()  # downsample
        required_right_context = required_right_context // self.downsampling_size
        x_out, _, cache_list, mask = self.backbone.forward_step(
            x_out, required_right_context, cache_list, mask
        )
        # cache_list = concat_global_states(cache_list, reshape=True, bsz=bsz)
        x_out = self.head(x_out)
        return x_out, mask, states, cache_list

    def forward(self, x, mask, states, cache_list):
        '''forward for export'''
        return self.forward_step_(x, mask, states, cache_list)

    @property
    def head(self):
        '''head'''
        return self.acoustic_head_module

    @property
    def frontend(self):
        '''frontend'''
        return self.acoustic_front_end_module

    @property
    def backbone(self):
        '''backbone'''
        return self.acoustic_backbone_module


class MaskStreamConformerExporter(torch.nn.Module):
    '''Conformer exporter test.'''

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, required_right_context, cache_list):
        '''forward.'''
        x_out, required_right_context, cache_list, mask = self.model.forward_step(
            x, required_right_context, cache_list, mask
        )
        return x_out, required_right_context, cache_list, mask


def test_stream_conv2d_pooling():
    '''test conv2d pooling.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.manual_seed(121738)
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, fbank_dim, chunk = 5, 1000, 80, 20
        # bsz, times, fbank_dim, chunk = 1, 12, 80, 4
        args = FalconDict(backbone_memory_size=512, fbank_dim=fbank_dim)
        right_view = 0

        x = torch.rand([bsz, times, fbank_dim]).cuda()
        model = Conv2dPooling(args).cuda()

        non_stream_y, _, _ = model(x)

        states = torch.zeros([bsz, model.state_size]).cuda()
        stream_y = []
        for i in range(0, times, chunk):
            start, end, x_sign = i, i + chunk + right_view, 0
            if start <= 0 and end >= times:
                x_sign = 3
            elif start <= 0:
                x_sign = 1
            elif end >= times:
                x_sign = 2
            x_sign = torch.tensor([x_sign]).int().cuda()
            xi = x[:, start:end, :]
            yi, states, _, _ = model.forward_step(xi, states=states, x_sign=x_sign)
            stream_y.append(yi)
        stream_y = torch.cat(stream_y, dim=1)
        assert torch.allclose(non_stream_y, stream_y, atol=1e-5)


def test_conformer():
    '''test conformer'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.manual_seed(121738)
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, feat_dim, slice_size = 5, 1000, 512, 20
        config = Config.fromfile('configs/asr/streaming_conformer_rnnt_input_10k.py')
        assert config.solution.acoustic_backbone_type in (
            'MaskedConformerBackbone',
            'ConformerBackbone',
        )

        x = torch.rand([bsz, times, feat_dim]).cuda()
        # x_mask = torch.ones([bsz, 1, times]).cuda().int().cuda()
        x_mask = torch.ones([bsz, 1, times], dtype=torch.int).cuda()

        model = eval(config.solution.acoustic_backbone_type)(config.solution).cuda()
        model = model.eval()

        required_right_context = sum(
            m.self_attn.right_kernel_size + m.conv_module.right_kernel_size for m in model.encoders
        )

        x_out = model.pos_enc(x)
        for m in model.encoders:
            x_out, x_mask = m([x_out, x_mask])

        stream_x_out_list = []

        cache_list = [
            [
                torch.zeros([bsz, m.self_attn.left_kernel_size, feat_dim]).type_as(x),
                torch.zeros([bsz, 1, m.self_attn.left_kernel_size]).type_as(x),
                torch.zeros([bsz, m.conv_module.left_kernel_size, feat_dim]).type_as(x),
            ]
            for m in model.encoders
        ]

        for i in range(0, times, slice_size):
            start, end = i, min(times, i + slice_size + required_right_context)
            stream_x = x[:, start:end, :]
            stream_x_mask = x_mask[:, :, start:end]
            stream_x_out = model.pos_enc.forward_step(stream_x)
            if isinstance(stream_x_out, tuple):
                input_tensor, pos_emb_tensor = stream_x_out
                attn_history_size = model.encoders[0].self_attn.left_kernel_size
                pe_max_len = int((pos_emb_tensor.size()[1] + 1) / 2)
                key_size = input_tensor.size()[1] + attn_history_size
                pos_emb_tensor = pos_emb_tensor[
                    :, pe_max_len - key_size : pe_max_len + key_size - 1
                ]
                stream_x_out = input_tensor, pos_emb_tensor
            for j, m in enumerate(model.encoders):
                att_cache, att_mask_cache, conv_cache = cache_list[j]
                stream_x_out, stream_x_mask, att_cache, att_mask_cache, conv_cache = m.forward_step(
                    [stream_x_out, stream_x_mask],
                    required_right_context,
                    att_cache,
                    att_mask_cache,
                    conv_cache,
                )
                cache_list[j] = [att_cache, att_mask_cache, conv_cache]
            stream_x_slice_out = stream_x_out[0][:, :slice_size, :]
            x_slice_out = x_out[0][:, i : i + slice_size, :]
            stream_x_out_list.append(stream_x_slice_out)
            print(
                "slice chunk allclose: {}, abs diff sum {}".format(
                    torch.allclose(
                        stream_x_slice_out, x_out[0][:, i : i + slice_size, :], rtol=1e-5, atol=1e-5
                    ),
                    (stream_x_slice_out - x_out[0][:, i : i + slice_size, :]).abs().sum(),
                )
            )
            assert torch.allclose(stream_x_slice_out, x_slice_out, rtol=1e-5, atol=1e-5)
        stream_x_out = torch.cat(stream_x_out_list, dim=1)
        x_out = x_out[0]

        print(
            "total sequence allclose: {}, abs diff sum {}".format(
                torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5),
                (stream_x_out - x_out).abs().sum(),
            )
        )
        assert torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5)


def test_conformer_backbone():
    '''test conformer layer'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.manual_seed(121738)
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, feat_dim, slice_size = 5, 1000, 512, 20
        config = Config.fromfile('configs/asr/streaming_conformer_rnnt_input_10k.py')
        assert config.solution.acoustic_backbone_type in (
            'MaskedConformerBackbone',
            'ConformerBackbone',
        )

        x = torch.rand([bsz, times, feat_dim]).cuda()
        x_mask = torch.ones([bsz, times]).cuda().int().cuda()

        model = eval(config.solution.acoustic_backbone_type)(config.solution).cuda()
        model = model.eval()

        required_right_context = sum(
            m.self_attn.right_kernel_size + m.conv_module.right_kernel_size for m in model.encoders
        )
        x_out = model(x, x_mask)

        stream_x_out_list = []
        cache_list = [
            [
                torch.zeros([bsz, m.self_attn.left_kernel_size, feat_dim]).type_as(x),
                torch.zeros([bsz, 1, m.self_attn.left_kernel_size]).type_as(x),
                torch.zeros([bsz, m.conv_module.left_kernel_size, feat_dim]).type_as(x),
            ]
            for m in model.encoders
        ]

        stream_x_out_list = []
        for i in range(0, times, slice_size):
            start, end = i, min(times, i + slice_size + required_right_context)
            stream_x = x[:, start:end, :]
            stream_x_mask = x_mask[:, start:end]

            stream_x_out, required_right_context, cache_list, stream_x_mask = model.forward_step(
                stream_x, required_right_context, cache_list, stream_x_mask
            )

            stream_x_slice_out = stream_x_out[:, :slice_size, :]
            x_slice_out = x_out[:, i : i + slice_size, :]
            stream_x_out_list.append(stream_x_slice_out)
            print(
                "slice chunk allclose: {}, abs diff sum {}".format(
                    torch.allclose(
                        stream_x_slice_out, x_out[:, i : i + slice_size, :], rtol=1e-5, atol=1e-5
                    ),
                    (stream_x_slice_out - x_out[:, i : i + slice_size, :]).abs().sum(),
                )
            )
            assert torch.allclose(stream_x_slice_out, x_slice_out, rtol=1e-5, atol=1e-5)
        stream_x_out = torch.cat(stream_x_out_list, dim=1)

        print(
            "total sequence allclose: {}, abs diff sum {}".format(
                torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5),
                (stream_x_out - x_out).abs().sum(),
            )
        )
        assert torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5)


def test_conformer_encoder():
    '''test conformer encoder with frontend'''
    # pylint:disable=too-many-locals
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.manual_seed(121738)
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, fbank_dim, feat_dim, slice_size = 5, 1000, 80, 512, 20
        downsample = 4
        config = Config.fromfile('configs/asr/streaming_conformer_rnnt_input_10k.py')
        # config = Config.fromfile('streaming_conformer_rnnt_finetune_input10k_wav.py')
        assert config.solution.acoustic_backbone_type in (
            'MaskedConformerBackbone',
            'ConformerBackbone',
        )

        x = torch.rand([bsz, times, fbank_dim]).cuda()
        x_mask = torch.ones([bsz, times]).cuda().int().cuda()
        config.solution.setdefault('fbank_dim', config.data.get("fbank_dim"))
        frontend = eval(config.solution.front_end_type)(config.solution).cuda()
        backbone = eval(config.solution.acoustic_backbone_type)(config.solution).cuda()
        head = eval(config.solution.head_type)(config.solution).cuda()
        model = MaskStreamConformerEncoderExporter(frontend, backbone, head, downsample)
        # checkpoint_hdfs = "step_160000.pth"
        # checkpoint = load_checkpoint(model, checkpoint_hdfs, map_location='cpu')
        model = model.eval()

        x_out, _ = model.forward_(x, x_mask)

        stream_x_out_list = []

        frontend_states = torch.zeros([bsz, model.frontend.state_size]).cuda()
        required_right_context = sum(
            m.self_attn.right_kernel_size + m.conv_module.right_kernel_size
            for m in model.backbone.encoders
        )

        cache_list = [
            [
                torch.zeros([bsz, m.self_attn.left_kernel_size, feat_dim]).type_as(x),
                torch.zeros([bsz, 1, m.self_attn.left_kernel_size]).type_as(x),
                torch.zeros([bsz, m.conv_module.left_kernel_size, feat_dim]).type_as(x),
            ]
            for m in model.backbone.encoders
        ]

        stream_x_out_list = []
        for i in range(0, times, slice_size):
            assert slice_size % downsample == 0
            start, end = i, min(times, i + slice_size + required_right_context * downsample)
            stream_x = x[:, start:end, :]
            stream_x_mask = x_mask[:, start:end]
            if start <= 0 and end >= times:
                x_sign = 3
            elif start <= 0:
                x_sign = 1
            elif end >= times:
                x_sign = 2
            else:
                x_sign = 0
            x_sign = torch.tensor([x_sign]).int().cuda()
            b, frames, h = stream_x.size()
            if x_sign == 2:
                padding = slice_size + model.required_right - frames
                pad_x = torch.zeros([b, padding, h], dtype=stream_x.dtype, device=stream_x.device)
                pad_mask = torch.zeros(
                    [b, padding], dtype=stream_x_mask.dtype, device=stream_x_mask.device
                )
                stream_x = torch.cat([stream_x, pad_x], dim=1)
                stream_x_mask = torch.cat([stream_x_mask, pad_mask], dim=1)
            stream_x_out, stream_x_mask, frontend_states, cache_list = model(
                stream_x,
                stream_x_mask,
                frontend_states,
                cache_list,
            )

            stream_x_slice_out = stream_x_out[:, : slice_size // downsample, :]
            x_slice_out = x_out[:, i // downsample : (i + slice_size) // downsample, :]
            stream_x_out_list.append(stream_x_slice_out)
            print(
                "slice chunk allclose: {}, abs diff sum {}".format(
                    torch.allclose(stream_x_slice_out, x_slice_out, rtol=1e-5, atol=1e-5),
                    (stream_x_slice_out - x_slice_out).abs().sum(),
                )
            )
            assert torch.allclose(stream_x_slice_out, x_slice_out, rtol=1e-5, atol=1e-5)
        stream_x_out = torch.cat(stream_x_out_list, dim=1)

        print(
            "total sequence allclose: {}, abs diff sum {}".format(
                torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5),
                (stream_x_out - x_out).abs().sum(),
            )
        )
        assert torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5)


def export_stream_conformer():
    '''export conformer backbone'''
    config = Config.fromfile('configs/asr/streaming_conformer_rnnt_input_10k.py')
    assert config.solution.acoustic_backbone_type in (
        'MaskedConformerBackbone',
        'ConformerBackbone',
    )
    model = eval(config.solution.acoustic_backbone_type)(config.solution).cuda()
    model = MaskStreamConformerExporter(model)
    model = model.eval()

    bsz, times, feat_dim = 1, 1, 512
    x = torch.rand([bsz, times, feat_dim]).cuda()
    x_mask = torch.ones([bsz, times]).cuda().int().cuda()

    required_right_context = sum(
        m.self_attn.right_kernel_size + m.conv_module.right_kernel_size
        for m in model.model.encoders
    )
    required_right_context = torch.Tensor([required_right_context]).int().cuda()

    cache_list = [
        [
            torch.zeros([bsz, m.self_attn.left_kernel_size, feat_dim]).type_as(x),
            torch.zeros([bsz, 1, m.self_attn.left_kernel_size]).type_as(x),
            torch.zeros([bsz, m.conv_module.left_kernel_size, feat_dim]).type_as(x),
        ]
        for m in model.model.encoders
    ]
    inputs = (x, x_mask, required_right_context, cache_list)
    input_names = {'x': {0: 'B', 1: 'T'}}
    output_names = {'x_out': {0: 'B', 1: 'T'}}
    dynamic_axes = input_names.copy()
    dynamic_axes.update(output_names)
    input_names = list(input_names.keys())
    output_names = list(output_names.keys())

    torch.onnx.export(
        model,
        inputs,
        'out.onnx',
        verbose=True,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        enable_onnx_checker=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def export_stream_conformer_encoder():
    '''export conformer encoder'''
    # pylint:disable=too-many-locals
    config = Config.fromfile('configs/asr/streaming_conformer_rnnt_input_10k.py')
    assert config.solution.acoustic_backbone_type in (
        'MaskedConformerBackbone',
        'ConformerBackbone',
    )
    config.solution.setdefault('fbank_dim', config.data.get("fbank_dim"))
    frontend = eval(config.solution.front_end_type)(config.solution).cuda()
    backbone = eval(config.solution.acoustic_backbone_type)(config.solution).cuda()
    head = eval(config.solution.head_type)(config.solution).cuda()
    downsample = 4

    model = MaskStreamConformerEncoderExporter(frontend, backbone, head, downsample)
    # checkpoint_hdfs = 'step_100000.pth'
    # load_checkpoint(model, checkpoint_hdfs, map_location='cpu')
    model = model.eval()

    slice_size = 20
    bsz, times, fbank_dim, feat_dim = 1, slice_size + model.required_right, 80, 512
    x = torch.rand([bsz, times, fbank_dim]).cuda()
    x_mask = torch.ones([bsz, times]).cuda().int().cuda()

    frontend_states = torch.zeros([bsz, model.frontend.state_size]).cuda()
    cache_list = [
        [
            torch.zeros([bsz, m.self_attn.left_kernel_size, feat_dim]).type_as(x),
            torch.zeros([bsz, 1, m.self_attn.left_kernel_size]).type_as(x),
            torch.zeros([bsz, m.conv_module.left_kernel_size, feat_dim]).type_as(x),
        ]
        for m in model.backbone.encoders
    ]
    # cache_list = concat_global_states(cache_list, reshape=True, bsz=bsz)
    inputs = (x, x_mask, frontend_states, cache_list)
    input_names = {
        'x': {0: 'B', 1: 'T'},
        'x_mask': {0: 'B', 1: 'T'},
        'frontend_states': {0: 'B'},
        # 'cache_list': {0: 'B'},
    }
    output_names = {
        'x_out': {0: 'B', 1: 'T'},
        'mask': {0: 'B', 1: 'T'},
        'states': {0: 'B'},
        # 'cache_list': {0: 'B'},
    }
    dynamic_axes = input_names.copy()
    dynamic_axes.update(output_names)
    input_names = list(input_names.keys())
    output_names = list(output_names.keys())

    torch.onnx.export(
        model,
        inputs,
        'out.onnx',
        verbose=True,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        enable_onnx_checker=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    # non-streaming test
    panther_inference = load_model("out.onnx", ['CUDAExecutionProvider'])
    input_data = convert_to_np(get_onnx_input(inputs))
    input_name = []
    for data in panther_inference.get_inputs():
        input_name.append(data.name)
    output_name = []
    for data in panther_inference.get_outputs():
        output_name.append(data.name)

    onnx_input = dict(zip(input_name, input_data))
    onnx_output = dict(zip(output_name, convert_to_tensor(panther_inference.run(None, onnx_input))))

    torch_output = model(x, x_mask, frontend_states, cache_list)
    print(
        "non-streaming test total sequence allclose: {}, abs diff sum {}".format(
            torch.allclose(onnx_output["x_out"], torch_output[0], rtol=1e-5, atol=1e-5),
            (onnx_output["x_out"] - torch_output[0]).abs().sum(),
        )
    )
    assert torch.allclose(onnx_output["x_out"], torch_output[0], rtol=1e-5, atol=1e-5)
    # streaming test
    bsz, times, fbank_dim, feat_dim, slice_size = 1, 1000, 80, 512, 20
    x = torch.rand([bsz, times, fbank_dim]).cuda()
    x_mask = torch.ones([bsz, times]).cuda().int().cuda()
    stream_x_out_list = []
    frontend_states = torch.zeros([bsz, model.frontend.state_size]).cuda()
    cache_list = [
        [
            torch.zeros([bsz, m.self_attn.left_kernel_size, feat_dim]).type_as(x),
            torch.zeros([bsz, 1, m.self_attn.left_kernel_size]).type_as(x),
            torch.zeros([bsz, m.conv_module.left_kernel_size, feat_dim]).type_as(x),
        ]
        for m in model.backbone.encoders
    ]
    # cache_list = concat_global_states(cache_list, reshape=True, bsz=bsz)
    x_out, _ = model.forward_(x, x_mask)
    for i in range(0, times, slice_size):
        assert slice_size % downsample == 0
        start, end = i, min(times, i + slice_size + model.required_right)
        stream_x = x[:, start:end, :]
        stream_x_mask = x_mask[:, start:end]
        if start <= 0 and end >= times:
            x_sign = 3
        elif start <= 0:
            x_sign = 1
        elif end >= times:
            x_sign = 2
        else:
            x_sign = 0
        b, frames, h = stream_x.size()
        if x_sign == 2:
            padding = slice_size + model.required_right - frames
            pad_x = torch.zeros([b, padding, h], dtype=stream_x.dtype, device=stream_x.device)
            pad_mask = torch.zeros(
                [b, padding], dtype=stream_x_mask.dtype, device=stream_x_mask.device
            )
            stream_x = torch.cat([stream_x, pad_x], dim=1)
            stream_x_mask = torch.cat([stream_x_mask, pad_mask], dim=1)
        inputs = (stream_x, stream_x_mask, frontend_states, cache_list)
        input_data = convert_to_np(get_onnx_input(inputs))
        onnx_input = dict(zip(input_name, input_data))
        onnx_output = panther_inference.run(None, onnx_input)
        onnx_output = dict(zip(output_name, convert_to_tensor(onnx_output)))
        stream_x_out, stream_x_mask, frontend_states, cache_list = model(
            stream_x,
            stream_x_mask,
            frontend_states,
            cache_list,
        )
        stream_x_slice_out = stream_x_out[:, : slice_size // downsample, :]
        x_slice_out = onnx_output["x_out"][:, : slice_size // downsample, :]
        stream_x_out_list.append(stream_x_slice_out)
        print(
            "streaming test slice chunk allclose: {}, abs diff sum {}".format(
                torch.allclose(stream_x_slice_out, x_slice_out, rtol=1e-5, atol=1e-5),
                (stream_x_slice_out - x_slice_out).abs().sum(),
            )
        )
        assert torch.allclose(stream_x_slice_out, x_slice_out, rtol=1e-5, atol=1e-5)

    stream_x_out = torch.cat(stream_x_out_list, dim=1)
    print(
        "streaming test total sequence allclose: {}, abs diff sum {}".format(
            torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5),
            (stream_x_out - x_out).abs().sum(),
        )
    )
    assert torch.allclose(stream_x_out, x_out, rtol=1e-5, atol=1e-5)


def load_model(model_filepath, providers):
    '''load onnx'''
    infer_option = panther.SessionOptions()
    infer_option.intra_op_num_threads = 1
    infer_option.graph_optimization_level = panther.GraphOptimizationLevel.PTH_ENABLE_ALL
    return panther.InferenceSession(model_filepath, providers=providers, sess_options=infer_option)


def get_onnx_input(data):
    '''get onnx input'''
    onnx_input = []
    if isinstance(data, (list, tuple)):
        for x in data:
            onnx_input += get_onnx_input(x)
    else:
        onnx_input.append(data)
    return onnx_input


# if __name__ == '__main__':
# test_conformer_backbone()
# test_stream_conv2d_pooling()
# test_conformer_backbone()
# test_conformer_encoder()
# export_stream_conformer()
# export_stream_conformer_encoder()
