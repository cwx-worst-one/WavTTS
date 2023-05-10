'''
acoustic frontend.
'''
import torch
from torch import nn
import torch.nn.functional as F

try:
    from byteslim.quant.torch.utils import QunatizeFuncProducer
except ImportError:
    QunatizeFuncProducer = None

# pylint: disable=no-name-in-module
# pylint: disable=too-many-branches
from core.models.layers.cnn import SamePaddingConv2D
from core.models.layers.cnn import MultiplicativeUnit
from core.models.layers.embedding import AbsPositionalEncoding
from core.models.layers.lstmp_layer import LSTMP
from core.models.layers.time_reduce_layer import *
from core.models.utils import streaming_conv2d_input
from core.utils.dict import FalconDict
from core.utils.math import ceil


class StackingFrameFrontEnd(nn.Module):
    '''StackingFrameFrontEnd.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.kernel_size = args.stacking_frontend_kernel_size
        self.stride = args.stacking_frontend_stride
        self.dilation = args.stacking_frontend_dilation
        self.padding = args.stacking_frontend_padding

        self.unfolder = nn.Unfold(
            kernel_size=(self.kernel_size, 1),
            dilation=(self.dilation, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1),
        )

    def forward(self, feat, feat_mask):
        '''forward.'''
        if not (
            self.kernel_size == 1 and self.stride == 1 and self.dilation == 1 and self.padding == 0
        ):
            feat, feat_mask = self._unfold(feat, feat_mask)
        return feat, feat_mask, 'BTN'

    def _unfold(self, feat, feat_mask):
        """A helper function for unfold fbank"""

        B, T, D = feat.shape
        assert feat_mask.shape[1] == T
        assert feat_mask.shape[0] == B

        unfold_feat = (
            self.unfolder(feat.unsqueeze(1))
            .reshape(B, self.kernel_size, -1, D)
            .transpose(1, 2)
            .reshape(B, -1, self.kernel_size * D)
        )
        unfold_feat_mask = (
            self.unfolder(feat_mask.unsqueeze(1).unsqueeze(3))
            .reshape(B, self.kernel_size, -1, 1)
            .transpose(1, 2)
            .reshape(B, -1, self.kernel_size)
            .sum(-1)
            > 0
        ).float()
        # XOR, if a stack of frames have one mask, all these frames are seen as masked.
        return unfold_feat, unfold_feat_mask


class FSMNModule(nn.Module):
    '''FSMNModule.'''

    def __init__(
        self, fsmn_backbone_memory_size, fsmn_left_kernel_size, fsmn_right_kernel_size, dilation=1
    ):
        '''init.'''
        super().__init__()
        kernel_size = fsmn_left_kernel_size + fsmn_right_kernel_size + 1
        self.memory = nn.Conv1d(
            fsmn_backbone_memory_size,
            fsmn_backbone_memory_size,
            kernel_size=kernel_size,
            padding=0,
            stride=1,
            dilation=dilation,
            groups=fsmn_backbone_memory_size,
        )
        self.fsmn_left_kernel_size = fsmn_left_kernel_size
        self.fsmn_right_kernel_size = fsmn_right_kernel_size
        self.dilation = dilation

    def forward(self, input_feat):
        '''forward.'''
        pad_input_feat = F.pad(
            input_feat,
            (
                self.fsmn_left_kernel_size * self.dilation,
                self.fsmn_right_kernel_size * self.dilation,
                0,
                0,
            ),
        )  # (B,N,T+(l+r)*d)
        memory_out = self.memory(pad_input_feat)
        return memory_out


class FSMNFrontEnd(nn.Module):
    '''FSMNFrontEnd.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        padding = args.input_concat_size // 2 - (1 - args.input_concat_size % 2)
        self.input_trans_fc = nn.Sequential(
            *[
                nn.Conv1d(
                    args.fbank_dim,
                    args.backbone_hidden_size,
                    args.input_concat_size,
                    stride=args.downsampling_size,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(args.backbone_hidden_size, args.backbone_memory_size, 1, bias=False),
                FSMNModule(
                    args.backbone_memory_size,
                    args.fsmn_left_kernel_size,
                    args.fsmn_right_kernel_size,
                    dilation=args.fsmn_dilation,
                ),
            ]
        )
        self.downsampling_size = args.downsampling_size

    def forward(self, fbank_feat, fbank_mask):
        '''forward.'''
        (bsz, frame_size, _) = fbank_feat.size()
        assert (
            frame_size % self.downsampling_size == 0
        ), "plz padding the fbank to 12,        which is suitable for downsamping size 3/4"
        fsmn_mask = fbank_mask.view(
            bsz, frame_size // self.downsampling_size, self.downsampling_size
        )[
            :, :, 0
        ]  # (B, T)
        fsmn_out = self.input_trans_fc(fbank_feat.transpose(1, 2).contiguous())  # (B, N, T)
        return fsmn_out * fsmn_mask.unsqueeze(1), fsmn_mask, "BNT"


class VGGFrontEnd(nn.Module):
    '''VGGFrontEnd.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args
        self.idim = args.fbank_dim
        self.conv0_out_ch = args.get('front_end_conv0_ch', 128)
        self.conv1_out_ch = args.get('front_end_conv1_ch', 128)
        self.frontend_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.conv0_out_ch, 3, 2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.conv0_out_ch, self.conv1_out_ch, 3, 2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_proj_fc = nn.Linear(
            self.conv1_out_ch * args.fbank_dim // 4, args.backbone_memory_size
        )

        self.input_trans_fc = nn.Sequential(
            *[
                nn.Conv1d(args.backbone_memory_size, args.backbone_hidden_size, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(args.backbone_hidden_size, args.backbone_memory_size, 1, bias=False),
                FSMNModule(
                    args.backbone_memory_size,
                    args.fsmn_left_kernel_size,
                    args.fsmn_right_kernel_size,
                    dilation=args.fsmn_dilation,
                ),
            ]
        )
        self.downsampling_size = args.downsampling_size
        self.fsmn_left_kernel_size = args.fsmn_left_kernel_size
        self.fsmn_right_kernel_size = args.fsmn_right_kernel_size
        self.memory_size = args.backbone_memory_size
        self.fsmn_dilation = args.fsmn_dilation
        assert self.downsampling_size == 4

    @property
    def state_size(self):
        '''streamed state size.'''
        conv0_state_size = 1 * ((3 - 1) * 1 + 1 - 2) * self.idim
        conv1_state_size = self.conv0_out_ch * ((3 - 1) * 1 + 1 - 2) * self.idim // 2
        if (
            self.frontend_conv[0].__class__.__name__ == 'SlimModule'
            and 'QATQuantize' in self.frontend_conv[0].pre_ops
            and self.args.get('backend', 'torch') == 'panther'
        ):
            conv0_state_size = ceil((conv0_state_size + 3) // 4, 16)
        if (
            self.frontend_conv[2].__class__.__name__ == 'SlimModule'
            and 'QATQuantize' in self.frontend_conv[2].pre_ops
            and self.args.get('backend', 'torch') == 'panther'
        ):
            conv1_state_size = ceil((conv1_state_size + 3) // 4, 16)
        # remain redundant fsmn_right_kernel_size * fsmn_dilation space
        # align to panther impl
        fsmn_state_size = (
            (self.fsmn_left_kernel_size + self.fsmn_right_kernel_size * 2)
            * self.fsmn_dilation
            * self.memory_size
        )
        return conv0_state_size + conv1_state_size + fsmn_state_size

    def forward(self, fbank_feat, fbank_mask=None):
        '''forward.'''
        (bsz, frame_size, _) = fbank_feat.size()
        assert (
            frame_size % self.downsampling_size == 0
        ), "plz padding the fbank to 12, which is suitable for downsamping size 3/4"
        if fbank_mask is not None:
            fsmn_mask = fbank_mask.view(
                bsz, frame_size // self.downsampling_size, self.downsampling_size
            )[
                :, :, 0
            ]  # (B, T)
        else:
            fsmn_mask = None
        if self.training or fbank_mask is None:
            # (B, C, T, H)
            conv_out = self.frontend_conv(fbank_feat.unsqueeze(1).contiguous())
        else:
            conv_out = fbank_feat.unsqueeze(1).contiguous()  # (B, C, T, H)
            conv_out = self.frontend_conv[0](conv_out)  # downsample 2 by conv2d
            conv_out *= fbank_mask[:, ::2].reshape(bsz, 1, frame_size // 2, 1)
            conv_out = self.frontend_conv[1](conv_out)  # relu
            conv_out = self.frontend_conv[2](conv_out)  # downsample 2 by conv2d
            conv_out = self.frontend_conv[3](conv_out)  # relu
        if fsmn_mask is not None:
            conv_out = conv_out * fsmn_mask.unsqueeze(1).unsqueeze(3)

        conv_proj_out = self.conv_proj_fc(
            conv_out.transpose(1, 2).contiguous().view(bsz, -1, 128 * self.args.fbank_dim // 4)
        )  # (B, T, N)
        fsmn_out = self.input_trans_fc(conv_proj_out.transpose(1, 2).contiguous())  # (B, N, T)
        if fsmn_mask is not None:
            fsmn_out = fsmn_out * fsmn_mask.unsqueeze(1)
        return fsmn_out, fsmn_mask, "BNT"

    def forward_step(self, x, x_mask=None, states=None, x_sign=None):
        '''forward step'''
        bsz, frame_size, _ = x.size()
        state_num = (3 - 1) * 1 + 1 - 2  # (kernel_shape[0] - 1) * dilation[0] + 1 - stride[0]
        if isinstance(states, torch.Tensor):
            conv0_state_size = 1 * state_num * self.idim
            conv1_state_size = self.conv0_out_ch * state_num * self.idim // 2
            fsmn_state_size = (
                (self.fsmn_left_kernel_size + self.fsmn_right_kernel_size)
                * self.fsmn_dilation
                * self.memory_size
            )
            conv0_state = states[:, :conv0_state_size].reshape(-1, 1, state_num, self.idim)
            conv1_state = states[:, conv0_state_size : conv0_state_size + conv1_state_size].reshape(
                -1, self.conv0_out_ch, state_num, self.idim // 2
            )
            conv_states = [conv0_state, conv1_state]
            fsmn_states = states[
                :,
                conv0_state_size
                + conv1_state_size : conv0_state_size
                + conv1_state_size
                + fsmn_state_size,
            ].reshape(
                -1,
                self.memory_size,
                (self.fsmn_left_kernel_size + self.fsmn_right_kernel_size) * self.fsmn_dilation,
            )

        x = x.unsqueeze(1)  # (B, C, T, H)
        new_x, conv0_state = streaming_conv2d_input(
            x,
            conv_states[0],
            x_sign,
            state_num=state_num,
            padding=1,
        )
        fake_quantize_symmetric = (
            QunatizeFuncProducer.get('FakeQuantizeSymmetric')
            if QunatizeFuncProducer is not None
            else None
        )
        frontend_conv_0 = (
            getattr(self.frontend_conv[0], '_torch_module')
            if self.frontend_conv[0].__class__.__name__ == 'PantherConv'
            else self.frontend_conv[0]
        )
        if (
            fake_quantize_symmetric is not None
            and frontend_conv_0.__class__.__name__ == 'SlimModule'
            and 'QATQuantize' in frontend_conv_0.pre_ops
        ):
            x = F.conv2d(
                fake_quantize_symmetric.apply(
                    new_x,
                    frontend_conv_0.pre_ops['QATQuantize'].input_pre_process_modules.get_scale(),
                    8,
                ),
                fake_quantize_symmetric.apply(
                    frontend_conv_0.weight,
                    frontend_conv_0.pre_ops['QATQuantize'].weight_pre_process_modules.get_scale(),
                    8,
                ),
                frontend_conv_0.bias,
                stride=2,
                padding=(0, 1),
            )
        else:
            x = F.conv2d(
                new_x,
                frontend_conv_0.weight,
                frontend_conv_0.bias,
                stride=2,
                padding=(0, 1),
            )
        if x_mask is not None:
            x *= x_mask[:, ::2].reshape(bsz, 1, frame_size // 2, 1)
        x = self.frontend_conv[1](x)  # relu
        new_x, conv1_state = streaming_conv2d_input(
            x,
            conv_states[1],
            x_sign,
            state_num=state_num,
            padding=1,
        )
        frontend_conv_2 = (
            getattr(self.frontend_conv[2], '_torch_module')
            if self.frontend_conv[2].__class__.__name__ == 'PantherConv'
            else self.frontend_conv[2]
        )
        if (
            fake_quantize_symmetric is not None
            and frontend_conv_2.__class__.__name__ == 'SlimModule'
            and 'QATQuantize' in frontend_conv_2.pre_ops
        ):
            x = F.conv2d(
                fake_quantize_symmetric.apply(
                    new_x,
                    frontend_conv_2.pre_ops['QATQuantize'].input_pre_process_modules.get_scale(),
                    8,
                ),
                fake_quantize_symmetric.apply(
                    frontend_conv_2.weight,
                    frontend_conv_2.pre_ops['QATQuantize'].weight_pre_process_modules.get_scale(),
                    8,
                ),
                frontend_conv_2.bias,
                stride=2,
                padding=(0, 1),
            )
        else:
            x = F.conv2d(
                new_x,
                frontend_conv_2.weight,
                frontend_conv_2.bias,
                stride=2,
                padding=(0, 1),
            )
        x = self.frontend_conv[3](x)  # relu

        b, c, t, f = x.size()
        conv_proj_out = self.conv_proj_fc(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is not None:
            x_mask = x_mask[:, :: self.downsampling_size]

        new_conv_states = [
            conv0_state.reshape(-1, conv0_state_size),
            conv1_state.reshape(-1, conv1_state_size),
        ]
        new_conv_states = torch.cat(new_conv_states, dim=1)

        if x_mask is not None:
            conv_proj_out = conv_proj_out * x_mask.unsqueeze(1).unsqueeze(3)

        conv_proj_out = self.input_trans_fc[0](conv_proj_out.transpose(1, 2).contiguous())
        conv_proj_out = self.input_trans_fc[1](conv_proj_out)
        conv_proj_out = self.input_trans_fc[2](conv_proj_out)

        if x_sign in (0, 1):
            # mid and first
            conv_proj_out = torch.cat([fsmn_states, conv_proj_out], dim=2)  # (B, N, T)
        elif x_sign == 2:
            # last
            conv_proj_out = torch.cat([fsmn_states, conv_proj_out], dim=2)  # (B, N, T)
            conv_proj_out = F.pad(
                conv_proj_out,
                (
                    0,
                    self.fsmn_right_kernel_size * self.fsmn_dilation,
                    0,
                    0,
                ),
            )  # (B,N,T+(l+r)*d)
        elif x_sign == 3:
            # nonstream: first&&last
            conv_proj_out = F.pad(
                conv_proj_out,
                (
                    self.fsmn_left_kernel_size * self.fsmn_dilation,
                    self.fsmn_right_kernel_size * self.fsmn_dilation,
                    0,
                    0,
                ),
            )  # (B,N,T+(l+r)*d)
        else:
            raise NotImplementedError('x_sign only support 0, 1, 2, 3 !')

        new_fsmn_states = conv_proj_out[
            :,
            :,
            -(self.fsmn_left_kernel_size + self.fsmn_right_kernel_size) * self.fsmn_dilation :,
        ]
        fsmn_out = self.input_trans_fc[3].memory(conv_proj_out)

        if x_sign == 1:
            fsmn_out = fsmn_out[:, :, self.fsmn_right_kernel_size * self.fsmn_dilation :]

        new_fsmn_states = new_fsmn_states.reshape(-1, fsmn_state_size)
        new_states = torch.cat([new_conv_states, new_fsmn_states], dim=1)
        states = torch.zeros([bsz, self.state_size]).type_as(new_states)
        states[:, : new_states.size(1)] = new_states

        if x_mask is not None:
            fsmn_out = fsmn_out * x_mask.unsqueeze(1)
        return fsmn_out, states, x_mask, "BNT"


class VGGFrontEndNoFsmn(nn.Module):
    '''VGGFrontEndNoFsmn.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args
        conv0_out_ch = args.get('front_end_conv0_ch', 128)
        conv1_out_ch = args.get('front_end_conv1_ch', 128)
        self.frontend_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, conv0_out_ch, 3, 2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(conv0_out_ch, conv1_out_ch, 3, 2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_proj_fc = nn.Linear(conv1_out_ch * args.fbank_dim // 4, args.backbone_memory_size)

        self.downsampling_size = args.downsampling_size
        assert self.downsampling_size == 4

    def forward(self, fbank, fbank_mask=None):
        '''forward.'''
        (bsz, frame_size, _) = fbank.size()
        assert (
            frame_size % self.downsampling_size == 0
        ), "plz padding the fbank to 4, which is suitable for downsamping"
        out_mask = fbank_mask.view(
            bsz, frame_size // self.downsampling_size, self.downsampling_size
        )[
            :, :, 0
        ]  # (B, T)
        if self.training or fbank_mask is None:
            # (B, C, T, H)
            conv_out = self.frontend_conv(fbank.unsqueeze(1).contiguous())
        else:
            conv_out = fbank.unsqueeze(1).contiguous()  # (B, C, T, H)
            conv_out = self.frontend_conv[0](conv_out)  # downsample 2 by conv2d
            conv_out *= fbank_mask[:, ::2].reshape(bsz, 1, frame_size // 2, 1)
            conv_out = self.frontend_conv[1](conv_out)  # relu
            conv_out = self.frontend_conv[2](conv_out)  # downsample 2 by conv2d
            conv_out = self.frontend_conv[3](conv_out)  # relu
        conv_proj_out = self.conv_proj_fc(
            conv_out.transpose(1, 2).contiguous().view(bsz, -1, 128 * self.args.fbank_dim // 4)
        )  # (B, T, N)
        output = conv_proj_out.transpose(1, 2).contiguous() * out_mask.unsqueeze(1)
        # (B, N, T)
        return output, out_mask, "BNT"


class VGGPosFrontEnd(nn.Module):
    '''VGGPosFrontEnd.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        assert args.downsampling_size == 4
        conv0_out_ch = args.get('front_end_conv0_ch', 128)
        conv1_out_ch = args.get('front_end_conv1_ch', 128)
        self.frontend_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, conv0_out_ch, 3, 2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(conv0_out_ch, conv1_out_ch, 3, 2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_proj_fc = nn.Linear(conv1_out_ch * args.fbank_dim // 4, args.backbone_memory_size)
        self.pos_en = AbsPositionalEncoding(args.backbone_memory_size, args.dropout)
        self.downsampling_size = args.downsampling_size
        self.fbank_dim = args.fbank_dim

    def forward(self, fbank, fbank_mask=None):
        '''forward.'''
        (bsz, _, _) = fbank.size()
        # (B, T)
        if fbank_mask is not None:
            out_mask1 = fbank_mask[:, ::2]
            out_mask = out_mask1[:, ::2]
        else:
            out_mask1 = None
            out_mask = None
        if self.training or fbank_mask is None:
            # (B, C, T, H)
            conv_out = self.frontend_conv(fbank.unsqueeze(1).contiguous())
        else:
            conv_out = fbank.unsqueeze(1).contiguous()  # (B, C, T, H)
            conv_out = self.frontend_conv[0](conv_out)  # downsample 2 by conv2d
            conv_out *= out_mask1.unsqueeze(1).unsqueeze(3)
            conv_out = self.frontend_conv[1](conv_out)  # relu
            conv_out = self.frontend_conv[2](conv_out)  # downsample 2 by conv2d
            conv_out = self.frontend_conv[3](conv_out)  # relu
        # (B, T, N, 15)
        conv_out = conv_out.transpose(1, 2).contiguous()
        # (B, T, N)
        proj_out = self.conv_proj_fc(conv_out.view(bsz, -1, 128 * self.fbank_dim // 4))
        pos_out = self.pos_en(proj_out)
        # (B, N, T)
        if out_mask is not None:
            output = pos_out.transpose(1, 2).contiguous() * out_mask.unsqueeze(1)
        else:
            output = pos_out.transpose(1, 2).contiguous()
        return output, out_mask, "BNT"


class TimeReduceLSTMP(nn.Module):
    '''TimeReduceLSTMP.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args
        time_reduce_type = args.get('time_reduce_type', 'TimeReduce')
        time_reduce_layer = args.get('time_reduce_layer', 2)
        reduce_layer = []
        if time_reduce_type in ('AvgPoolTimeReduce', 'MaxPoolTimeReduce'):
            for i in range(time_reduce_layer):
                reduce_layer.append(eval(time_reduce_type)(kernel_size=2, stride=2, padding=0))
            lstm_input_size = args.backbone_memory_size
        elif time_reduce_type == 'Conv1dTimeReduce':
            for i in range(time_reduce_layer):
                reduce_layer.append(Conv1dTimeReduce(args.backbone_memory_size))
            lstm_input_size = args.backbone_memory_size
        else:
            for i in range(time_reduce_layer):
                reduce_layer.append(eval(time_reduce_type)(2, args.backbone_memory_size, padding=0))
            if time_reduce_type == 'LinearTimeReduce':
                lstm_input_size = args.backbone_memory_size
            else:
                lstm_input_size = 2 * args.backbone_memory_size
        self.fbank_dim = args.fbank_dim + args.get('domain_num', 0)
        self.input_concat_size = args.input_concat_size
        self.backbone_hidden_size = args.backbone_hidden_size
        self.unfold = torch.nn.Unfold(
            (args.input_concat_size, 1), stride=(1, 1), padding=(args.input_concat_size // 2, 0)
        )
        self.frontend_list = [
            LSTMP(
                self.fbank_dim * args.input_concat_size,
                args.backbone_hidden_size,
                bidirectional=bool(args.backbone_bilstm),
                dropout=args.dropout,
                output_size=args.backbone_memory_size,
                residual=bool(args.backbone_residual),
            )
        ]
        for i in range(time_reduce_layer):
            self.frontend_list.append(reduce_layer[i])
            self.frontend_list.append(
                LSTMP(
                    lstm_input_size,
                    args.backbone_hidden_size,
                    bidirectional=bool(args.backbone_bilstm),
                    dropout=args.dropout,
                    output_size=args.backbone_memory_size,
                    residual=bool(args.backbone_residual),
                )
            )
        self.frontend = nn.Sequential(*self.frontend_list)
        self.time_reduce_layer = time_reduce_layer
        self.downsampling_size = self.time_reduce_layer * 2

    def run_unfold(self, fbank, fbank_mask=None):
        '''run unfold'''
        (bsz, _, _) = fbank.size()
        if fbank_mask is not None:
            unfold_fbank = (
                (self.unfold(fbank.unsqueeze(1)))
                .contiguous()
                .view(bsz, self.input_concat_size, -1, self.fbank_dim)
            )
            unfold_fbank = (
                unfold_fbank.transpose(1, 2)
                .contiguous()
                .view(bsz, -1, self.input_concat_size * self.fbank_dim)
            )
            unfold_fbank_mask = self.unfold(fbank_mask.unsqueeze(1).unsqueeze(3))
            encoder_mask_out = unfold_fbank_mask[:, self.input_concat_size // 2, 1::2].contiguous()
            for _ in range(1, self.time_reduce_layer):
                encoder_mask_out = encoder_mask_out[:, 1::2].contiguous()
        else:
            encoder_mask_out = None
            unfold_fbank = fbank
        return unfold_fbank, encoder_mask_out

    def forward(self, fbank, fbank_mask=None):
        '''forward.'''
        unfold_fbank, encoder_mask_out = self.run_unfold(fbank, fbank_mask)
        encoder_out = self.frontend(unfold_fbank).transpose(1, 2).contiguous()
        return encoder_out, encoder_mask_out, "BNT"

    def forward_step(self, unfold_fbank, fbank_mask=None, states=None, **_kwargs):
        '''forward_step'''
        assert unfold_fbank.dim() == 3
        assert unfold_fbank.shape[1] % self.downsampling_size == 0
        assert unfold_fbank.shape[2] == self.fbank_dim * self.input_concat_size
        x = unfold_fbank

        # split global_state_in
        if isinstance(states, torch.Tensor):
            new_states = []
            state_offset = 0
            for _ in range(self.time_reduce_layer + 1):
                h0 = states[:, state_offset : state_offset + self.backbone_hidden_size]
                state_offset += self.backbone_hidden_size
                c0 = states[:, state_offset : state_offset + self.backbone_hidden_size]
                state_offset += self.backbone_hidden_size
                new_states.append((h0.unsqueeze(0), c0.unsqueeze(0)))
            states = new_states

        new_states = []
        for i in range(self.time_reduce_layer * 2 + 1):
            if i % 2 == 0:  # lstm layer
                x, (h1, c1) = self.frontend[i].forward_step(x, states[i // 2])
                new_states.append((h1.squeeze(0), c1.squeeze(0)))
            else:  # time reduce layer
                x = self.frontend[i](x)
                if fbank_mask is not None:
                    fbank_mask = fbank_mask[:, ::2]
        frontend_out = x.transpose(1, 2).contiguous()
        # new_states is an instance of [(hn, cn), (hn, cn), ...]
        return frontend_out, new_states, fbank_mask, "BNT"

    @property
    def state_size(self):
        '''streamed state size.'''
        return (1 + self.time_reduce_layer) * 2 * self.backbone_hidden_size


class Conv2dPooling(torch.nn.Module):
    '''Conv2dPooling'''

    def __init__(self, args):
        '''constructor'''
        super().__init__()
        self.idim = args.fbank_dim
        self.conv0_out_ch = args.get('front_end_conv0_ch', 128)
        self.conv1_out_ch = args.get('front_end_conv1_ch', 128)
        self.conv_padding = args.get('front_end_padding', 1)
        odim = args.backbone_memory_size

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.conv0_out_ch, 3, 2, self.conv_padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.conv0_out_ch, self.conv1_out_ch, 3, 2, self.conv_padding),
            torch.nn.ReLU(),
        )

        if self.conv_padding == 0:
            li_dim = ((self.idim - 1) // 2 - 1) // 2
        else:
            li_dim = self.idim // 4
        self.out = torch.nn.Linear(self.conv1_out_ch * li_dim, odim)
        self.downsampling_size = args.get('downsampling_size', 4)
        assert self.downsampling_size == 4

    def forward(self, x, x_mask=None):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
               or Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        """
        bsz, frame_size, _ = x.size()
        if self.training:
            x = x.unsqueeze(3).contiguous()  # (B,T,H) -> (B,T,H,1) [N,H,W,C]
            x = x.permute(0, 3, 1, 2)  # (B,T,H,1) [N,H,W,C] -> (B,1,T,H) [N,C,H,W]
        else:
            x = x.unsqueeze(1)
        if self.training or x_mask is None or self.conv_padding == 0:
            # (B,1,T,H)  [N,C,H,W]
            x = self.conv(x)
        else:
            x = self.conv[0](x)  # downsample 2 by conv2d
            x *= x_mask[:, ::2].reshape(bsz, 1, (frame_size + 1) // 2, 1)
            x = self.conv[1](x)  # relu
            x = self.conv[2](x)  # downsample 2 by conv2d
            x = self.conv[3](x)  # relu
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None, "BTN"
        if self.conv_padding == 0:
            return x, x_mask[:, :-2:2][:, :-2:2].squeeze(1), "BTN"
        return x, x_mask[:, :: self.downsampling_size], "BTN"

    def forward_step(self, x, x_mask=None, states=None, x_sign=None, required_right_context=0):
        '''
        forward step.
        Args:
            x_sign: tensor(dtype=int32, shape=[1])
                0: middle frame
                1: first frame
                2: last frame
                3: first frame and last frame
        '''
        state_num = (3 - 1) * 1 + 1 - 2  # (kernel_shape[0] - 1) * dilation[0] + 1 - stride[0]
        if isinstance(states, torch.Tensor):
            conv0_state_size = 1 * state_num * self.idim
            conv1_state_size = self.conv0_out_ch * state_num * self.idim // 2
            conv0_state = states[:, :conv0_state_size].reshape(-1, 1, state_num, self.idim)
            conv1_state = states[:, conv0_state_size:].reshape(
                -1, self.conv0_out_ch, state_num, self.idim // 2
            )
            states = [conv0_state, conv1_state]

        bsz, frame_size, _ = x.size()
        x = x.unsqueeze(1)  # (B, C, T, H)
        new_x, conv0_state = streaming_conv2d_input(
            x,
            states[0],
            x_sign,
            state_num=state_num,
            padding=1,
            required_right_context=required_right_context,
        )
        x = F.conv2d(new_x, self.conv[0].weight, self.conv[0].bias, stride=2, padding=(0, 1))
        if x_mask is not None:
            x *= x_mask[:, ::2].reshape(bsz, 1, frame_size // 2, 1)
        x = self.conv[1](x)  # relu
        new_x, conv1_state = streaming_conv2d_input(
            x,
            states[1],
            x_sign,
            state_num=state_num,
            padding=1,
            required_right_context=required_right_context // 2,
        )
        x = F.conv2d(new_x, self.conv[2].weight, self.conv[2].bias, stride=2, padding=(0, 1))
        x = self.conv[3](x)  # relu
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is not None:
            x_mask = x_mask[:, :: self.downsampling_size]

        new_states = [
            conv0_state.reshape(-1, conv0_state_size),
            conv1_state.reshape(-1, conv1_state_size),
        ]
        new_states = torch.cat(new_states, dim=1)
        return x, new_states, x_mask, "BTN"

    @property
    def state_size(self):
        '''streamed state size.'''
        conv0_state_size = 1 * ((3 - 1) * 1 + 1 - 2) * self.idim
        conv1_state_size = self.conv0_out_ch * ((3 - 1) * 1 + 1 - 2) * self.idim // 2
        return conv0_state_size + conv1_state_size


# compatible with old config
Conv2dPooling4 = Conv2dPooling


# TODO: use a more descriptive class name for this frontend
class CifOriginalFrontend(nn.Module):
    '''CIF Original Frontend'''

    def __init__(self, args):
        super().__init__()
        self.args = FalconDict(args)
        if self.args.use_cnn_frontend:
            layers = []
            feat_channel = self.args.fbank_channel
            feat_dim = int(self.args.fbank_dim)
            # Downsampling by strided cnns
            for _ in range(self.args.down_sample_conv_num_layers):
                layers.append(
                    SamePaddingConv2D(
                        feat_channel,
                        self.args.down_sample_conv_num_filters,
                        3,
                        2,
                        norm_type=self.args.conv_norm_type,
                        act_type='relu',
                    )
                )
                feat_channel = self.args.down_sample_conv_num_filters
                feat_dim /= 2
            # Additional module
            if self.args.additional_module:
                filters = self.args.additional_module_num_filters
                if self.args.additional_module == "mu33":
                    raise NotImplementedError('mu33 module not support yet')
                if self.args.additional_module == "mu33_2":
                    raise NotImplementedError('mu33_2 module not support yet')
                if self.args.additional_module == "new_mu33_2":
                    for _ in range(self.args.additional_module_num_layers):
                        layers.append(MultiplicativeUnit(feat_channel, filters))
                        feat_channel = filters
                else:
                    raise NotImplementedError('unknown additional module')
            self.frontend_convs = nn.Sequential(*layers)
            feat_size = int(feat_channel * feat_dim)
        else:
            raise NotImplementedError('non-cnn frontend not support yet')
        # output projection
        self.frontend_dense = nn.Linear(feat_size, self.args.hidden_size)

    def forward(self, inputs, inputs_mask):
        '''forward of CifOriginalFrontend'''
        # Store the orginal inputs
        org_inputs = inputs

        # Acquire original ignore padding which should be downsampled in ASR
        ishape = inputs.shape
        inputs = torch.reshape(inputs, [ishape[0], ishape[1], -1])
        # embedding_to_padding in tf implementation
        if self.args.use_input_padding:
            encoder_padding = 1.0 - inputs_mask
        else:
            encoder_padding = (inputs.abs().sum(dim=-1) == 0.0).float()
        # attention_bias_ignore_padding in tf implementation
        org_ignore_padding = (encoder_padding * -1e9).unsqueeze(1).unsqueeze(1)

        if self.args.use_chunk_hopping:
            org_inputs = org_inputs + 1e-10

        if self.args.use_cnn_frontend:
            downsampling_rate = 2**self.args.down_sample_conv_num_layers
            # transpose inputs from nhwc to nchw
            frontend_inps = org_inputs.permute(0, 3, 1, 2)
            frontend_outs = self.frontend_convs(frontend_inps)
            # transpose outputs from nchw to nhwc
            frontend_outs = frontend_outs.permute(0, 2, 3, 1).contiguous()
            out_shape = frontend_outs.shape
            frontend_outs = frontend_outs.reshape(out_shape[0], out_shape[1], -1)
        else:
            raise NotImplementedError('non-cnn frontend not support yet')
        encoder_input = self.frontend_dense(frontend_outs)

        # acquire attention bias
        ignore_padding = F.max_pool2d(
            org_ignore_padding,
            (1, downsampling_rate),
            (1, downsampling_rate),
            ceil_mode=True,
        )

        return encoder_input, ignore_padding


class SamePaddingConv2DFrontend(nn.Module):
    '''Same Padding Conv2d Frontend'''

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.args = args
        layers = []
        conv_layers = args.get('conv_layers', 2)
        conv_kernels = eval(args.get('conv_kernels', '[3, 3]'))
        conv_filters = eval(args.get('conv_filters', '[64, 128]'))
        conv_strides = eval(args.get('conv_strides', '[2, 2]'))
        self.conv_strides = conv_strides
        feat_dim = self.args.fbank_dim
        odim = args.backbone_memory_size
        # Downsampling by strided cnns
        feat_channel = 1
        for i in range(conv_layers):
            layers.append(
                SamePaddingConv2D(
                    feat_channel,
                    conv_filters[i],
                    conv_kernels[i],
                    conv_strides[i],
                    norm_type=self.args.conv_norm_type,
                    act_type='relu',
                    norm_momentum=args.get('norm_momentum', 0.001),
                    norm_eps=args.get('norm_eps', 0.001),
                )
            )
            feat_channel = conv_filters[i]
            feat_dim /= conv_strides[i]

        self.frontend_convs = nn.Sequential(*layers)
        feat_size = int(feat_channel * feat_dim)

        # output projection
        self.frontend_dense = nn.Linear(feat_size, odim)

    def forward(self, inputs, inputs_mask):
        '''forward'''

        # Acquire original ignore padding which should be downsampled in ASR
        ishape = inputs.shape
        inputs = torch.reshape(inputs, [ishape[0], ishape[1], -1, 1])

        # transpose inputs from nhwc to nchw
        frontend_inps = inputs.permute(0, 3, 1, 2)
        frontend_outs = frontend_inps
        bsz = inputs_mask.size(0)
        for i, frontend_conv in enumerate(self.frontend_convs):
            inputs_mask = inputs_mask[:, self.conv_strides[i] - 1 :: self.conv_strides[i]]
            frontend_outs = frontend_conv(frontend_outs) * inputs_mask.view(bsz, 1, -1, 1)
        # transpose outputs from nchw to nhwc
        frontend_outs = frontend_outs.permute(0, 2, 3, 1).contiguous()
        out_shape = frontend_outs.shape
        frontend_outs = frontend_outs.reshape(out_shape[0], out_shape[1], -1)
        encoder_input = self.frontend_dense(frontend_outs)
        return encoder_input, inputs_mask, "BTN"
