'''
coustic_head.py.
'''
import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.feed_forward import LinearClamp
from core.models.layers.stat_pooling import StatPoolingLayer, StreamStatPoolingLayer


class HybridCEHead(nn.Module):
    '''
    HybridCEHead.
    '''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        mtl_clamp = args.get('mtl_clamp', 0)
        self.head_fc_trans = nn.Sequential(
            *[
                LinearClamp(args.backbone_memory_size, args.head_hidden_size, clamp=mtl_clamp),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                # connot insert new module in this list.
                # which will cause resume and save not compatible.
                LinearClamp(args.head_hidden_size, args.head_hidden_size, clamp=mtl_clamp),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                LinearClamp(
                    args.head_hidden_size, args.head_lowrank_size, bias=False, clamp=mtl_clamp
                ),
                LinearClamp(args.head_lowrank_size, args.tgt_vocab_size, clamp=mtl_clamp),
            ]
        )

    def forward(self, input_data):
        '''forward.'''
        ce_logits = self.head_fc_trans(input_data)
        return ce_logits


class CTCHead(nn.Module):
    '''CTCHead'''

    def __init__(self, args):
        super().__init__()
        # consistent with samiasr
        self.ctchead_input_size = args.get('ctchead_input_size', args.backbone_memory_size)
        self.ctchead_num_blocks = args.get('ctchead_num_blocks', 1)
        self.ctchead_hidden_size = args.get('ctchead_hidden_size', 2048)
        self.dropout_rate = args.mtl_dropout

        if self.ctchead_num_blocks == 1:
            self.head_fc_trans = nn.Linear(self.ctchead_input_size, args.tgt_vocab_size)
        elif self.ctchead_num_blocks == 2:
            self.fc1 = nn.Linear(args.ctchead_input_size, self.ctchead_hidden_size)
            self.act = nn.ReLU(inplace=False)
            self.fc2 = nn.Linear(self.ctchead_hidden_size, args.tgt_vocab_size)
        else:
            raise NotImplementedError()

    def forward(self, input_data):
        '''forward'''
        if self.ctchead_num_blocks == 1:
            input_data = self.head_fc_trans(
                F.dropout(input_data, p=self.dropout_rate, training=self.training)
            )
        else:
            input_data = self.fc1(input_data)
            input_data = self.act(input_data)
            input_data = self.fc2(input_data)
        return input_data


class LASHead(nn.Module):
    '''head with BN'''

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.layers = []
        self.layers += [
            nn.Linear(
                args.backbone_memory_size,
                args.decoder_input_size,
                bias=args.get('head_bias', False),
            )
        ]
        self.layers += [
            nn.BatchNorm1d(
                args.decoder_input_size,
                momentum=args.norm_momentum,
                eps=args.get('norm_eps', 0.001),
                affine=args.get('batchnorm_affine', True),
            )
        ]
        self.layers += [nn.ReLU(inplace=False)]
        if args.get('head_ln', False):
            self.layers += [
                nn.LayerNorm(args.decoder_input_size, eps=args.get('backbone_ln_eps', 1e-5))
            ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, input_data):
        '''forward'''
        # pylint: disable=invalid-name
        B, T, N = input_data.size()
        input_data = self.layers(input_data.view(-1, N))
        input_data = input_data.view(B, T, -1)
        return input_data


class LidHead(nn.Module):
    '''Lid Head'''

    def __init__(self, args):
        '''init'''
        super().__init__()
        stream_lid = args.get('stream_lid', False)
        if stream_lid:
            self.pooling = StreamStatPoolingLayer([1, 2], args.backbone_memory_size)
        else:
            self.pooling = StatPoolingLayer([1, 2], args.backbone_memory_size)
        self.lid_fc = torch.nn.Sequential(
            *[
                torch.nn.Linear(args.backbone_memory_size * 2, args.backbone_memory_size),
                torch.nn.ReLU(),
                torch.nn.Linear(args.backbone_memory_size, args.lang_num),
            ]
        )

    def forward(self, input_data, mask=None):
        '''forward'''
        out, _ = self.pooling(input_data, mask)
        return self.lid_fc(out)


class RnntBaseHead(nn.Module):
    '''RnntBaseHead.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args
        self.acoustic_head_proj_fc = nn.Linear(args.backbone_memory_size, args.jointer_hidden_size)
        self.layer_norm_after = args.get('layer_norm_after', False)
        if self.layer_norm_after:
            self.layer_norm = nn.LayerNorm(self.args.jointer_hidden_size)
        else:
            self.layer_norm = nn.LayerNorm(self.args.backbone_memory_size)

    def forward(self, backbone_output):
        '''forward.'''
        if self.layer_norm_after:
            acoustic_head_out = self.layer_norm(self.acoustic_head_proj_fc(backbone_output))
        else:
            acoustic_head_out = self.acoustic_head_proj_fc(self.layer_norm(backbone_output))
        return acoustic_head_out


class RnntDownsampleHead(nn.Module):
    '''RnntDownsampleHead'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args
        self.acoustic_head_downsample = nn.Conv1d(
            args.backbone_memory_size, args.jointer_hidden_size, 3, 2, 1
        )
        self.layer_norm_after = args.get('layer_norm_after', False)
        if self.layer_norm_after:
            self.layer_norm = nn.LayerNorm(self.args.jointer_hidden_size)
        else:
            self.layer_norm = nn.LayerNorm(self.args.backbone_memory_size)

    def forward(self, backbone_output, backbone_mask):
        '''forward.'''
        if self.layer_norm_after:
            h = self.acoustic_head_downsample(backbone_output.transpose(1, 2)).transpose(1, 2)
            acoustic_head_out = self.layer_norm(h)
        else:
            h = self.layer_norm(backbone_output)
            acoustic_head_out = self.acoustic_head_downsample(h.transpose(1, 2)).transpose(1, 2)
        return acoustic_head_out, backbone_mask[:, ::2]


class RnntSimpleHead(nn.Module):
    '''RnntSimpleHead.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args
        self.acoustic_head_proj_fc = nn.Linear(args.backbone_memory_size, args.jointer_hidden_size)

    def forward(self, backbone_output):
        '''forward.'''
        acoustic_head_out = self.acoustic_head_proj_fc(backbone_output)
        return acoustic_head_out


class RnntEncoderBaseHead(nn.Module):
    '''for DFSMN encoder ce pretrain'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.acoustic_head_proj_fc = nn.Linear(args.backbone_memory_size, args.head_hidden_size)
        self.layer_norm = nn.LayerNorm(self.args.head_hidden_size)
        self.relu = nn.ReLU()
        self.pred_fc = nn.Linear(args.head_hidden_size, args.tgt_vocab_size)

    def forward(self, backbone_output):
        '''for DFSMN encoder ce pretrain'''
        acoustic_head_out = self.acoustic_head_proj_fc(backbone_output)
        if self.args.get('head_clamp', False):
            acoustic_head_out = torch.clamp(acoustic_head_out, -4e4, 4e4)
        acoustic_head_out = self.layer_norm(acoustic_head_out)
        acoustic_head_out = self.pred_fc(self.relu(acoustic_head_out))
        return acoustic_head_out
