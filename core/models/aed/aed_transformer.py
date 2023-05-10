''' Transformer '''
import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.multi_head_attn import MultiheadAttention


class PantherVarMeanFunc(torch.autograd.Function):
    '''symbolic for panther custom VarMean op'''

    @staticmethod
    def forward(ctx, x, dim, keepdim):
        '''forward'''
        return torch.var_mean(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx):
        '''backward'''
        raise RuntimeError('backward of VarMeanPantherFunc is not supported')

    @staticmethod
    def symbolic(g, x, dim, keepdim):
        '''symbolic'''
        return g.op('PantherVarMean', x, dim_i=dim, keepdim_i=keepdim, outputs=2)


class TransformerBlock(nn.Module):
    '''Transformer block'''

    def __init__(self, d_models, num_heads):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_models)
        self.self_attn = MultiheadAttention(embed_dim=d_models, num_heads=num_heads)
        self.layer_norm_2 = nn.LayerNorm(d_models)
        self.fead_forwards = nn.Sequential(
            nn.Linear(d_models, 4 * d_models), nn.GLU(dim=-1), nn.Linear(2 * d_models, d_models)
        )

    def forward(self, encode):
        '''forward for transformer block'''
        residual = encode
        encode = self.layer_norm_1(encode)
        encode, _ = self.self_attn(encode, key_padding_mask=None, attn_mask=None)
        encode = encode + residual
        residual = encode
        encode = self.layer_norm_2(encode)
        encode = self.fead_forwards(encode)
        encode = encode + residual
        return encode


class CNNTransformer(nn.Module):
    '''CNN Transformer in AED and LID'''

    def __init__(
        self,
        args,
        cnn_base_channels=32,
        num_cnn_blocks=3,
        num_cnn_layers_pre_block=2,
        d_models=512,
        num_transformer_layers=4,
        transformer_dropout=0,
        num_heads=8,
    ):

        super().__init__()
        self.args = args
        self.cnn_base_channels = cnn_base_channels
        self.num_cnn_blocks = num_cnn_blocks
        self.num_cnn_layers_pre_block = num_cnn_layers_pre_block
        self.d_models = d_models
        self.num_transformer_layers = num_transformer_layers
        self.transformer_dropout = transformer_dropout
        self.num_heads = num_heads

        # conv block
        norm_momentum = 1.0 - self.args.batch_norm_decay
        norm_eps = self.args.batch_norm_eps
        # conv block
        conv_layers = []
        for i in range(self.num_cnn_blocks):
            if i == 0:
                conv_layers.append(nn.Conv2d(1, cnn_base_channels * (i + 1), 3, 1, 1))
                conv_layers.append(
                    nn.BatchNorm2d(cnn_base_channels * (i + 1), norm_eps, norm_momentum)
                )
                conv_layers.append(nn.ReLU(inplace=True))
                for _ in range(num_cnn_layers_pre_block - 1):
                    conv_layers.append(
                        nn.Conv2d(cnn_base_channels * (i + 1), cnn_base_channels * (i + 1), 3, 1, 1)
                    )
                    conv_layers.append(
                        nn.BatchNorm2d(cnn_base_channels * (i + 1), norm_eps, norm_momentum)
                    )
                    conv_layers.append(nn.ReLU(inplace=True))
                conv_layers.append(nn.MaxPool2d((2, 2)))
            else:
                conv_layers.append(
                    nn.Conv2d(cnn_base_channels * i, cnn_base_channels * (i + 1), 3, 1, 1)
                )
                conv_layers.append(
                    nn.BatchNorm2d(cnn_base_channels * (i + 1), norm_eps, norm_momentum)
                )
                conv_layers.append(nn.ReLU(inplace=True))
                for _ in range(num_cnn_layers_pre_block - 1):
                    conv_layers.append(
                        nn.Conv2d(cnn_base_channels * (i + 1), cnn_base_channels * (i + 1), 3, 1, 1)
                    )
                    conv_layers.append(
                        nn.BatchNorm2d(cnn_base_channels * (i + 1), norm_eps, norm_momentum)
                    )
                    conv_layers.append(nn.ReLU(inplace=True))
                conv_layers.append(nn.MaxPool2d((2, 2)))
        self.convs = nn.Sequential(*conv_layers)
        # encoder block
        self.linear = nn.Linear(
            int(self.args.fbank_dim / (2**self.num_cnn_blocks))
            * self.cnn_base_channels
            * self.num_cnn_blocks,
            self.d_models,
        )
        transformer_layers = []
        for _ in range(self.num_transformer_layers):
            transformer_layers.append(TransformerBlock(self.d_models, self.num_heads))
        self.transformer_blocks = nn.Sequential(*transformer_layers)

        if self.args.use_detection:
            self.first_linear = nn.Linear(self.d_models, self.args.feature_dim, bias=False)
        else:
            self.first_linear = nn.Linear(2 * self.d_models, self.args.feature_dim, bias=False)
        # fc block
        if self.args.use_batch_norm:
            self.fcs = nn.Sequential(
                self.first_linear,
                nn.BatchNorm1d(self.args.feature_dim, norm_eps, norm_momentum),
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.feature_dim, bias=False),
                nn.BatchNorm1d(self.args.feature_dim, norm_eps, norm_momentum),
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.num_classes, bias=True),
            )
        else:
            self.fcs = nn.Sequential(
                self.first_linear,
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.feature_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.num_classes, bias=True),
            )
        self.reset_parameters()

    def reset_parameters(self):
        '''reset_parameters'''
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _verify_batch_size(self, inputs):
        '''verify_batch_size'''
        if not self.training or not self.args.use_batch_norm:
            return
        if inputs.shape[0] <= 1:
            raise RuntimeError(
                'Expected more than 1 samples when training with batch_norm'
                ', get input size {}'.format(inputs.shape)
            )

    def forward(self, inputs):
        '''forward for cnn_transformer model'''
        self._verify_batch_size(inputs)
        conv_feat = self.convs(inputs).permute(0, 2, 3, 1)
        conv_feat_squeeze = conv_feat.reshape(conv_feat.size(0), conv_feat.size(1), -1).transpose(
            0, 1
        )
        encode_input = self.linear(conv_feat_squeeze)
        encode = F.dropout(encode_input, p=self.transformer_dropout, training=self.training)
        encode_feat = self.transformer_blocks(encode).transpose(0, 1)
        if not self.args.use_detection:
            if (torch.jit.is_scripting() or torch.jit.is_tracing()) and not self.training:
                feat_var, feat_mean = PantherVarMeanFunc.apply(encode_feat, 1, False)
            else:
                feat_var, feat_mean = torch.var_mean(encode_feat, dim=1, keepdim=False)
            feat_mean = torch.flatten(feat_mean, 1).contiguous()
            feat_var = torch.flatten(feat_var, 1).contiguous()
            feat = torch.cat([feat_mean, feat_var], dim=1)
        else:
            feat = encode_feat
        logits = self.fcs(feat)
        predicts = F.sigmoid(logits) if self.args.multi_labels else F.softmax(logits, -1)
        return logits, predicts
