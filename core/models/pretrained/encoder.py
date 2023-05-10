""" encoder """


import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from core.extensions import checkpoint_wrapper
from core.models.layers.transformer import BiTransformerLayer, RelTransformerLayer
from core.models.layers.multi_head_attn import MultiheadAttention
from core.models.layers.cnn import Conv1d
from core.models.pretrained.frontend import TransposeLast
from core.models.pretrained.utils import pad_to_multiple


class SamePad(nn.Module):
    """SamePad"""

    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        """forward"""
        if self.remove:
            x = x[:, :, :-1]
        return x


def make_conv_block(e, k, g, l):
    '''make_conv_block'''
    return nn.Sequential(
        *[
            nn.Sequential(
                Conv1d(
                    e,
                    e,
                    kernel_size=k,
                    padding=k // 2,
                    groups=g,
                ),
                SamePad(k),
                TransposeLast(),
                nn.LayerNorm(e, elementwise_affine=False),
                TransposeLast(),
                nn.GELU(),
            )
            for _ in range(l)
        ]
    )


def make_conv_pos(e, k, g):
    '''make_conv_pos'''
    pos_conv = Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class TransformerEncoder(nn.Module):
    """TransformerEncoder"""

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.output_layer_result = args.get('output_layer_result', False)
        self.fused_transformer = args.get('fused_transformer', True)
        self.required_seq_len_multiple = args.get('required_seq_len_multiple', 2)
        self.pad_to_multiple = args.get('pad_to_multiple', False)

        pos_conv_depth = args.get("pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)
            self.pos_conv = make_conv_block(self.embedding_dim, k, args.conv_pos_groups, num_layers)
        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        self.layers = nn.ModuleList(
            [self.layer_wrapper(args, i) for i in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(self.reset_parameters)

        self.num_updates = 0

    @staticmethod
    def reset_parameters(module):
        """
        Initialize the weights specific to the BERT Model.
        This overrides the default initializations depending on the specified arguments.
            1. If normal_init_linear_weights is set then weights of linear
            layer will be initialized using the normal distribution and
            bais will be set to the specified value.
            2. If normal_init_embed_weights is set then weights of embedding
            layer will be initialized using the normal distribution.
            3. If normal_init_proj_weights is set then weights of
            in_project_weight for MultiHeadAttention initialized using
            the normal distribution (to be validated).
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            if module.merge_qkv:
                module.in_proj.weight.data.normal_(mean=0.0, std=0.02)
            else:
                module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
                module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
                module.v_proj.weight.data.normal_(mean=0.0, std=0.02)

    def layer_wrapper(self, args, idx):
        """layer_wrapper"""
        num_chkpt_layers = args.get('num_chkpt_layers', 0)
        transformer_model = args.get("transformer_model", "BiTransformerLayer")
        if transformer_model == 'BiTransformerLayer':
            cls = BiTransformerLayer
        elif transformer_model == 'RelTransformerLayer':
            cls = RelTransformerLayer
        layer = cls(
            embed_dim=self.embedding_dim,
            attention_heads=args.encoder_attention_heads,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_dropout=args.attention_dropout,
            hidden_dropout=self.dropout,
            activation_dropout=args.activation_dropout,
            activation=args.get("pretrained_activation_fn", args.activation_fn),
            normalize_before=args.layer_norm_first,
            squeeze_mem=idx >= num_chkpt_layers and args.get('squeeze_mem', True),
            gru_rel_pos=args.get("gru_rel_pos", True),
            attention_model=args.get("attention_model", "PosMultiHeadAttention"),
            adapter_mode=args.get('adapter_mode', 'none'),
            adapter_type=args.get('adapter_type', 'none'),
            adapter_embed_dim=args.get('adapter_embed_dim', -1),
        )
        if idx < num_chkpt_layers:
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
        self,
        x,
        padding_mask=None,
        layer=None,
        output_all_layers=False,
        include_first_layer=False,
        eval_fused=False,
        **kwargs,
    ):
        """forward"""
        x, layer_results = self.extract_features(
            x,
            padding_mask,
            layer,
            output_all_layers=output_all_layers,
            include_first_layer=include_first_layer,
            eval_fused=eval_fused,
            **kwargs,
        )

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    # pylint:disable=too-many-branches
    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        start_after_layer=-1,
        output_all_layers=False,
        include_first_layer=False,
        eval_fused=False,
        **kwargs,
    ):
        """
        Extract features.
        Args:
            x: [B x T x H], input data
            padding_mask: [B x T], padding mask
            tgt_layer: [int], target layer
            output_all_layers: [bool], whether to output all layer results
            include_first_layer: [bool], whether to include first layer
            eval_fused: [bool], whether to use fusion op in evaluation mode

        Returns:
            x: [B x T x H], transformer encoder output features
            layer_results: list of tuples containing:
                1. x: [B x T x H], transformer layer output
                2. attn_weight: temporarily None
                3. layer_results: [B x T x H], transformer ffn layer result

        """
        if tgt_layer is not None and tgt_layer < 0:
            return x

        if padding_mask is not None:
            x = x * (1 - padding_mask.long()).view(padding_mask.shape[0], padding_mask.shape[1], 1)

        # forward from every begining
        if start_after_layer < 0:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x += x_conv

            if not self.layer_norm_first:
                x = self.layer_norm(x)

        if self.pad_to_multiple:
            # pad to the sequence length dimension
            x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)
            if pad_length > 0 and padding_mask is None:
                padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
                padding_mask[:, -pad_length:] = True
            else:
                padding_mask, _ = pad_to_multiple(
                    padding_mask, self.required_seq_len_multiple, dim=-1, value=True
                )

        x = F.dropout(x, p=self.dropout, training=self.training)

        # NOTE(liuyi): This forward behaves different from forward_freeze_internal and
        # forward_attention. The input of the first transformer should be included.
        layer_results = []
        if include_first_layer:
            layer_results.append(x)
        r = None
        for i, layer in enumerate(self.layers):
            if start_after_layer >= i:
                continue  # skip layer
            use_this_layer = False
            dropout_probability = random.random() if self.training else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x = layer(
                    x,
                    encoder_padding_mask=padding_mask,
                    fused=self.fused_transformer,
                    eval_fused=eval_fused,
                    output_layer_result=self.output_layer_result,
                    batch_first=True,
                    **kwargs,
                )
                use_this_layer = True

            if self.output_layer_result:
                if i >= min_layer and (use_this_layer or output_all_layers):
                    layer_results.append(x)
                if use_this_layer:
                    x, _, _ = x
            else:
                if tgt_layer is not None and (use_this_layer or output_all_layers):
                    layer_results.append(x)

            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # undo paddding
        if self.pad_to_multiple and pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:, :-pad_length],
                    b[:, :-pad_length] if b is not None else b,
                    c[:, :-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def forward_internal(
        self,
        x,
        padding_mask=None,
        layer=-1,
        output_all_layers=False,
        include_first_layer=False,
        eval_fused=False,
        **kwargs,
    ):
        """forward"""
        x, layer_results = self.extract_features(
            x,
            padding_mask,
            start_after_layer=layer,
            output_all_layers=output_all_layers,
            include_first_layer=include_first_layer,
            eval_fused=eval_fused,
            **kwargs,
        )

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x, layer_results

    def forward_freeze_internal(
        self,
        x,
        padding_mask=None,
        freeze_layers=-1,
        output_hidden_states=False,
        output_all_layers=False,
        **kwargs,
    ):
        """forward_freeze_internal"""
        if padding_mask is not None:
            x = x * (1 - padding_mask.long()).view(padding_mask.shape[0], padding_mask.shape[1], 1)

        with torch.no_grad():
            x_conv = self.pos_conv(x.transpose(1, 2).contiguous())
            x_conv = x_conv.transpose(1, 2)
            x += x_conv

            if not self.layer_norm_first:
                x = self.layer_norm(x)

            x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = [x]
        for i, layer in enumerate(self.layers):
            use_this_layer = False
            if i >= freeze_layers:
                dropout_probability = random.random() if self.training else 1
                if dropout_probability > self.layerdrop:
                    x = layer(
                        x,
                        encoder_padding_mask=padding_mask,
                        fused=self.fused_transformer,
                        batch_first=kwargs.get('batch_first', True),
                        **kwargs,
                    )
                    use_this_layer = True
            else:
                with torch.no_grad():
                    x = layer(
                        x,
                        encoder_padding_mask=padding_mask,
                        fused=self.fused_transformer,
                        **kwargs,
                    )
                    use_this_layer = True
            if use_this_layer or output_all_layers:
                layer_results.append(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        if not output_hidden_states:
            layer_results = None
        return x, layer_results

    def forward_attention(self, x, padding_mask=None, output_hidden_states=False, **kwargs):
        """forward_attention"""
        if padding_mask is not None:
            x = x * (1 - padding_mask.long()).view(padding_mask.shape[0], padding_mask.shape[1], 1)

        x_conv = self.pos_conv(x.transpose(1, 2).contiguous())
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = [x]
        attn_weights = []
        for layer in self.layers:
            # x, attn = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=True)
            x = layer(
                x,
                encoder_padding_mask=padding_mask,
                fused=self.fused_transformer,
                batch_first=kwargs.get('batch_first', True),
                **kwargs,
            )
            layer_results += [x]
            # FIXME(zhengyijie): this transformer not support output attention weights
            # attn_weights += [attn]

        if self.layer_norm_first:
            x = self.layer_norm(x)

        if not output_hidden_states:
            layer_results = None

        return x, layer_results, attn_weights

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    # pylint: disable=unused-argument
    @staticmethod
    def upgrade_state_dict_named(state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
