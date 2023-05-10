"""
acoustic_backbone.py.
"""
# pylint:disable=too-many-lines
import torch
from torch import nn
from core.models.layers.embedding import (
    AbsPositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from core.models.layers.conformer import ConformerLayer


def rnnt_transpose(inputs, out_shape, frontend_shape="BTN"):
    "rnnt transpose"
    if out_shape == frontend_shape:
        return inputs
    pos = [0, 1, 2]
    for i in range(3):
        for j in range(3):
            if out_shape[i] == frontend_shape[j]:
                pos[i] = j
    inputs = inputs.permute(int(pos[0]), int(pos[1]), int(pos[2])).contiguous()
    return inputs


class ConformerBackbone(nn.Module):
    """ConformerBackbone"""

    def __init__(self, args):
        # pylint:disable=too-many-branches
        super().__init__()
        attention_dim = args.backbone_memory_size
        num_blocks = args.conformer_num_blocks
        positional_dropout_rate = args.conformer_positional_dropout_rate

        pos_enc_layer_type = args.conformer_pos_enc_layer_type

        normalize_before = args.conformer_normalize_before
        self.normalize_before = normalize_before
        self.export_fused_conformer = args.get('export_fused_conformer', False)

        # mask
        conformer_mask_topology = args.conformer_mask_topology
        if conformer_mask_topology is not None:
            conformer_mask_topology = eval(conformer_mask_topology)
            assert (
                len(conformer_mask_topology) == num_blocks
            ), "conformer_mask_topology not match num_blocks"
            self.export_rel_pos_embeding_len = 0
            for attn_left, attn_right in conformer_mask_topology:
                self.export_rel_pos_embeding_len = max(
                    self.export_rel_pos_embeding_len, attn_left, attn_right
                )
            self.export_rel_pos_embeding_len += 1
        else:
            conformer_mask_topology = [None] * num_blocks
            self.export_rel_pos_embeding_len = None
        dual_mode = args.get("conformer_dual_mode", False)
        if dual_mode:
            stream_conformer_mask_topology = args.get("stream_conformer_mask_topology", None)
            assert stream_conformer_mask_topology is not None
            stream_conformer_mask_topology = eval(stream_conformer_mask_topology)
            assert len(conformer_mask_topology) == len(
                stream_conformer_mask_topology
            ), "stream_conformer_mask_topology must match conformer_mask_topology"

        # net
        # if export to ONNX, conformer_positional_max_len is equal to
        # Max(attn_left_conext, attn_right_context) + 1
        positional_max_len = args.get('conformer_positional_max_len', 5000)
        if pos_enc_layer_type == "abs_pos":
            self.pos_enc = AbsPositionalEncoding(attention_dim, positional_dropout_rate)
        elif pos_enc_layer_type == "scaled_abs_pos":
            self.pos_enc = ScaledPositionalEncoding(attention_dim, positional_dropout_rate)
        elif pos_enc_layer_type in ("rel_pos", "fix_rel_pos"):
            assert args.conformer_selfattention_layer_type == "rel_selfattn"
            self.pos_enc = RelPositionalEncoding(
                attention_dim,
                positional_dropout_rate,
                dim=1,
                max_len=positional_max_len,
            )
            args.position_encoding_dim = 1  # keep the same with dim of RelPositionalEncoding
        elif pos_enc_layer_type == "none":
            self.pos_enc = None
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        assert args.conformer_layernorm_interval <= 0
        assert not args.conformer_half_pooling
        args.export_rel_pos_embeding_len = self.export_rel_pos_embeding_len
        encoders = []
        for lnum in range(num_blocks):
            encoders.append(ConformerLayer(args, conformer_mask_topology[lnum], lnum))
            if dual_mode:
                encoders[-1].self_attn.init_dual_mode_stream_mask(
                    stream_conformer_mask_topology[lnum]
                )
        self.encoders = nn.Sequential(*encoders)

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(attention_dim)

        if args.conformer_weight_scale < 1.0:
            for _, param in self.named_parameters():
                param.data.mul_(args.conformer_weight_scale)

    def set_stream_mode(self, stream_mode=False):
        """set_stream_mode"""
        for encoder in self.encoders:
            encoder.set_stream_mode(stream_mode)

    def set_stream_index(self, stream_index=0):
        """set_stream_mode"""
        for encoder in self.encoders:
            encoder.set_stream_index(stream_index)

    def forward(self, front_end_out, acoustic_mask=None, frontend_shape="BTN", **_kwargs):
        """Encode input sequence.

        :param torch.Tensor front_end_out: input tensor # (B, T, N)
        :param torch.Tensor acoustic_mask: input mask   # (B, T)
        :rtype torch.Tensor:
        """
        front_end_out = rnnt_transpose(front_end_out, "BTN", frontend_shape)
        if (
            (torch.jit.is_scripting() or torch.jit.is_tracing())
            and self.pos_enc.__class__.__name__ == 'RelPositionalEncoding'
            and not self.training
            and self.export_fused_conformer
        ):
            conformer_input = self.pos_enc(front_end_out, self.export_rel_pos_embeding_len)
        else:
            conformer_input = self.pos_enc(front_end_out)
        if acoustic_mask is None:
            conformer_mask = None
        else:
            conformer_mask = acoustic_mask.unsqueeze(1).squeeze(-1)
        conformer_out, conformer_mask = self.encoders([conformer_input, conformer_mask])
        if isinstance(conformer_out, tuple):
            conformer_out = conformer_out[0]

        if self.normalize_before:
            conformer_out = self.after_norm(conformer_out)
        return conformer_out  # , conformer_mask.squeeze(1)

    def forward_step(
        self,
        front_end_out,
        required_right_context,
        cache_list=None,
        acoustic_mask=None,
        frontend_shape="BTN",
    ):
        """Encode input sequence.

        :param torch.Tensor front_end_out: input tensor # (B, T, N)
        :param torch.Tensor acoustic_mask: input mask   # (B, T)
        :rtype torch.Tensor:
        """
        front_end_out = rnnt_transpose(front_end_out, "BTN", frontend_shape)
        conformer_input = self.pos_enc.forward_step(front_end_out)
        if acoustic_mask is None:
            conformer_mask = None
        else:
            conformer_mask = acoustic_mask.unsqueeze(1)
        if isinstance(conformer_input, tuple):
            x, pos_emb = conformer_input
            pe_max_len = int((pos_emb.size(1) + 1) / 2)
            attn_history_size = self.encoders[0].self_attn.left_kernel_size
            assert pe_max_len >= attn_history_size
            key_size = x.size(1) + attn_history_size
            pos_emb = pos_emb[:, pe_max_len - key_size : pe_max_len + key_size - 1]
            conformer_input = x, pos_emb

        for i, m in enumerate(self.encoders):
            att_cache, att_mask_cache, conv_cache = cache_list[i]
            conformer_input, conformer_mask, att_cache, att_mask_cache, conv_cache = m.forward_step(
                [conformer_input, conformer_mask],
                required_right_context,
                att_cache,
                att_mask_cache,
                conv_cache,
            )
            cache_list[i] = [att_cache, att_mask_cache, conv_cache]
        conformer_out = conformer_input

        if isinstance(conformer_out, tuple):
            conformer_out = conformer_out[0]

        if self.normalize_before:
            conformer_out = self.after_norm(conformer_out)
        return (
            conformer_out,
            required_right_context,
            cache_list,
            acoustic_mask,
        )  # , conformer_mask.squeeze(1)


class MaskedConformerBackbone(ConformerBackbone):
    """
    MaskedConformerBackbone
    compatible with old code.
    """
