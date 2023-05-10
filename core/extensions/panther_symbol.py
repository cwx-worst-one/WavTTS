'''
panther fused function for onnx export.
'''
from itertools import repeat
import collections.abc
import torch
import torch.nn.functional as F


def unused(g):
    """
    optional inputs
    """
    n = g.op("prim::Constant")
    n.setType(torch.jit.annotations.OptionalType.ofTensor())
    return n


def graph_const(g, tensor):
    """
    create Constant node
    """
    return g.op("Constant", value_t=torch.tensor(tensor, requires_grad=False))


def _ntuple(n):
    '''_ntuple'''

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class PantherUnfoldFunc(torch.autograd.Function):
    """
    PantherUnfold:
        im2col
    """

    @staticmethod
    def symbolic(g, x, kernel_size, dilation, padding, stride):
        '''symbolic'''
        return g.op(
            "PantherUnfold",
            x,
            kernel_size_i=_pair(kernel_size),
            dilation_i=_pair(dilation),
            padding_i=_pair(padding),
            stride_i=_pair(stride),
        )

    @staticmethod
    def forward(ctx, x, kernel_size, dilation, padding, stride):
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride
        return torch.nn.functional.unfold(x, kernel_size, dilation, padding, stride)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return (
            torch.nn.functional.fold(
                grad_output,
                list(x.size())[2:],
                ctx.kernel_size,
                ctx.dilation,
                ctx.padding,
                ctx.stride,
            ),
            None,
            None,
            None,
            None,
        )


class PantherAdaptiveSoftmaxFunc(torch.autograd.Function):
    """
    PantherAdaptiveSoftmax
    """

    @staticmethod
    def symbolic(
        g,
        x,
        head,
        tails,
        bins,
        temperature,
        tails_left=None,
        tails_right=None,
        tail_dim_start=50,
        tail_dim_decrease=1,
    ):
        """
        symbolic
        """
        new_tails = tails if tails is not None else unused(g)
        new_tails_left = tails_left if tails_left is not None else unused(g)
        new_tails_right = tails_right if tails_right is not None else unused(g)
        if isinstance(temperature, float):
            return g.op(
                "PantherAdaptiveSoftmax",
                x,
                head,
                new_tails,
                new_tails_left,
                new_tails_right,
                temperature_f=temperature,
                bins_i=bins,
                tail_dim_start_i=tail_dim_start,
                tail_dim_decrease_i=tail_dim_decrease,
            )
        return g.op(
            "PantherAdaptiveSoftmax",
            x,
            head,
            new_tails,
            new_tails_left,
            new_tails_right,
            temperature,
            bins_i=bins,
            tail_dim_start_i=tail_dim_start,
            tail_dim_decrease_i=tail_dim_decrease,
        )

    @staticmethod
    def forward(
        ctx,
        x,
        head,
        tails,
        bins,
        temperature,
        tails_left=None,
        tails_right=None,
        tail_dim_start=50,
        tail_dim_decrease=1,
    ):
        assert tails is not None or tails_left is not None
        if tails_left is not None:
            assert tails_right is not None
        if tails is None:
            idx = 0
            length = tail_dim_start
            tail_list = []
            for _ in range(bins):
                tail_list.append(
                    torch.matmul(
                        tails_left[:, idx : idx + length], tails_right[idx : idx + length, :]
                    )
                )
                idx += length
                length -= tail_dim_decrease
            tails = torch.cat(tail_list, dim=1)
        pred_head = F.log_softmax(x.matmul(head) * temperature, dim=-1)
        batch_size = x.shape[0]
        dims_per_bin = tails.shape[1] // bins
        pred_tail = F.log_softmax(x.matmul(tails).view(batch_size, -1, dims_per_bin), dim=-1)
        head_to_tail = pred_head[:, -bins:].unsqueeze(2)
        pred_tail = pred_tail + head_to_tail
        res = torch.cat((pred_head[:, :-bins], pred_tail.view(batch_size, -1)), dim=-1)
        return res

    @staticmethod
    def backward(ctx):
        raise RuntimeError('backward of PantherAdaptiveSoftmaxFunc is not supported')


class RnntJointerLMSymbolic(torch.autograd.Function):
    """
    Symbolic for export PantherJointer
    """

    @staticmethod
    def symbolic(g, model, acoustic_out, predictor_out, temperature=1.0):
        """
        export symbol for panther infer
        """
        gemm0_w = graph_const(g, model.jointer_module.joint_module[0].weight)
        gemm0_b = model.jointer_module.joint_module[0].bias
        gemm0_b = graph_const(g, gemm0_b) if gemm0_b is not None else unused(g)
        head = graph_const(g, model.criterion_module.log_softmax_fc.w_head)
        tail_left = graph_const(g, model.criterion_module.log_softmax_fc.w_tail_left)
        tail_right = graph_const(g, model.criterion_module.log_softmax_fc.w_tail_right)
        bins = len(model.criterion_module.log_softmax_fc.cutoff) - 1
        tail_dim_start = model.criterion_module.log_softmax_fc.tail_hidden_dim
        tail_dim_decrease = model.criterion_module.log_softmax_fc.tail_hidden_dim_decrease
        kwargs = {
            'slope_f': model.jointer_module.joint_module[1].negative_slope,
            'bins_i': bins,
            'tail_dim_start_i': tail_dim_start,
            'tail_dim_decrease_i': tail_dim_decrease,
            'gemm_trans_a_i': 0,
            'gemm_trans_b_i': 1,
        }

        if isinstance(temperature, float):
            kwargs['temperature_f'] = temperature
            args = [acoustic_out, predictor_out, gemm0_w, gemm0_b, head, tail_left, tail_right]
        else:
            args = [
                acoustic_out,
                predictor_out,
                gemm0_w,
                gemm0_b,
                head,
                tail_left,
                tail_right,
                temperature,
            ]
        return g.op('PantherJointer', *args, **kwargs, outputs=2)

    @staticmethod
    def forward(ctx, model, acoustic_out, predictor_out, temperature=1.0):
        """
        forward of PantherJointer
        """
        jointer_out1 = model.jointer_module.forward_step(acoustic_out, predictor_out)
        shape = jointer_out1.shape
        output1 = model.criterion_module.log_softmax_fc.jit_forward_non_combined(
            jointer_out1.reshape(-1, shape[-1]),
            temperature,
        )
        jointer_out2 = model.jointer_module.forward_step(
            torch.zeros_like(acoustic_out, device=acoustic_out.device),
            predictor_out,
        )
        output2 = model.criterion_module.log_softmax_fc.jit_forward_non_combined(
            jointer_out2.reshape(-1, shape[-1]),
            temperature,
        )
        output2[:, 0] = float('-inf')
        output2 = torch.log_softmax(output2, dim=-1)
        output2[:, 0] = 0
        return output1, output2

    @staticmethod
    def backward(ctx):
        """
        backward of PantherJointer
        """
        raise RuntimeError('backward of PantherJointer is not supported')


class MultiheadAttentionSymbolicExport(torch.autograd.Function):
    """Panther Transformer symbolic class"""

    @staticmethod
    def symbolic(
        g,
        x,
        key_padding_mask,
        in_proj_weight,
        in_proj_bias,
        out_proj_weight,
        out_proj_bias,
        num_heads,
        embed_dim,  # pylint: disable=unused-argument
        drop_out,  # pylint: disable=unused-argument
        need_weight,  # pylint: disable=unused-argument
    ):
        """symbolic"""
        if key_padding_mask is None:
            key_padding_mask = unused(g)

        kwargs = {
            'x_format_s': "TBN",
            'y_format_s': "TBN",
            'num_heads_i': num_heads,
        }

        return g.op(
            "MultiHeadAttention",
            x,
            key_padding_mask,
            in_proj_weight,
            in_proj_bias,
            out_proj_weight,
            out_proj_bias,
            **kwargs,
        )

    @staticmethod
    def forward(
        ctx,
        x,
        key_padding_mask,
        in_proj_weight,
        in_proj_bias,
        out_proj_weight,
        out_proj_bias,
        num_heads,
        embed_dim,
        drop_out,
        need_weight,
    ):
        return F.multi_head_attention_forward(
            x,
            x,
            x,
            embed_dim,
            num_heads,
            in_proj_weight.t(),
            in_proj_bias,
            None,
            None,
            False,
            drop_out,
            out_proj_weight.t(),
            out_proj_bias,
            False,
            key_padding_mask,
            need_weight,
        )[0]

    @staticmethod
    def backward(ctx):
        raise RuntimeError('backward of TransformerLayer is not supported')


def fuse_conv_bn(conv, bn, mode='conv1d'):
    '''fuse conv and bn weight.'''
    w = conv.weight
    b = w.new_zeros(w.size(0)) if conv.bias is None else conv.bias
    mean = bn.running_mean

    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    if mode == 'conv1d':
        w_scale = (bn.weight / var_sqrt).reshape([conv.out_channels, 1, 1])
    elif mode == 'conv2d':
        w_scale = (bn.weight / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    else:
        raise RuntimeError('fuse_conv_bn: unsupport mode', mode)
    w = w * w_scale
    b = (b - mean) / var_sqrt * bn.weight + bn.bias
    return w, b


class PantherConformerV1(torch.autograd.Function):
    """PantherConformerFunction"""

    # pylint: disable=unused-argument,too-many-locals
    @staticmethod
    def symbolic(
        g,
        x,
        key_padding_mask,
        pos_rel,
        ffn1_ln,
        ffn1_lw,
        ffn1_lb,
        ffn1_pw,
        ffn1_pb,
        attn_ln,
        linear_pos,
        attn_in_pw,
        attn_in_pb,
        attn_out_pw,
        attn_out_pb,
        attn_pos_bias_uv,
        conv_ln,
        conv_gate_w,
        conv_gate_b,
        conv_w,
        conv_b,
        conv2_ln,
        conv_pw,
        conv_pb,
        ffn2_ln,
        ffn2_lw,
        ffn2_lb,
        ffn2_pw,
        ffn2_pb,
        out_ln,
        x_format='BTN',
        y_format='BTN',
        num_heads=0,
        embed_dim=0,
        linear_dim=0,
        ffn1_ln_eps=1e-5,
        attn_ln_eps=1e-5,
        conv_ln_eps=1e-5,
        conv2_ln_eps=1e-5,
        ffn2_ln_eps=1e-5,
        ffn1_activation='',
        conv_activation='',
        ffn2_activation='',
        attn_l_context=10000,
        attn_r_context=10000,
        conv_l_context=1,
        conv_r_context=1,
    ):
        """symbolic"""
        if key_padding_mask is None:
            key_padding_mask = unused(g)

        if pos_rel is None:
            pos_rel = unused(g)

        if conv2_ln is None:
            conv2_ln = unused(g)

        attrs = {
            'x_format_s': x_format,
            'y_format_s': y_format,
            'num_heads_i': num_heads,
            'embed_dim_i': embed_dim,
            'linear_dim_i': linear_dim,
            'ffn1_ln_eps_f': ffn1_ln_eps,
            'attn_ln_eps_f': attn_ln_eps,
            'conv_ln_eps_f': conv_ln_eps,
            'conv2_ln_eps_f': conv2_ln_eps,
            'ffn2_ln_eps_f': ffn2_ln_eps,
            'ffn1_activation_s': ffn1_activation,
            'conv_activation_s': conv_activation,
            'ffn2_activation_s': ffn2_activation,
            'attn_left_context_i': attn_l_context,
            'attn_right_context_i': attn_r_context,
            'conv_left_kernel_size_i': conv_l_context,
            'conv_right_kernel_size_i': conv_r_context,
        }

        return g.op(
            "ConformerLayerV1",
            x,
            key_padding_mask,
            pos_rel,
            ffn1_ln,
            ffn1_lw,
            ffn1_lb,
            ffn1_pw,
            ffn1_pb,
            attn_ln,
            linear_pos,
            attn_in_pw,
            attn_in_pb,
            attn_out_pw,
            attn_out_pb,
            attn_pos_bias_uv,
            conv_ln,
            conv_gate_w,
            conv_gate_b,
            conv_w,
            conv_b,
            conv2_ln,
            conv_pw,
            conv_pb,
            ffn2_ln,
            ffn2_lw,
            ffn2_lb,
            ffn2_pw,
            ffn2_pb,
            out_ln,
            **attrs,
        )

    @staticmethod
    def forward(ctx, x, *args, **kwargs):
        # fake output, the shape and dtype must match the real ouput!
        return x

    @staticmethod
    def backward(ctx):
        raise RuntimeError('backward of ConformerLayer is not supported')


def panther_conformer(
    x,
    key_padding_mask,
    pos_emb,
    ffn1_ln,
    ffn1,
    attn_ln,
    attn,
    conv_ln,
    conv,
    ffn2_ln,
    ffn2,
    out_ln,
    activation_fn='',
    num_heads=0,
    embed_dim=0,
    linear_dim=0,
    ff_scale=1.0,
):
    '''panther conformer.'''
    conv2_ln = None
    conv2_ln_eps = 1e-5
    fused_dconv = [conv.depthwise_conv_weight, conv.depthwise_conv_bias]
    if conv.norm is not None and 'LayerNorm' in str(conv.norm.norm.__class__):
        conv2_ln = torch.cat([conv.norm.weight, conv.norm.bias], dim=-1).reshape(-1)
        conv2_ln_eps = conv.norm.eps
    else:
        fused_dconv = fuse_conv_bn(conv.depthwise_conv, conv.norm, mode='conv1d')

    activation_fn = activation_fn.capitalize()
    assert activation_fn in ('Relu', 'Gelu')
    assert attn.linear_pos.bias is None
    return PantherConformerV1.apply(
        x,
        key_padding_mask,
        pos_emb,
        torch.cat([ffn1_ln.weight, ffn1_ln.bias], dim=-1).reshape(-1),
        ffn1.w_1.weight.transpose(0, 1).contiguous(),
        ffn1.w_1.bias,
        ffn1.w_2.weight.transpose(0, 1).contiguous() * ff_scale,
        ffn1.w_2.bias * ff_scale,
        torch.cat([attn_ln.weight, attn_ln.bias], dim=-1).reshape(-1),
        attn.linear_pos.weight.transpose(0, 1).contiguous(),
        attn.in_proj.weight.transpose(0, 1).contiguous(),
        attn.in_proj.bias,
        attn.out_proj.weight.transpose(0, 1).contiguous(),
        attn.out_proj.bias,
        torch.cat([attn.pos_bias_u, attn.pos_bias_v], dim=0).reshape(-1),
        torch.cat([conv_ln.weight, conv_ln.bias], dim=-1).reshape(-1),
        conv.pointwise_conv1.weight.transpose(0, 1).contiguous(),
        conv.pointwise_conv1.bias,
        fused_dconv[0].squeeze(1).transpose(0, 1).contiguous(),
        fused_dconv[1],
        conv2_ln,
        conv.pointwise_conv2.weight.transpose(0, 1).contiguous(),
        conv.pointwise_conv2.bias,
        torch.cat([ffn2_ln.weight, ffn2_ln.bias], dim=-1).reshape(-1),
        ffn2.w_1.weight.transpose(0, 1).contiguous(),
        ffn2.w_1.bias,
        ffn2.w_2.weight.transpose(0, 1).contiguous() * ff_scale,
        ffn2.w_2.bias * ff_scale,
        torch.cat([out_ln.weight, out_ln.bias], dim=-1).reshape(-1),
        'BTN',
        'BTN',
        num_heads,
        embed_dim,
        linear_dim,
        ffn1_ln.eps,
        attn_ln.eps,
        conv_ln.eps,
        conv2_ln_eps,
        ffn2_ln.eps,
        activation_fn,
        activation_fn,
        activation_fn,
        attn.attn_left_kernel_size,
        attn.attn_right_kernel_size,
        conv.depthwise_conv_left_kernel_size,
        conv.depthwise_conv_right_kernel_size,
    )
