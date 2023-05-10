"""test bert layer"""
import torch
from core.models.layers.transformer import BiTransformerLayer
from core.models.pretrained.bert_model import BertLayer
from core.utils import FalconDict


def _compatible_load_hook(
    state_dict,
    prefix,
    _local_metadata,
    _strict,
    _missing_keys,
    _unexpected_keys,
    _error_msgs,
):
    '''
    compatible for load previous checkpoint.
    '''
    for wb in ["weight", "bias"]:
        # self attn qkv
        for name, new_name in zip(["query", "key", "value"], ['q_proj', 'k_proj', 'v_proj']):
            val = state_dict.pop(prefix + "attention.self.{}.{}".format(name, wb))
            if val is not None:
                state_dict[prefix + "self_attn.{}.{}".format(new_name, wb)] = val
        # self attn out proj
        val = state_dict.pop(prefix + "attention.output.dense.{}".format(wb))
        if val is not None:
            state_dict[prefix + "self_attn.out_proj_{}".format(wb)] = val
        # self_attn_layer_norm
        val = state_dict.pop(prefix + "attention.output.LayerNorm.{}".format(wb))
        if val is not None:
            state_dict[prefix + "self_attn_layer_norm.{}".format(wb)] = val
        # fc1
        val = state_dict.pop(prefix + "intermediate.dense.{}".format(wb))
        if val is not None:
            state_dict[prefix + "fc1.{}".format(wb)] = val
        # fc2
        val = state_dict.pop(prefix + "output.dense.{}".format(wb))
        if val is not None:
            state_dict[prefix + "fc2.{}".format(wb)] = val
        # final layernorm
        val = state_dict.pop(prefix + "output.LayerNorm.{}".format(wb))
        if val is not None:
            state_dict[prefix + "final_layer_norm.{}".format(wb)] = val


def test_bert_layer():
    """test bert layer"""
    args = FalconDict()

    args.hidden_size = 256
    args.num_attention_heads = 8
    args.intermediate_size = 1024
    args.attention_probs_dropout_prob = 0.0
    args.hidden_dropout_prob = 0.0
    args.hidden_act = "gelu"
    args.layer_norm_eps = 1e-8

    args.chunk_size_feed_forward = 0
    args.is_decoder = False
    args.add_cross_attention = False

    bert = BertLayer(args)
    bi = BiTransformerLayer(
        embed_dim=args.hidden_size,
        attention_heads=args.num_attention_heads,
        ffn_embed_dim=args.intermediate_size,
        attention_dropout=args.attention_probs_dropout_prob,
        hidden_dropout=args.hidden_dropout_prob,
        activation_dropout=0.0,
        activation=args.hidden_act,
        normalize_before=False,
        clamp_inf=False,
        squeeze_mem=False,
        layernorm_eps=args.layer_norm_eps,
    )
    bert.cuda()
    bi.cuda()

    # pylint:disable=protected-access
    bi._register_load_state_dict_pre_hook(_compatible_load_hook)
    bi.load_state_dict(bert.state_dict())

    bsz = 8
    seq_len = 80
    emb = 256

    x = torch.nn.Parameter(torch.randn(bsz, seq_len, emb)).cuda()
    mask = torch.randint(low=0, high=2, size=(bsz, seq_len)).cuda().float()
    extended_attention_mask = mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    x1 = x.clone().detach_().requires_grad_()

    y = bert(hidden_states=x, attention_mask=extended_attention_mask)
    extended_attention_mask = extended_attention_mask[:, 0, 0, :]
    extended_attention_mask = extended_attention_mask / -10000.0
    y1 = bi(
        x=x1,
        encoder_padding_mask=extended_attention_mask.bool(),
        output_layer_result=True,
        fused=True,
        batch_first=True,
    )

    assert torch.allclose(y[0], y1[0], atol=5e-4)

    dy = torch.rand_like(y[0]).cuda()

    x.retain_grad()
    x1.retain_grad()

    y[0].backward(dy)
    y1[0].backward(dy)

    assert torch.allclose(x.grad.data, x1.grad.data, atol=3e-4)


if __name__ == '__main__':
    test_bert_layer()
