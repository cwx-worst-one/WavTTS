# CREDITS: Taken largely from Karpathy's excellent notebook:
# https://github.com/karpathy/nanoGPT/blob/master/transformer_sizing.ipynb

from collections import OrderedDict


def palm_flops(config, seq_len: int, N: int):
    """estimate of the model flops following PaLM paper formula"""
    # non-embedding model parameters. note that we do not subtract the
    # embedding/token params because those are tied and get used in the last layer.
    L, H, Q, T = config.n_layer, config.n_head, config.n_embd // config.n_head, seq_len
    mf_per_token = 6 * N + 12 * L * H * Q * T
    mf = mf_per_token * seq_len
    return mf


def flops(config, vocab_size: int, seq_len: int):
    """We only count Weight FLOPs, all other layers (LayerNorm, Softmax, etc)
    are effectively irrelevant.

    We count actual FLOPs, not MACs. Hence 2* all over the place
    Basically for any matrix multiply:
    A (BxC) @ B (CxD) -> (BxD) flops are 2*B*C*D

    Args:
        config (_type_): _description_
        vocab_size (int): _description_
        seq_len (int): _description_

    Returns:
        _type_: _description_
    """

    out = OrderedDict()
    head_size = config.n_embd // config.n_head

    # attention blocks
    # 1) the projection to key, query, values
    out["attention/kqv"] = 2 * seq_len * (config.n_embd * 3 * config.n_embd)
    # 2) calculating the attention scores
    out["attention/scores"] = 2 * seq_len * seq_len * config.n_embd
    # 3) the reduction of the values (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    out["attention/reduce"] = 2 * config.n_head * (seq_len * seq_len * head_size)
    # 4) the final linear projection
    out["attention/proj"] = 2 * seq_len * (config.n_embd * config.n_embd)
    out["attention"] = sum(
        out["attention/" + k] for k in ["kqv", "scores", "reduce", "proj"]
    )

    # MLP blocks
    ffw_size = config.n_inner  # feed forward size
    out["mlp/ffw1"] = 2 * seq_len * (config.n_embd * ffw_size)
    out["mlp/ffw2"] = 2 * seq_len * (ffw_size * config.n_embd)
    out["mlp"] = out["mlp/ffw1"] + out["mlp/ffw2"]

    # the transformer and the rest of it
    out["block"] = out["attention"] + out["mlp"]
    out["transformer"] = config.n_layer * out["block"]
    out["dense"] = 2 * seq_len * (config.n_embd * vocab_size)

    # forward,backward,total
    out["forward_total"] = out["transformer"] + out["dense"]
    out["backward_total"] = (
        2 * out["forward_total"]
    )  # use common estimate of bwd = 2*fwd
    out["total"] = out["forward_total"] + out["backward_total"]

    return out


def flops_achieved(measured_time: float, batch_size: int, total_flops: int):
    measured_throughput = batch_size / measured_time
    return total_flops * measured_throughput
