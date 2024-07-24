from byteformers.components.attention import MultiHeadAttention
from torch.nn import MultiheadAttention as MultiheadAttentionPT


def convert_pytorch_to_byteformers_attention_weights(
    pt_mha: MultiheadAttentionPT, bf_mha: MultiHeadAttention
) -> MultiHeadAttention:
    pt_state_dict = pt_mha.state_dict()
    d_model = bf_mha.d_model
    compatible_dict = {}
    compatible_dict["to_q.weight"] = pt_state_dict["in_proj_weight"][:d_model]
    compatible_dict["to_k.weight"] = pt_state_dict["in_proj_weight"][
        d_model : d_model * 2
    ]
    compatible_dict["to_v.weight"] = pt_state_dict["in_proj_weight"][d_model * 2 :]
    compatible_dict["to_q.bias"] = pt_state_dict["in_proj_bias"][:d_model]
    compatible_dict["to_k.bias"] = pt_state_dict["in_proj_bias"][d_model : d_model * 2]
    compatible_dict["to_v.bias"] = pt_state_dict["in_proj_bias"][d_model * 2 :]
    compatible_dict["WO.weight"] = pt_state_dict["out_proj.weight"]
    compatible_dict["WO.bias"] = pt_state_dict["out_proj.bias"]
    bf_mha.load_state_dict(compatible_dict)
    return bf_mha
