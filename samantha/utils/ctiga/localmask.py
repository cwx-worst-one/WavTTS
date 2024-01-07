import torch
from einops import rearrange

ELEMWISE_WINDOW_MASK = "elemwise"
BLOCKWISE_WINDOW_MASK = "blockwise"
WINDOW_MASK_TYPES = {ELEMWISE_WINDOW_MASK: 0, BLOCKWISE_WINDOW_MASK: 1}


def get_window_mask_typename(type_id):
    return {0: ELEMWISE_WINDOW_MASK, 1: BLOCKWISE_WINDOW_MASK}[type_id]


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=[-1, -1],  # -1 means infinite window size
    window_type=ELEMWISE_WINDOW_MASK,
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    window_size = list(window_size)
    if window_type not in WINDOW_MASK_TYPES:
        raise ValueError(
            f"no support window_type '{window_type}'(expect {WINDOW_MASK_TYPES})"
        )
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )

    if window_size[0] < 0:
        window_size[0] = 1000000
    if window_size[1] < 0:
        window_size[1] = 1000000
    if window_size[0] < 0:
        local_mask = col_idx >= (
            (row_idx + 1 + sk - sq + window_size[1])
            if window_type == ELEMWISE_WINDOW_MASK
            else (row_idx + sk - sq + window_size[1] - (row_idx) % window_size[1])
        )
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        local_mask = torch.logical_or(
            col_idx
            >= torch.minimum(
                (row_idx + 1 + sk - sq + window_size[1])
                if window_type == ELEMWISE_WINDOW_MASK
                else (row_idx + sk - sq + window_size[1] - (row_idx) % window_size[1]),
                sk,
            ),
            col_idx < row_idx + sk - sq - window_size[0],
        )
    # print_tensor(local_mask[None,None,:,:],"<<<local_mask>>>")
    return local_mask


if __name__ == "__main__":
    masks = {
        "no_mask": construct_local_mask(12, 12, (-1, -1), ELEMWISE_WINDOW_MASK),
        "causal_mask": construct_local_mask(12, 12, (-1, 0), ELEMWISE_WINDOW_MASK),
        "elemwise_window": construct_local_mask(12, 12, (2, 3), ELEMWISE_WINDOW_MASK),
        "block_window": construct_local_mask(12, 12, (-1, 3), BLOCKWISE_WINDOW_MASK),
    }
    for m_cls, mask in masks.items():
        print(f"{m_cls}\n{mask.int()}")
