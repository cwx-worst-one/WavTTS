import math

import torch
from einops import rearrange

ELEMWISE_WINDOW_MASK = "elemwise"
BLOCKWISE_WINDOW_MASK = "blockwise"
WINDOW_MASK_TYPES = {ELEMWISE_WINDOW_MASK: 0, BLOCKWISE_WINDOW_MASK: 1}


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

    # if window_size[0] < 0:
    #     window_size[0] = 1000000
    # if window_size[1] < 0:
    #     window_size[1] = 1000000
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
                (
                    (row_idx + 1 + sk - sq + window_size[1])
                    if window_type == ELEMWISE_WINDOW_MASK
                    else (
                        row_idx
                        + sk
                        - sq
                        + window_size[1]
                        - (row_idx + torch.maximum(sk - sq, torch.zeros_like(sk)))
                        % window_size[1]
                    )
                ),
                sk,
            ),
            col_idx
            < torch.maximum(
                (
                    (row_idx + sk - sq - window_size[0])
                    if window_type == ELEMWISE_WINDOW_MASK
                    else (
                        row_idx
                        + sk
                        - sq
                        - window_size[0]
                        - (row_idx + torch.maximum(sk - sq, torch.zeros_like(sk)))
                        % window_size[1]
                    )
                ),
                torch.zeros_like(sk),
            ),
        )
    return local_mask


def ctiga_fwd_local_block_mask(
    seqlen_q,
    seqlen_k,
    window_size=[-1, -1],  # -1 means infinite window size
    window_type=ELEMWISE_WINDOW_MASK,
    block_size=[128, 128],
    device=None,
    local_mask=None,
):
    if window_type not in WINDOW_MASK_TYPES:
        raise ValueError(
            f"no support window_type '{window_type}'(expect {WINDOW_MASK_TYPES})"
        )
    window_size = [1000000 if s == -1 else s for s in window_size]
    m_block = math.ceil(seqlen_q / block_size[0])
    n_block = math.ceil(seqlen_k / block_size[1])
    block_mask = torch.zeros(m_block, n_block, dtype=torch.int, device=device)

    for mi in range(m_block):
        n_block_min = max(
            0,
            (
                (mi * block_size[0] + seqlen_k - seqlen_q - window_size[0])
                if window_type == ELEMWISE_WINDOW_MASK
                else (
                    (mi * block_size[0] + max(0, seqlen_k - seqlen_q))
                    // window_size[1]
                    * window_size[1]
                    + min(0, seqlen_k - seqlen_q)
                    - window_size[0]
                )
            )
            / block_size[1],
        )
        n_block_max = min(
            n_block,
            math.ceil(
                (
                    ((mi + 1) * block_size[0] + seqlen_q - seqlen_k + window_size[1])
                    if window_type == ELEMWISE_WINDOW_MASK
                    else (
                        math.ceil(
                            ((mi + 1) * block_size[0] + max(0, seqlen_k - seqlen_q))
                            / window_size[1]
                        )
                        * window_size[1]
                        + min(0, seqlen_k - seqlen_q)
                    )
                )
                / block_size[1]
            ),
        )
        print(f"FWD: {(mi, int(n_block_min), int(n_block_max))=}")
        block_mask[mi, int(n_block_min) : int(n_block_max)] += 1

    if local_mask is not None:
        ref_block_mask = torch.zeros_like(block_mask)
        for mi in range(m_block):
            for ni in range(n_block):
                row_left = mi * block_size[0]
                row_right = min(row_left + block_size[0], seqlen_q)
                col_left = ni * block_size[1]
                col_right = min(col_left + block_size[1], seqlen_k)
                if local_mask[row_left:row_right, col_left:col_right].sum() < (
                    row_right - row_left
                ) * (col_right - col_left):
                    ref_block_mask[mi : mi + 1, ni : ni + 1] += 1

        assert (
            ref_block_mask - block_mask
        ).abs().sum() == 0, f"\n{ref_block_mask=}\n{block_mask=}"
        print("\n")
    return block_mask


def ctiga_bwd_local_block_mask(
    seqlen_q,
    seqlen_k,
    window_size=[-1, -1],  # -1 means infinite window size
    window_type=ELEMWISE_WINDOW_MASK,
    block_size=[128, 128],
    device=None,
    local_mask=None,
):
    if window_type not in WINDOW_MASK_TYPES:
        raise ValueError(
            f"no support window_type '{window_type}'(expect {WINDOW_MASK_TYPES})"
        )
    window_size = [1000000 if s == -1 else s for s in window_size]
    m_block = math.ceil(seqlen_q / block_size[0])
    n_block = math.ceil(seqlen_k / block_size[1])
    block_mask = torch.zeros(m_block, n_block, dtype=torch.int, device=device)

    # bwd
    do_apply_mask_res = torch.zeros(m_block, n_block, dtype=torch.int)
    for ni in range(n_block):
        m_block_min = max(
            0,
            (
                (ni * block_size[1] + seqlen_q - seqlen_k - window_size[1])
                if window_type == ELEMWISE_WINDOW_MASK
                else (
                    (ni * block_size[1] + max(0, seqlen_q - seqlen_k))
                    // window_size[1]
                    * window_size[1]
                    + min(0, seqlen_q - seqlen_k)
                )
            )
            // block_size[0],
        )
        m_block_max = min(
            m_block,
            math.ceil(
                (
                    ((ni + 1) * block_size[1] + seqlen_q - seqlen_k + window_size[0])
                    if window_type == ELEMWISE_WINDOW_MASK
                    else (
                        math.ceil(
                            (
                                (ni + 1) * block_size[1]
                                + max(0, seqlen_q - seqlen_k)
                                + window_size[0]
                            )
                            / window_size[1]
                        )
                        * window_size[1]
                        + min(0, seqlen_q - seqlen_k)
                    )
                )
                / block_size[0]
            ),
        )
        print(f"BWD: {(ni, int(m_block_min), int(m_block_max))=}")
        block_mask[int(m_block_min) : int(m_block_max), ni] += 1
        for mi in range(m_block_min, m_block_max):
            cond1 = mi * block_size[0] < (
                (ni + 1) * block_size[1] + max(0, seqlen_q - seqlen_k)
            ) // window_size[1] * window_size[1] + min(0, seqlen_q - seqlen_k)
            cond2 = (mi + 1) * block_size[0] > math.ceil(
                (ni * block_size[1] + max(0, seqlen_q - seqlen_k) + window_size[0])
                / window_size[1]
            ) * window_size[1] + min(0, seqlen_q - seqlen_k)
            cond3 = (ni + 1) * block_size[1] >= seqlen_k
            do_apply_mask = cond1 or cond2 or cond3
            if ni < 3:
                print(f"{(mi,ni,cond1,cond2,cond3)=}")
            do_apply_mask_res[mi, ni] = int(do_apply_mask)
    print(f"{block_mask=}")
    print(f"{do_apply_mask_res=}")
    if local_mask is not None:
        ref_block_mask = torch.zeros_like(block_mask)
        for mi in range(m_block):
            for ni in range(n_block):
                row_left = mi * block_size[0]
                row_right = min(row_left + block_size[0], seqlen_q)
                col_left = ni * block_size[1]
                col_right = min(col_left + block_size[1], seqlen_k)
                if local_mask[row_left:row_right, col_left:col_right].sum() < (
                    row_right - row_left
                ) * (col_right - col_left):
                    ref_block_mask[mi : mi + 1, ni : ni + 1] += 1

        assert (
            ref_block_mask - block_mask
        ).abs().sum() == 0, f"\n{ref_block_mask=}\n{block_mask=}"
        print("\n")
    return block_mask


def construct_ctiga_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=[-1, -1],  # -1 means infinite window size
    window_type=ELEMWISE_WINDOW_MASK,
    fwd_block_size=(128, 128),
    bwd_block_size=(128, 128),
    device=None,
):
    window_size = list(window_size)
    if window_type not in WINDOW_MASK_TYPES:
        raise ValueError(
            f"no support window_type '{window_type}'(expect {WINDOW_MASK_TYPES})"
        )
    window_size = [1000000 if s == -1 else s for s in window_size]
    local_mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.int, device=device)
    for row_idx in range(seqlen_q):
        n_limit_left = max(
            0,
            (
                (row_idx + seqlen_k - seqlen_q - window_size[0])
                if window_type == ELEMWISE_WINDOW_MASK
                else (
                    row_idx
                    + seqlen_k
                    - seqlen_q
                    - window_size[0]
                    - (row_idx + max(0, seqlen_k - seqlen_q)) % window_size[1]
                )
            ),
        )
        n_limit_right = min(
            seqlen_k,
            (
                (row_idx + 1 + seqlen_k - seqlen_q + window_size[1])
                if window_type == ELEMWISE_WINDOW_MASK
                else (
                    row_idx
                    + seqlen_k
                    - seqlen_q
                    + window_size[1]
                    - (row_idx + max(0, seqlen_k - seqlen_q)) % window_size[1]
                )
            ),
        )
        # print(f"{(row_idx, n_limit_left, n_limit_right)=}")
        for col_idx in range(seqlen_k):
            if col_idx < n_limit_left or col_idx >= n_limit_right:
                local_mask[row_idx, col_idx] = 1

    print(f"{local_mask}")
    print("\n")
    fwd_local_block_mask = ctiga_fwd_local_block_mask(
        seqlen_q, seqlen_k, window_size, window_type, fwd_block_size, device, local_mask
    )
    bwd_local_block_mask = ctiga_bwd_local_block_mask(
        seqlen_q, seqlen_k, window_size, window_type, bwd_block_size, device, local_mask
    )
    print(f"{fwd_local_block_mask}")
    print(f"{bwd_local_block_mask}")
    return local_mask


if __name__ == "__main__":
    torch.set_printoptions(linewidth=200, edgeitems=20)
    seqlen_q = 245
    seqlen_k = 124
    window = [48, 61]
    headdim = 80
    from samantha.components.ctiga.ops.flash_attn_2_3_interface import (
        _get_block_size as _get_fwd_block_size,
    )
    from samantha.components.ctiga.ops.flash_attn_2_3_interface import (
        _get_bwd_block_size,
    )

    fwd_block_size = _get_fwd_block_size("cuda", headdim, False, False)
    bwd_block_size = _get_bwd_block_size("cuda", headdim, False, False)
    # block_size=[4,4]
    print(
        f"{(seqlen_q,seqlen_k)=}, {window=}, "
        f"{fwd_block_size=}, {bwd_block_size=}, {headdim=}"
    )
    torch.testing.assert_close(
        construct_local_mask(seqlen_q, seqlen_k, window, BLOCKWISE_WINDOW_MASK).int(),
        construct_ctiga_local_mask(
            seqlen_q,
            seqlen_k,
            window,
            BLOCKWISE_WINDOW_MASK,
            fwd_block_size=fwd_block_size,
            bwd_block_size=bwd_block_size,
        ).int(),
    )
