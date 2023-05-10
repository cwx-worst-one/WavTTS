'''
test rnnt alignment.
'''
import numpy as np
import torch
from core.extensions import rnnt_force_alignment


def test_rnnt_force_alignment():
    '''
    test rnnt force alignment.
    '''

    bsz, frame_len, tgt_len = 3, 217, 35
    lprobs = torch.rand([bsz, frame_len, tgt_len, 2])
    frame_lens = torch.randint(2, frame_len + 1, [bsz]).long()
    tgt_lens = torch.randint(2, tgt_len + 1, [bsz]).long()

    # org func
    rnnt_probs = lprobs.cpu().detach().numpy()
    input_lengths = frame_lens.cpu().tolist()
    target_lengths = tgt_lens.cpu().tolist()
    bp_trace = np.zeros((bsz, frame_len, tgt_len))
    alpha = np.ones((bsz, frame_len, tgt_len)) * float('-inf')
    path = torch.ones((bsz, frame_len, tgt_len))
    # init
    bp_trace[:, 0, 0] = -1
    alpha[:, 0, 0] = 0
    # edge
    for t in range(1, frame_len):
        bp_trace[:, t, 0] = 0
        alpha[:, t, 0] = alpha[:, t - 1, 0] + rnnt_probs[:, t - 1, 0, 0]
    for u in range(1, tgt_len):
        bp_trace[:, 0, u] = 1
        alpha[:, 0, u] = alpha[:, 0, u - 1] + rnnt_probs[:, 0, u - 1, 1]
    # fill
    for t in range(1, frame_len):
        for u in range(1, tgt_len):
            blk_trans = alpha[:, t - 1, u] + rnnt_probs[:, t - 1, u, 0]
            nonblk_trans = alpha[:, t, u - 1] + rnnt_probs[:, t, u - 1, 1]
            alpha[:, t, u] = np.maximum(blk_trans, nonblk_trans)
            bp_trace[:, t, u] = (blk_trans < nonblk_trans).astype('int')
    my_aligns = torch.zeros((bsz, tgt_len)).long()
    # trace
    for b in range(bsz):
        i = input_lengths[b] - 1
        j = target_lengths[b] - 1
        path[0][i][j] = 1
        while bp_trace[b][i][j] >= 0:
            if bp_trace[b][i][j]:
                my_aligns[b, j] = i
                j -= 1
                # path[b][max(i - 1, 0)][j] = float(slf_ali)
            else:
                i -= 1

    f_aligns = rnnt_force_alignment(lprobs, frame_lens, tgt_lens)
    assert torch.equal(my_aligns, f_aligns.cpu())
