import numpy as np


def log_softmax(values, axis=1):

    row_max = values.max(axis=axis)

    row_max = row_max.reshape(-1, 1)
    values = values - row_max

    values_exp = np.exp(values)
    values_sum = np.sum(values_exp, axis=axis, keepdims=True)
    res = values_exp / values_sum
    # for np.log, next step no use res[:, 0]
    # value when gather no_blank_score
    res[:, 0] = 1
    res = np.log(res)
    return res


def get_log_prob_noblk(
    jointer_hidden, adapt_softmax_thresh=0.995, only_head=False, rnnt_temperature=1.0
):

    '''get the log probs
    Args:
        jointer_hidden: [B, T, U, N], the output of log_softmax(jointer)
        adapt_softmax_thresh: for RNNTAdaptiveSoftmax based log_softmax_fc
        only_head: for RNNTAdaptiveSoftmax based log_softmax_fc
    Return:
        the log prob with shape [B, T, U, V]
    '''
    jointer_hidden[:, 0] = float('-inf')
    log_softmax_out = log_softmax(jointer_hidden)
    return log_softmax_out
