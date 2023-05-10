'''
functions for label smoothin
'''
import numpy as np
import torch.nn.functional as F


def uniform_label_smooth(confidence, num_labels, targets, logits, eps=1e-20):
    '''uniform label smooth'''
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / float(num_labels - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
        confidence * np.log(confidence)
        + float(num_labels - 1) * low_confidence * np.log(low_confidence + eps)
    )

    onehot_targets = F.one_hot(targets.long(), num_labels).float()
    soft_targets = -(onehot_targets * confidence + (1 - onehot_targets) * low_confidence)
    ce_lprob = F.log_softmax(logits, dim=-1)
    smoothed_ce_loss = (soft_targets * ce_lprob).sum(-1) - normalizing

    return ce_lprob, smoothed_ce_loss
