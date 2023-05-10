"""data2vec criterion"""

import math
import torch
import torch.nn.functional as F
from core.utils import get_world_size, dist_allreduce, ReduceOp

# pylint:disable=unused-argument
def data2vec_criterion(
    x,
    y,
    loss_beta,
    loss_scale,
    result,
    is_training,
    num_updates,
    min_target_var,
    min_pred_var,
    ema,
    features_pen,
    batch_data,
    coe,
):
    """data2vec criterion"""
    sz = x.size(-1)

    if loss_beta == 0:
        loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)  # (bsz, tsz)
    else:
        loss = F.smooth_l1_loss(
            x.float(), y.float(), reduction="none", beta=loss_beta  # (bsz, tsz)
        ).sum(dim=-1)

    if loss_scale >= 0.0:
        scale = loss_scale
    else:
        scale = 1 / math.sqrt(sz)

    sample_size = loss.numel()
    result["backward_loss"] = loss.sum() * scale / sample_size

    with torch.no_grad():
        allreduce_flag = is_training and num_updates % 100 == 0
        result["target_var"] = compute_var(y, allreduce_flag)
        result["pred_var"] = compute_var(x.float(), allreduce_flag)

    if num_updates > 5000 and num_updates % 100 == 0 and result["target_var"] < min_target_var:
        raise Exception(f"target var is {result['target_var'].item()} < {min_target_var}, exiting")
    if num_updates > 5000 and num_updates % 100 == 0 and result["pred_var"] < min_pred_var:
        raise Exception(f"pred var is {result['pred_var'].item()} < {min_pred_var}, exiting")

    if ema is not None and hasattr(ema, 'get_decay'):
        result["ema_decay"] = ema.get_decay() * 1000

    # logging
    result["feature_l2_loss"] = features_pen
    result["loss"] = result["backward_loss"]
    result["frame_size"] = batch_data['src_mask'].float().sum()
    result["tgt_size"] = sample_size
    result["batch_size"] = sample_size
    if coe is not None:
        coe = coe.view(-1)
        result["lw_mean"] = torch.mean(coe)
        result["lw_var"] = torch.var(coe)
        result["lw_max"] = torch.max(coe)
        result["lw_last"] = coe[-1]

    return result


def compute_var(y, is_training=True):
    '''compute_var'''
    # Note: for dev set, each worker will process all assigned batches (batch number may vary),
    #      To avoid hanging, we didn't apply allreduce for dev data
    y = y.view(-1, y.size(-1))
    if get_world_size() > 1 and is_training:
        zc = torch.tensor(y.size(0)).cuda()
        zs = y.sum(dim=0)
        zss = (y**2).sum(dim=0)

        dist_allreduce(zc, name='data2vec_zc', op=ReduceOp.SUM)
        dist_allreduce(zs, name='data2vec_zs', op=ReduceOp.SUM)
        dist_allreduce(zss, name='data2vec_zss', op=ReduceOp.SUM)

        var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
        return torch.sqrt(var + 1e-6).mean()
    return torch.sqrt(y.var(dim=0) + 1e-6).mean()
