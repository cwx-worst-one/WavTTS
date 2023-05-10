"""solutions utils"""
import torch


@torch.no_grad()
def rnnt_rlt_neaten(raw_rlt, keep_tensor=False):
    '''neaten inference result by removing blank'''
    if isinstance(raw_rlt, torch.Tensor):
        neaten_rlt = raw_rlt[raw_rlt != 0]
        if keep_tensor:
            return neaten_rlt.view(-1)
        return neaten_rlt.view(-1).cpu().tolist()
    neaten_rlt = []
    for item in raw_rlt:
        if item != 0:
            neaten_rlt.append(item)
    return neaten_rlt
