import torch
from torch import nn
import sys, os
import math
import numpy as np

def _repara(stats):
    m, logs = torch.split(stats, 32, dim=-1)
    z = m + torch.randn_like(m) * torch.exp(logs)
    return z

ckpt_path = sys.argv[1]
bn_dir = sys.argv[2]
out_dir = sys.argv[3]
suffixes = sys.argv[4]

os.makedirs(out_dir, exist_ok=True)

prenet = nn.Linear(32, 1536, bias=False)
torch.nn.init.normal_(
    prenet.weight, mean=0.0, std=0.02 / math.sqrt(2 * 1)
)
state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
new_state_dict = []
new_state_dict = {k.replace("model.prenet.",""):v for k, v in state_dict.items() if 'prenet.weight' in k}
prenet.load_state_dict(new_state_dict, strict=True)

bn_names = [x for x in os.listdir(bn_dir) if x.endswith('_bn.npy')]
for bn_name in bn_names:
    basename = bn_name[:-7]
    bn_path = os.path.join(bn_dir, bn_name)
    bn = np.load(bn_path)
    bn = torch.from_numpy(bn)
    bn_in_z = _repara(bn)
    bn_in_h = prenet(bn_in_z)

    out_path = os.path.join(out_dir, basename + '_{}.npy'.format(suffixes))
    np.save(out_path, bn_in_h.detach().numpy())