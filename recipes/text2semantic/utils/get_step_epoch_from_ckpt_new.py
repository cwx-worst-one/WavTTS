import torch
import sys

def get_step_epoch_from_ckpt(ckpt_path):
    data = torch.load(ckpt_path)
    global_step = data["global_step"]
    epoch = data["epoch"]
    return global_step, epoch

ckpt_path = sys.argv[1]
out_path = sys.argv[2]

f_w = open(out_path, 'w')

step, epoch = get_step_epoch_from_ckpt(ckpt_path)

out_str = "step" + str(step // 1000) + "k_epoch" + str(epoch)
f_w.write(out_str)
