import torch
import sys

def get_step_epoch_from_ckpt(ckpt_path):
    data = torch.load(ckpt_path)
    global_step = data["global_step"]
    epoch = data["epoch"]
    return global_step, epoch

ckpt_path = sys.argv[1]
step, epoch = get_step_epoch_from_ckpt(ckpt_path)
print(f"{step//1000}k_epoch{epoch}")
