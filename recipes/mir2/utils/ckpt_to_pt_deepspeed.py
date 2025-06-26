from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

convert_zero_checkpoint_to_fp32_state_dict('/mnt/bn/mir-tasks/ju-chiang.wang/MusicFM/25Hz_SSTK_330M/checkpoints/last.ckpt', '/mnt/bn/mir-tasks/ju-chiang.wang/MusicFM/pretrained/25Hz_SSTK_330M_580k.pt')