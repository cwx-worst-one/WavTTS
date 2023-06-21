from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

save_path = "/mnt/bn/zongyu-lq/logs/best_rq/mcc40m_511m_40a100s_lr1e-4/checkpoints/step=011000-tr_loss=3.3906-val_loss_0=3.2432.ckpt"
output_path = "/mnt/bn/audio-diffusion/ckpts/best_rq/step=011000.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)