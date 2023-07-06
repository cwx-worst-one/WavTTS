from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

save_path = "/mnt/bn/zongyu-lq/logs/best_rq/pgc600k_25hz_4x8192_522m/checkpoints/step=085000-tr_loss=4.5238-val_loss_0=4.1285.ckpt"
output_path = "/mnt/bn/zongyu-lq/logs/best_rq/pgc600k_25hz_4x8192_522m/checkpoints/step=085000-tr_loss=4.5238-val_loss_0=4.1285.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)