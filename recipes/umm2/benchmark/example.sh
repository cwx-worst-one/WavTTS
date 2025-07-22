model_name="debug2"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/VQ/RP_50Hz_1x32768_lr1e-4_100k_bs4.5/checkpoints/step=0300000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/VQ/25Hz_1x32768_window2.5-2.5sec_stage3/checkpoints/step=0150000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_lr3e-5_bs4_VQ_baseline/checkpoints/step=0090000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/50Hz_lr1e-4_cyc200k_bs5/checkpoints/step=0175000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/VQ/RP_50Hz_1x32768_lr1e-4_100k_bs4.5/checkpoints/step=0220000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_lr3e-5_bs4p5_RP-VQ_CTC0p2/checkpoints/step=0050000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz/checkpoints/step=0220000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/VQ/VQ_entropy_1x32768_lr3e-5_30k_pitchpdt_re2/checkpoints/step=0200000.ckpt"

# model_name="stage1to3_VQ_RP_50Hz_CTC0.2_1x32768_240k"
# stage2
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/50Hz_lr1e-4_cyc200k_bs5/checkpoints/step=0175000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/50Hz_lr2e-4_cyc200k_bs5_fixall/checkpoints/step=0065000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/100Hz_lr2e-4_bs2p4_cyc200k_CTC25Hz_normhead/checkpoints/step=0120000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/25Hz_lr2e-4_cyc200k_bs8_otf_5-60s/checkpoints/step=0105000.ckpt"
# stage3 
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_lr3e-5_bs4p5_RP-VQ_CTC0p2/checkpoints/step=0240000.ckpt"
# # model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz/checkpoints/step=0220000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz_base2w_reweight/checkpoints/step=0135000.ckpt"

model_name="stage4"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/VQEntropy_15bit_50Hz_lr3e-5_cyc30k_bs4p5_CTC25hz/checkpoints/step=0235000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz_base2w_reweight/checkpoints/step=0220000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/VQEntropy_15bit_25Hz_otf_data/checkpoints/step=0050000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz/checkpoints/step=0220000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/VQ/VQ_entropy_1x32768_lr3e-5_30k_pitchpdt_re2/checkpoints/step=0220000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/VQEntropy_15bit_50Hz_lr3e-5_cyc30k_bs4p5_CTC25hz_reweight/checkpoints/step=0195000.ckpt"
# model_path="hdfs://haruna/home/byte_speech_sv/ju-chiang.wang/umm_tag/karaoke/umm_stage4_artist_weight_mel_chroma_ctc_vq/checkpoints/step=0320000.ckpt"

model_cls="recipes.umm2.modules.stages.stage3.Stage3RVQ"
out_dir="/mnt/bn/music-llm-nas-lq/qinxin/outputs/tokenizer_eval/"
nsample=20
slice_modes='even'
slice_dur=60

model_name="debug"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/VQEntropy_15bit_25Hz_otf_data_shuffle800_fix_RP/checkpoints/step=0220000.ckpt"
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/VQ/25Hz_1x32768_window2.5-2.5sec_stage3/checkpoints/step=0220000.ckpt"


tasks="loss,locality"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --slice_modes $slice_modes --slice_dur $slice_dur


tasks="ctc_wer"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --slice_modes $slice_modes --slice_dur $slice_dur


# batch 0, length=146.01725, avg_length=146.01725, no_lyric_flag=False
# 0 {'loss_rvq': tensor([0.0137], device='cuda:0'), 'loss_mel': tensor(0.5556, device='cuda:0'), 'loss_ctc': tensor(15.8841, device='cuda:0'), 'loss_chroma': tensor(0.3894, device='cuda:0'), 'loss_f0_vuv': tensor(0.1546, device='cuda:0')}
# 1 {'loss_rvq': tensor([0.0141], device='cuda:0'), 'loss_mel': tensor(0.5530, device='cuda:0'), 'loss_ctc': tensor(14.2160, device='cuda:0'), 'loss_chroma': tensor(0.3885, device='cuda:0'), 'loss_f0_vuv': tensor(0.2399, device='cuda:0')}
# 2 {'loss_rvq': tensor([0.0140], device='cuda:0'), 'loss_mel': tensor(0.5550, device='cuda:0'), 'loss_ctc': tensor(15.1300, device='cuda:0'), 'loss_chroma': tensor(0.3787, device='cuda:0'), 'loss_f0_vuv': tensor(0.1405, device='cuda:0')}
# 3 {'loss_rvq': tensor([0.0127], device='cuda:0'), 'loss_mel': tensor(0.5360, device='cuda:0'), 'loss_ctc': tensor(0., device='cuda:0'), 'loss_chroma': tensor(0.3211, device='cuda:0'), 'loss_f0_vuv': tensor(0.0894, device='cuda:0')}