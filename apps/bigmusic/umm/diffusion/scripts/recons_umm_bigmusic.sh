


# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_wo_vc.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_vc2.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test_clip.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test_no_vc.lst
meta_lst=/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/assets/voice_condition_valsets/test2.lst
# meta_lst=/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/assets/samples_sstk_30s/test.lst



# ====== tokenizer version ====== 
# | tokenizer type | local path                                                                                                                                             | ckpt                                                                                                                                                                  |
# | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | Conformer mix  | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/Stage3_Multilingua_Music-TTS_MKii/step=0330000.ckpt                                             | hdfs:///home/byte_speech_sv/zongyu.yin/logs/umm_mix/umm_stage3_preclipped_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0330000.ckpt                   |
# | Conformer ZH   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/step=070000.ckpt                             | hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/checkpoints/step=070000.ckpt                                 |
# | Conv           | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/step=0160000.ckpt | hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0160000.ckpt |
# | Dual Conv V3   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0219_dualummv1_novocadv_1/step=0710000.ckpt                                                     | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0219_dualummv1_novocadv_1/checkpoints/step=0710000.ckpt                              |
# | Dual Conv V2   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0208_dualummv2_fullinp_1/step=0970000.ckpt                                                      | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0208_dualummv2_fullinp_1/checkpoints/step=0970000.ckpt                               |
# | Dual Conv V1   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0131_dualummv2_noref_2/step=0580000.ckpt                                                        | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0131_dualummv2_noref_2/checkpoints/step=0580000.ckpt                                 |





# Conformer ZH + Music 125hz
umm_ckpt_path=/mnt/bn/bigmusic-lf/user/weituo/inference_assets/zh_step=070000.ckpt
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1641_prompt_16xH800_0224/checkpoints/epoch=00-step=160000-loss=0.38.ckpt
cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConformer_125hzSS.yaml
sub_dir=exp1_160k
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConformer_125hzSS.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1641_prompt_16xH800_0224/checkpoints/epoch=00-step=160000-loss=0.38.ckpt
# sub_dir=exp1_160k
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1641_prompt_16xH800_0224/checkpoints/epoch=00-step=330000-loss=0.36.ckpt
# sub_dir=exp1_330k
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConformer_125hzSS_dim64.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1759_prompt_16xH800_0305/checkpoints/epoch=00-step=500000-loss=0.14.ckpt
# sub_dir=exp27_500k

# Conformer ZH + Music 40hz
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConformer_40hzSS.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1614_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.19.ckpt
# sub_dir=exp13_500k
# sub_dir=exp2_500k
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1738_prompt_16xH800_0304/checkpoints/epoch=00-step=470000-loss=0.18.ckpt
# sub_dir=exp25_470k

# DualConv + Music 125hz
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConv_125hzSS.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1609_prompt_16xH800_0223/checkpoints/epoch=00-step=430000-loss=0.36.ckpt
# sub_dir=exp4_430k
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1609_drop07_0228/checkpoints/epoch=00-step=500000-loss=0.36.ckpt
# sub_dir=exp18_500k
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1609_drop10_0228/checkpoints/epoch=00-step=420000-loss=0.36.ckpt
# sub_dir=exp17_420k
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1609_prompt_16xH800_0223/checkpoints/epoch=00-step=430000-loss=0.36.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConv_125hzSS.yaml
# sub_dir=exp4_430k
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConv_125hzSS_dim64.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1758_prompt_16xH800_0305/checkpoints/epoch=00-step=500000-loss=0.14.ckpt
# sub_dir=exp26_500k

# DualConv + Music 40hz
cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConv_40hzSS.yaml
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1588_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.18.ckpt
sub_dir=exp5_500k
# sub_dir=exp14_500k
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1737_prompt_16xH800_0304/checkpoints/epoch=00-step=260000-loss=0.17.ckpt
# sub_dir=exp24_260k
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1737_prompt_16xH800_0304/checkpoints/epoch=00-step=440000-loss=0.18.ckpt
# sub_dir=exp24_440k

# Conv + Music 125hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1617_prompt_16xH800_0223/checkpoints/epoch=00-step=430000-loss=0.36.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConv_125hzSS.yaml
# sub_dir=exp7_430k
# cfg_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1758_prompt_16xH800_0305/checkpoints/epoch=00-step=500000-loss=0.14.ckpt
# sub_dir=exp26_500k

# Conv + Music 40hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1611_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.20.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConv_40hzSS.yaml
# sub_dir=exp8_500k

# DualConvV1 + Music 125hz
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConvV1_125hzSS.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1646_prompt_16xH800_0227/checkpoints/epoch=00-step=910000-loss=0.35.ckpt
# sub_dir=exp15_910k


# DualConvV1 + Music 40hz
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConvV1_40hzSS.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1645_prompt_16xH800_0226/checkpoints/epoch=00-step=500000-loss=0.19.ckpt
# sub_dir=exp16_500k

# DualConvV3 + Music 125hz (24k_125hz_dim32_baseline)
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConvV3_125hzSS.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1833_prompt_16xH800_0308/checkpoints/epoch=00-step=470000-loss=0.36.ckpt
# sub_dir=exp28_470k

# DualConvV3 + Music 125hz (24k_125hz_dim64_sa)
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConvV3_125hzSS_dim64.yaml
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1834_prompt_16xH800_0308/checkpoints/epoch=00-step=500000-loss=0.13.ckpt
# sub_dir=exp29_500k


    # --run_opts.infer_type diffusion-vocoder \
    # --run_opts.infer_type vocoder \

# cleanup vocoder model in case of freq mismatch
rm ./.module_cache/soundstream*
# rm ./data/tmp/*.wav
out_dir=/mnt/bn/data-storage-hl/user/zhangshuo/data/tmp/20240311/${sub_dir}
mkdir -p $out_dir


bash launch.sh predict \
	-c $cfg_path \
	--run_opts.meta_lst $meta_lst \
	--run_opts.umm_ckpt_path $umm_ckpt_path \
	--run_opts.output_dir $out_dir \
	--run_opts.diffusion_ckpt_path $diffusion_ckpt_path \
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
	--run_opts.infer_type diffusion-vocoder \
	--bn_config.wav_norm True \
	--pl_module.text_cfg_w 0

